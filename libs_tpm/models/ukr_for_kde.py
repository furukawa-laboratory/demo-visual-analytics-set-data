import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from libs_tpm.models.kde import KDE
from somf.libs.tools.create_zeta import create_zeta


class UKRForWeightedKDE():
    def __init__(self, weight_of_group, member_features, n_embedding,
                 bandwidth_kde, bandwidth_nadaraya,
                 is_compact, lambda_, metric_evaluation_method,
                 metric='kl',
                 resolution_quadrature=None,
                 init='random', is_save_history=True, random_state=None):
        if weight_of_group.shape[1] != member_features.shape[0]:
            raise ValueError('group_features.shape[1] and member_features[0] must be match.')
        if np.any(weight_of_group < 0.0):
            raise ValueError('weight_of_group bust be non-negative')

        # set input
        self.normalized_weight_of_group = weight_of_group / weight_of_group.sum(axis=1)[:, None]
        self.member_features = member_features
        self.n_groups = weight_of_group.shape[0]
        self.n_members = member_features.shape[0]
        self.n_features = member_features.shape[1]
        self.bandwidth_kde = bandwidth_kde
        self.bandwidth_nw = bandwidth_nadaraya
        self.precision_kde = 1 / (bandwidth_kde * bandwidth_kde)
        self.precision_nadaraya = 1 / (bandwidth_nadaraya * bandwidth_nadaraya)
        self.n_embedding = n_embedding
        self.is_compact = is_compact
        self.lambda_ = lambda_

        # exception handling
        if metric in ['kl', 'l2']:
            self.metric = metric
        else:
            raise ValueError('invalid metric={}'.format(metric))

        if metric_evaluation_method in ['jensen', 'quadrature_by_parts']:
            self.metric_evaluation_method = metric_evaluation_method
            if metric_evaluation_method == 'jensen' and metric == 'l2':
                raise ValueError("Set metric l2, metric_evaluation_method must be 'quadrature_by_parts'")
            if metric_evaluation_method == 'quadrature_by_parts':
                if isinstance(resolution_quadrature, int) and resolution_quadrature > 0:
                    self.resolution_quadrature = resolution_quadrature
                else:
                    raise ValueError(
                        'if kl divergence is evaluated using quadrature by parts, resolution_quadrature must be set integer')
        else:
            raise ValueError('{} is not supported to evaluate kl divergence'.format(metric_evaluation_method))

        # if kl divergence is evaluated by 'quadrature_by_parts', calculate pdf by kernel density estimation
        if self.metric_evaluation_method == 'quadrature_by_parts':
            self.quadrature_points, self.step = create_zeta(
                zeta_min=self.member_features.min() - 2.0 * self.bandwidth_kde,
                zeta_max=self.member_features.max() + 2.0 * self.bandwidth_kde,
                latent_dim=self.n_features,
                resolution=self.resolution_quadrature,
                include_min_max=False, return_step=True)
            kde = KDE()
            kde.fit(dataset=self.member_features,
                    bandwidth=self.bandwidth_kde,
                    weights=self.normalized_weight_of_group)
            self.data_densities = kde.pdf(self.quadrature_points)

        self.Z = None
        if isinstance(init, str) and init in 'random':
            from sklearn.utils import check_random_state
            random_state = check_random_state(random_state)
            self.Z = random_state.normal(0, self.bandwidth_nw * 0.1, (self.n_groups, self.n_embedding))
        elif isinstance(init, str) and 'svd' in init:
            if init == 'normalize_svd':
                u, s, v = np.linalg.svd(self.normalized_weight_of_group)
            elif init == 'non_normalize_svd':
                u, s, v = np.linalg.svd(weight_of_group)
            else:
                raise ValueError('invalid init: {}'.format(init))
            self.Z = u[:, :self.n_embedding] * self.bandwidth_nw * 0.5
        elif isinstance(init, str) and init == 'pca_densities':
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            pca = PCA(n_components=self.n_embedding, random_state=random_state)
            # shift mean to zero
            standardizer = StandardScaler()
            data = standardizer.fit_transform(self.data_densities)
            init_value = pca.fit_transform(data)
            init_value = init_value / np.std(init_value[:, 0])
            self.Z = init_value * self.bandwidth_nw * 0.1
        elif isinstance(init, np.ndarray) and init.shape == (self.n_groups, self.n_embedding):
            # 外部から形が合っている配列が渡された場合
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.is_save_history = is_save_history
        if self.is_save_history:
            self.history = {}
        self.is_done_fit = False

    def fit(self, n_epoch, learning_rate, verbose=True):
        # prepare dictionary of history
        if self.is_save_history:
            if self.is_done_fit:
                self.history['z'] = np.concatenate([self.history['z'],
                                                    np.zeros((n_epoch,
                                                              self.n_groups,
                                                              self.n_embedding))], axis=0)
                self.history['zvar'] = np.concatenate([self.history['zvar'],
                                                       np.zeros((n_epoch,
                                                                 self.n_embedding))], axis=0)
                self.history['obj_func'] = np.concatenate([self.history['obj_func'],
                                                           np.zeros(n_epoch)])
                self.history['learning_rate'] = np.concatenate([self.history['learning_rate'],
                                                                np.ones(n_epoch) * learning_rate])
            else:
                self.history['z'] = np.zeros((n_epoch, self.n_groups, self.n_embedding))
                self.history['zvar'] = np.zeros((n_epoch, self.n_embedding))
                self.history['obj_func'] = np.zeros(n_epoch)
                self.history['learning_rate'] = np.ones(n_epoch) * learning_rate

        # save cumulative number of epoch
        if self.is_done_fit:
            last_n_epoch = self.nb_epoch
        else:
            last_n_epoch = 0
        self.nb_epoch = n_epoch + last_n_epoch

        if self.metric == 'kl':
            if self.metric_evaluation_method == 'jensen':
                all_members_distance = cdist(self.member_features, self.member_features, "sqeuclidean")
                cross_entropy_gaussians = self.precision_kde * all_members_distance + \
                                          self.n_features * (
                                                  1.0 + np.log(self.bandwidth_kde ** 2.0) + np.log(2 * np.pi))
                cross_entropy_gaussians = cross_entropy_gaussians * 0.5
                self.upper_cross_entropy_kdes = np.einsum("tn,im,nm->ti",
                                                          self.normalized_weight_of_group,
                                                          self.normalized_weight_of_group,
                                                          cross_entropy_gaussians)

            if verbose:
                bar = tqdm(range(n_epoch))
            else:
                bar = range(n_epoch)

            for epoch in bar:

                dFdZ, obj_func = self.grad(return_obj_func_value=True)
                self.Z -= learning_rate * dFdZ
                if self.is_compact:
                    self.Z = np.clip(self.Z, -1.0, 1.0)
                else:
                    self.Z -= self.Z.mean(axis=0)

                if self.is_save_history:
                    self.history['z'][last_n_epoch + epoch] = self.Z
                    self.history['zvar'][last_n_epoch + epoch] = np.mean(np.square(self.Z - self.Z.mean(axis=0)),
                                                                         axis=0)
                    self.history['obj_func'][last_n_epoch + epoch] = obj_func
        elif self.metric == 'l2':
            from somf.libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as RUKR
            regular_ukr = RUKR(X=self.data_densities, n_components=self.n_embedding,
                               bandwidth_gaussian_kernel=self.bandwidth_nw, is_compact=self.is_compact,
                               lambda_=self.lambda_, init=self.Z.copy(), is_loocv=False,
                               is_save_history=self.is_save_history)
            regular_ukr.fit(nb_epoch=n_epoch, verbose=verbose, eta=learning_rate)
            self.Z = regular_ukr.Z.copy()
            if self.is_save_history:
                for query in ['z', 'zvar', 'obj_func']:
                    self.history[query][last_n_epoch:] = regular_ukr.history[query]
        else:
            raise ValueError('invalid metric={}'.format(self.metric))

        self.is_done_fit = True
        if self.is_save_history:
            return self.history

    def grad(self, return_obj_func_value=False):
        if self.metric == 'kl':
            pass
        else:
            raise ValueError("To call grad method, metric must be 'kl'")

        DeltaZZ = self.Z[:, None, :] - self.Z[None, :, :]

        DistZZ = np.sum(np.square(DeltaZZ), axis=2)
        H = np.exp(-0.5 * self.precision_nadaraya * DistZZ)
        G = np.sum(H, axis=1)[:, None]
        GInv = 1 / G
        R = H * GInv

        if self.metric_evaluation_method == 'jensen':
            obj_func = np.sum(R * self.upper_cross_entropy_kdes) + self.lambda_ * np.sum(np.square(self.Z))

            weighted_mean_uce = np.sum(R * self.upper_cross_entropy_kdes, axis=1)[:, None]

            A = R * (weighted_mean_uce - self.upper_cross_entropy_kdes)
        elif self.metric_evaluation_method == 'quadrature_by_parts':
            approximated_densities = R @ self.data_densities
            obj_func = -np.sum(self.data_densities * np.log(approximated_densities)) * (
                    self.step ** self.n_features)
            ratio_data_and_approx = self.data_densities / approximated_densities
            similarities = np.einsum("nk,ik->ni", ratio_data_and_approx, self.data_densities) * (
                    self.step ** self.n_features)
            weighted_mean_similarities = np.sum(R * similarities, axis=1)[:, None]
            A = -R * (weighted_mean_similarities - similarities)
        dFdZ = self.precision_nadaraya * np.sum((A + A.T)[:, :, None] * DeltaZZ, axis=1)
        dFdZ += 2.0 * self.lambda_ * self.Z

        if return_obj_func_value:
            return dFdZ, obj_func
        else:
            return dFdZ

    def set_z(self, Z: np.ndarray):
        if isinstance(Z, np.ndarray):
            self.Z = Z.copy()
        else:
            raise ValueError('invalid Z={}'.format(Z))

    def transform(self, weights_of_group, n_epoch, learning_rate, verbose=True):
        if self.metric == 'kl' and self.metric_evaluation_method == 'quadrature_by_parts':
            pass
        else:
            raise ValueError('Not implementation yet metric={}'.format(self.metric))
        if self.is_save_history:
            self.history['znew'] = np.zeros((n_epoch,
                                             weights_of_group.shape[0],
                                             self.n_embedding))
            self.history['obj_func_znew'] = np.zeros(n_epoch)
        from libs_tpm.models.kde import KDE
        kde = KDE()
        kde.fit(dataset=self.member_features,
                bandwidth=self.bandwidth_kde,
                weights=weights_of_group)
        target_densities = kde.pdf(x=self.quadrature_points)
        return self.transform_from_densities(target_densities=target_densities,
                                             n_epoch=n_epoch,
                                             learning_rate=learning_rate,
                                             verbose=verbose)

    def transform_from_datasets(self, datasets, weights, n_epoch, learning_rate, verbose=True):
        if self.metric == 'kl' and self.metric_evaluation_method == 'quadrature_by_parts':
            pass
        else:
            raise ValueError('Not implementation yet metric={}'.format(self.metric))
        if self.is_save_history:
            self.history['znew'] = np.zeros((n_epoch,
                                             datasets.shape[0],
                                             self.n_embedding))
            self.history['obj_func_znew'] = np.zeros(n_epoch)

        target_densities = np.empty([0, self.data_densities.shape[1]])
        if datasets.ndim == 2:
            datasets = np.expand_dims(datasets, axis=0)

        for dataset, weight in zip(datasets, weights):
            kde = KDE()
            kde.fit(dataset=dataset, bandwidth=self.bandwidth_kde, weights=weight)
            density = kde.pdf(x=self.quadrature_points)
            target_densities = np.concatenate([target_densities, density[None, :]], axis=0)

        return self.transform_from_densities(target_densities,
                                             n_epoch=n_epoch,
                                             learning_rate=learning_rate,
                                             verbose=verbose)


    def transform_from_densities(self, target_densities, n_epoch, learning_rate, verbose=True):
        # calcualte nearest data density to initialize
        if target_densities.shape[1] != self.data_densities.shape[1]:
            raise ValueError('invalid target_densities.shape[1]={}'.format(target_densities.shape[1]))
        cross_entropy = -np.sum(target_densities[:, None, :] * np.log(self.data_densities)[None, :, :],
                                axis=2)
        nearest_data = cross_entropy.argmin(axis=1)
        Znew = self.Z[nearest_data, :].copy()

        if verbose:
            bar = tqdm(range(n_epoch))
        else:
            bar = range(n_epoch)

        for epoch in bar:
            Delta = Znew[:, None, :] - self.Z[None, :, :]
            sqdist = np.sum(np.square(Delta), axis=2)
            H = np.exp(-0.5 * self.precision_nadaraya * sqdist)
            G = H.sum(axis=1)[:, None]
            GInv = np.reciprocal(G)
            R = H * GInv
            model_densities = R @ self.data_densities
            ratios_density = target_densities / model_densities
            obj_func = -np.sum(target_densities * np.log(model_densities))
            obj_func *= (self.step ** self.n_embedding)

            Deltabar = np.einsum('kt,ktl->kl', R, Delta)
            product_ratio_density = np.einsum("kd,td->kt",
                                              ratios_density,
                                              self.data_densities)
            smoothed_product = R * product_ratio_density
            dCEdZnew = np.einsum('kt,ktl->kl',
                                 smoothed_product,
                                 Deltabar[:, None, :] - Delta)
            # dCEdZnew = np.einsum("kt,kd,td,ktl->kl",
            #                      R,
            #                      ratios_density,
            #                      self.data_densities,
            #                      Deltabar[:, None, :] - Delta)
            dCEdZnew *= -self.precision_nadaraya * (self.step ** self.n_embedding)

            Znew -= learning_rate * dCEdZnew
            if self.is_compact:
                Znew = np.clip(Znew, -1.0, 1.0)
            else:
                pass

            if self.is_save_history:
                self.history['znew'][epoch] = Znew
                self.history['obj_func_znew'][epoch] = obj_func

        return Znew

    def inverse_transform(self, latent_variables):
        raise ValueError(
            'One latent variable is mapped to one density function which is infinte dimension. Use inverse_transformed_pdf with any point x in space of random variable')

    def inverse_transformed_pdf(self, x, latent_variables):

        if x.ndim == 1:
            if x.shape[0] != self.n_features:
                raise ValueError('x.shape[0] must be equal n_features={}'.format(self.n_features))
            else:
                x = x.reshape(-1, self.n_features)
        elif x.ndim == 2:
            if x.shape[1] != self.n_features:
                raise ValueError('x.shape[1] must be equal n_features={}'.format(self.n_features))
        else:
            raise ValueError('x must be 1 or 2 dimensional array')

        if latent_variables.ndim == 1:
            if latent_variables.shape[0] != self.n_embedding:
                raise ValueError('latent_variable.shape[0] must be equal n_embedding={}'.format(self.n_embedding))
            else:
                latent_variables = latent_variables.reshape(-1, self.n_embedding)
        elif latent_variables.ndim == 2:
            if latent_variables.shape[1] != self.n_embedding:
                raise ValueError('latent_variable.shape[1] must be equal n_embedding={}'.format(self.n_embedding))
        else:
            raise ValueError('latent variables must be 1 or 2 dimensional array')

        # If x.shape=(K,L),latent_variables.shape=(M,L)
        # self.Z.shape=(T,L), self.member_feature.shape=(N,D)
        sqdist = cdist(latent_variables, self.Z, metric='sqeuclidean')  # MxT
        H = np.exp(-0.5 * self.precision_nadaraya * sqdist)  # MxT
        G = H.sum(axis=1)[:, None]  # Mx1
        GInv = np.reciprocal(G)  # Mx1
        R = H * GInv  # MxT
        smoothed_weight = R @ self.normalized_weight_of_group  # MxT @ TxN = MxN

        kde = KDE()
        kde.fit(dataset=self.member_features,
                bandwidth=self.bandwidth_kde,
                weights=smoothed_weight)
        return kde.pdf(x)  # MxK

    def visualize(self, n_grid_points=30, label_groups=None,
                  is_latent_space_middle_color_zero=False,
                  is_show_all_label_groups=False,
                  is_show_ticks=True,
                  params_imshow_data_space=None,
                  params_imshow_latent_space=None,
                  params_scatter_data_space=None,
                  params_scatter_latent_space=None,
                  title_latent_space=None, title_data_space=None,
                  fig=None, fig_size=None, ax_latent_space=None, ax_data_space=None):

        # import necessary library to draw
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        self._initialize_to_visualize(n_grid_points=n_grid_points,
                                      params_imshow_data_space=params_imshow_data_space,
                                      params_imshow_latent_space=params_imshow_latent_space,
                                      params_scatter_data_space=params_scatter_data_space,
                                      params_scatter_latent_space=params_scatter_latent_space,
                                      label_groups=label_groups,
                                      title_latent_space=title_latent_space,
                                      title_data_space=title_data_space,
                                      fig=fig,
                                      fig_size=fig_size,
                                      ax_latent_space=ax_latent_space,
                                      ax_data_space=ax_data_space,
                                      is_latent_space_middle_color_zero=is_latent_space_middle_color_zero,
                                      is_show_all_label_groups=is_show_all_label_groups)

        self._draw_latent_space()
        self._draw_data_space()

        # connect figure and method defining action when latent space is clicked
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)
        if self.label_groups is not None and self.is_show_all_label_groups is False:
            self.fig.canvas.mpl_connect('motion_notify_event', self.__mouse_over_fig)
        plt.show()

    def __onclick_fig(self, event):
        self.is_initial_view = False
        if event.xdata is not None:
            # クリックされた座標の取得
            click_coordinates = np.array([event.xdata, event.ydata])
            if event.inaxes == self.ax_latent_space.axes:
                self._set_data_space_from_latent_space(click_coordinates)
                self._draw_latent_space()
                self._draw_data_space()
            elif event.inaxes == self.ax_data_space.axes:
                self._set_latent_space_from_data_space(click_coordinates)
                self._draw_latent_space()
                self._draw_data_space()

    def __mouse_over_fig(self, event):
        if event.xdata is not None:
            # クリックされた座標の取得
            over_coordinates = np.array([event.xdata, event.ydata])
            if event.inaxes == self.ax_latent_space.axes:
                self._set_shown_label_in_latent_space(over_coordinates)
                self._draw_latent_space()
                self._draw_data_space()

            elif event.inaxes == self.ax_data_space:
                pass

    def _initialize_to_visualize(self, n_grid_points, params_imshow_data_space, params_imshow_latent_space,
                                 params_scatter_data_space, params_scatter_latent_space,
                                 label_groups, title_latent_space, title_data_space, fig, fig_size,
                                 ax_latent_space, ax_data_space, is_latent_space_middle_color_zero,
                                 is_show_all_label_groups, is_show_ticks):
        # invalid check
        if self.n_embedding != 2 or self.n_features != 2:
            raise ValueError('Now support only n_embedding = 2 and n_features = 2')

        if isinstance(n_grid_points, int):
            # 代表点の数を潜在空間の次元ごとに格納
            self.n_grid_points_latent_space = np.ones(self.n_embedding, dtype='int8') * n_grid_points
            self.n_grid_points_data_space = np.ones(self.n_features, dtype='int8') * n_grid_points
        else:
            raise ValueError('Only support n_grid_points is int')
        if self.is_compact:
            self.grid_points_latent_space = create_zeta(-1.0, 1.0, self.n_embedding, n_grid_points)
        else:
            raise ValueError('Not support is_compact=False')  # create_zetaの整備が必要なので実装は後で
        self.grid_points_data_space = create_zeta(self.member_features.min(), self.member_features.max(),
                                                  self.n_features, n_grid_points)
        self.threshold_radius_show = np.abs(self.grid_points_latent_space.max()
                                            - self.grid_points_latent_space.min()) * 0.05

        if params_imshow_data_space is None:
            self.params_imshow_data_space = {}
        elif isinstance(params_imshow_data_space, dict):
            self.params_imshow_data_space = params_imshow_data_space
        else:
            raise ValueError('invalid params_imshow_data_space={}'.format(params_imshow_data_space))

        if params_imshow_latent_space is None:
            self.params_imshow_latent_space = {}
        elif isinstance(params_imshow_data_space, dict):
            self.params_imshow_latent_space = params_imshow_latent_space
        else:
            raise ValueError('invalid params_imshow_latent_space={}'.format(params_imshow_data_space))

        if params_scatter_latent_space is None:
            self.params_scatter_latent_space = {'s': 7}
        elif isinstance(params_scatter_latent_space, dict):
            self.params_scatter_latent_space = params_scatter_latent_space
        else:
            raise ValueError('invalid params_scatter_latent_space={}'.format(params_scatter_latent_space))

        if params_scatter_data_space is None:
            self.params_scatter_data_space = {'s': 10}
        elif isinstance(params_scatter_data_space, dict):
            self.params_scatter_data_space = params_scatter_data_space
        else:
            raise ValueError('invalid params_scatter_data_space={}'.format(params_scatter_data_space))

        if label_groups is None:
            self.label_groups = label_groups
        elif isinstance(label_groups, list):
            self.label_groups = label_groups
        elif isinstance(label_groups, np.ndarray):
            if np.squeeze(label_groups).ndim == 1:
                self.label_groups = np.squeeze(label_groups)
            else:
                raise ValueError('label_groups must be 1d array')
        else:
            raise ValueError('label_groups must be 1d array or list')

        if title_latent_space is None:
            self.title_latent_space = 'Latent space'
        else:
            self.title_latent_space = title_latent_space

        if title_data_space is None:
            self.title_data_space = 'Data space'
        else:
            self.title_data_space = title_data_space

        if fig is None:
            import matplotlib.pyplot as plt
            if fig_size is None:
                self.fig = plt.figure(figsize=(15, 6))
            else:
                self.fig = plt.figure(figsize=fig_size)
        else:
            self.fig = fig

        if ax_latent_space is None and ax_data_space is None:
            self.ax_latent_space = self.fig.add_subplot(1, 2, 1, aspect='equal')
            self.ax_data_space = self.fig.add_subplot(1, 2, 2)
        else:
            self.ax_latent_space = ax_latent_space
            self.ax_data_space = ax_data_space

        self.is_latent_space_middle_color_zero = is_latent_space_middle_color_zero
        self.is_show_all_label_groups = is_show_all_label_groups
        self.is_show_ticks = is_show_ticks

        self.click_coordinates_latent_space = None
        self.click_coordinates_data_space = None
        self.is_initial_view = True
        self.grid_values_to_draw_data_space = None
        self.grid_values_to_draw_latent_space = None
        self.index_data_label_shown = None
        self.used_bright_range = [0.0, 1.0]
        self.mask_shown_member = np.full(self.n_members, True, bool)
        self.is_select_latent_variable = False
        self.mesh_in_latent_space = None
        self.comment_latent_space = None

        # shape=(grid_points**self.n_embedding, self.resolution_quadrature**self.n_feature)
        # 要するに潜在空間の離散化数xデータの空間の離散化数
        self.grid_mapping = self.inverse_transformed_pdf(x=self.grid_points_data_space,
                                                         latent_variables=self.grid_points_latent_space)

        epsilon = 0.03 * np.abs(self.grid_points_latent_space.max() - self.grid_points_latent_space.min())
        self.noise_label = epsilon * (np.random.rand(self.n_groups, self.n_embedding) * 2.0 - 1.0)

    def _set_data_space_from_latent_space(self, click_coordinates):
        if self.click_coordinates_latent_space is not None:
            # If a coodinates are clicked previously
            _, dist_previous_click_coordinates = self.__calc_nearest_candidate(click_coordinates,
                                                                               self.click_coordinates_latent_space.reshape(1, -1),
                                                                               retdist=True)
            epsilon = 0.02 * np.abs(self.grid_points_latent_space.max() - self.grid_points_latent_space.min())
            if dist_previous_click_coordinates < epsilon:
                self.click_coordinates_latent_space = None
                self.grid_values_to_draw_data_space = None
                self.mask_shown_member = np.full(self.n_members, True, bool)
                is_unconditioning = True
            else:
                is_unconditioning = False
        else:
            # If no coordinates are clicked
            is_unconditioning = False

        if is_unconditioning:
            pass
        else:
            index_nearest_latent_variable, dist_min_z = self.__calc_nearest_candidate(click_coordinates,
                                                                                      self.Z,
                                                                                      retdist=True)
            index_nearest_grid_point, dist_min_grid = self.__calc_nearest_candidate(click_coordinates,
                                                                                    self.grid_points_latent_space,
                                                                                    retdist=True)
            if dist_min_z < dist_min_grid:
                bag_of_members = self.normalized_weight_of_group[index_nearest_latent_variable]
                self.mask_shown_member = bag_of_members != 0.0
                self.bag_of_shown_member = bag_of_members[self.mask_shown_member]
                kde = KDE()
                kde.fit(dataset=self.member_features, bandwidth=self.bandwidth_kde, weights=bag_of_members)
                self.grid_values_to_draw_data_space = kde.pdf(self.grid_points_data_space)
                self.click_coordinates_latent_space = self.Z[index_nearest_latent_variable]
                self.is_select_latent_variable = True
                self.index_team_selected = index_nearest_latent_variable
                print('clicked team is {}'.format(self.index_team_selected))
            else:
                self.mask_shown_member = np.full(self.n_members, True, bool)
                self.bag_of_shown_member = None
                self.grid_values_to_draw_data_space = self.grid_mapping[index_nearest_grid_point]
                self.click_coordinates_latent_space = self.grid_points_latent_space[index_nearest_grid_point]
                self.is_select_latent_variable = False
                self.index_team_selected = None

    def _set_latent_space_from_data_space(self, click_coordinates):
        if self.click_coordinates_data_space is not None:
            # If a coodinates are clicked previously
            _, dist_previous_click_coordinates = self.__calc_nearest_candidate(click_coordinates,
                                                                               self.click_coordinates_data_space.reshape(1, -1),
                                                                               retdist=True)
            epsilon = 0.02 * np.abs(self.grid_points_data_space.max() - self.grid_points_data_space.min())
            if dist_previous_click_coordinates < epsilon:
                self.click_coordinates_data_space = None
                self.grid_values_to_draw_latent_space = None
                is_unconditioning = True
            else:
                is_unconditioning = False
        else:
            # If no coordinates are clicked
            is_unconditioning = False
        if is_unconditioning:
            pass
        else:
            index_nearest_grid_point, dist_min_grid = self.__calc_nearest_candidate(click_coordinates,
                                                                                          self.grid_points_data_space,
                                                                                          retdist=True)
            self.grid_values_to_draw_latent_space = self.grid_mapping[:, index_nearest_grid_point]
            self.click_coordinates_data_space = self.grid_points_data_space[index_nearest_grid_point]

            self.comment_latent_space = 'Density of clicked member'

    def get_grid_values_to_draw_data_space(self):
        return self.grid_values_to_draw_data_space

    def set_params_imshow_latent_space(self, params: dict):
        if isinstance(params, dict):
            self.params_imshow_latent_space.update(params)
        else:
            raise ValueError('invalid params={}')

    def set_params_imshow_data_space(self, params: dict):
        if isinstance(params, dict):
            self.params_imshow_data_space.update(params)
        else:
            raise ValueError('invalid params={}')

    def set_grid_values_to_draw_latent_space(self, grid_values):
        self.grid_values_to_draw_latent_space = grid_values

    def _set_shown_label_in_latent_space(self, click_coordinates):
        index, dist = self.__calc_nearest_candidate(click_coordinates,
                                                    candidates=self.Z,
                                                    retdist=True)
        if dist <= self.threshold_radius_show:
            self.index_data_label_shown = index
        else:
            self.index_data_label_shown = None

    def set_title_latent_space(self, title):
        self.title_latent_space = title

    def set_title_data_space(self, title):
        self.title_data_space = title

    def set_used_bright_range(self, used_bright_range):
        self.used_bright_range = used_bright_range

    def set_mesh_in_latent_space(self, mesh: np.ndarray):
        self.mesh_in_latent_space = mesh

    def set_comment_in_latent_space(self, comment: str):
        self.comment_latent_space = comment

    def _draw_latent_space(self):
        import matplotlib.pyplot as plt
        from matplotlib import patheffects as path_effects
        self.ax_latent_space.cla()
        self.ax_latent_space.set_title(self.title_latent_space)
        self.ax_latent_space.set_xlim(self.Z[:, 0].min() * 1.05, self.Z[:, 0].max() * 1.05)
        self.ax_latent_space.set_ylim(self.Z[:, 1].min() * 1.05, self.Z[:, 1].max() * 1.05)
        if self.grid_values_to_draw_latent_space is not None:
            if self.grid_values_to_draw_latent_space.ndim == 2:
                if self.grid_values_to_draw_latent_space.shape[1] == 2:
                    from libs_tpm.tools.cmap2d_value_and_brightness import cmap2d_base_and_brightness
                    if self.is_latent_space_middle_color_zero:
                        max_grid_value = self.grid_values_to_draw_latent_space[0].max()
                        min_grid_value = self.grid_values_to_draw_latent_space[0].min()
                        vmin = -max(abs(max_grid_value), abs(min_grid_value))
                        vmax = max(abs(max_grid_value), abs(min_grid_value))
                    else:
                        vmin = None
                        vmax = None
                    if 'cmap' in self.params_imshow_latent_space.keys():
                        base_cmap = self.params_imshow_latent_space['cmap']
                    else:
                        base_cmap = None
                    grid_values_to_imshow = cmap2d_base_and_brightness(
                        value=self.grid_values_to_draw_latent_space[:, 0],
                        brightness=self.grid_values_to_draw_latent_space[:, 1],
                        base_cmap=base_cmap,
                        vmin=vmin, vmax=vmax,
                        used_bright_range=self.used_bright_range)
                    grid_values_to_contour = self.grid_values_to_draw_latent_space[:, 0]
                else:
                    if self.is_latent_space_middle_color_zero:
                        max_grid_value = self.grid_values_to_draw_latent_space.max()
                        min_grid_value = self.grid_values_to_draw_latent_space.min()
                        vmin = -max(abs(max_grid_value), abs(min_grid_value))
                        vmax = max(abs(max_grid_value), abs(min_grid_value))
                    else:
                        vmin = None
                        vmax = None
                    raise ValueError('invalid grid_value shape={}'.format(self.grid_values_to_draw_latent_space.shape))
            elif self.grid_values_to_draw_latent_space.ndim == 1:
                grid_values_to_imshow = self.grid_values_to_draw_latent_space
                grid_values_to_contour = self.grid_values_to_draw_latent_space
                vmin = None
                vmax = None
            else:
                raise ValueError('invalid grid_value ndim={}'.format(self.grid_values_to_draw_latent_space.ndim))
            grid_points_3d = self.__unflatten_grid_array(self.grid_points_latent_space,
                                                         self.n_grid_points_latent_space)
            grid_values_to_imshow_3d = self.__unflatten_grid_array(grid_values_to_imshow,
                                                                   self.n_grid_points_latent_space)
            grid_values_to_contour_3d = self.__unflatten_grid_array(grid_values_to_contour,
                                                                    self.n_grid_points_data_space)
            # set coordinate of axis
            any_index = 0
            if grid_points_3d[any_index, 0, 0] < grid_points_3d[any_index, -1, 0]:
                coordinate_ax_left = grid_points_3d[any_index, 0, 0]
                coordinate_ax_right = grid_points_3d[any_index, -1, 0]
            else:
                coordinate_ax_left = grid_points_3d[any_index, -1, 0]
                coordinate_ax_right = grid_points_3d[any_index, 0, 0]
                grid_values_to_imshow_3d = np.flip(grid_values_to_imshow_3d, axis=1).copy()

            if grid_points_3d[-1, any_index, 1] < grid_points_3d[0, any_index, 1]:
                coordinate_ax_bottom = grid_points_3d[-1, any_index, 1]
                coordinate_ax_top = grid_points_3d[0, any_index, 1]
            else:
                coordinate_ax_bottom = grid_points_3d[0, any_index, 1]
                coordinate_ax_top = grid_points_3d[-1, any_index, 1]
                grid_values_to_imshow_3d = np.flip(grid_values_to_imshow_3d, axis=0).copy()

            # extent = [left, right, bottom, top]
            self.ax_latent_space.imshow(grid_values_to_imshow_3d,
                                        extent=[coordinate_ax_left,
                                                coordinate_ax_right,
                                                coordinate_ax_bottom,
                                                coordinate_ax_top
                                                ],
                                        vmin=vmin,
                                        vmax=vmax,
                                        **self.params_imshow_latent_space
                                        )
            ctr = self.ax_latent_space.contour(grid_points_3d[:, :, 0],
                                               grid_points_3d[:, :, 1],
                                               grid_values_to_contour_3d, 6, colors='k',
                                               vmin=vmin,
                                               vmax=vmax)
            plt.setp(ctr.collections, path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                                                    path_effects.Normal()])
            clbls = self.ax_latent_space.clabel(ctr)
            plt.setp(clbls, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'),
                                          path_effects.Normal()])
        self.ax_latent_space.scatter(self.Z[:, 0], self.Z[:, 1], **self.params_scatter_latent_space)

        # Write label
        if self.label_groups is None:
            pass
        else:
            if self.is_show_all_label_groups:
                for z, noise, label in zip(self.Z, self.noise_label, self.label_groups):
                    point_label = z + noise
                    text = self.ax_latent_space.text(point_label[0], point_label[1], label,
                                                     ha='center', va='bottom', color='black')
                    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                           path_effects.Normal()])
            else:
                if self.index_data_label_shown is not None:
                    text = self.ax_latent_space.text(self.Z[self.index_data_label_shown, 0],
                                                     self.Z[self.index_data_label_shown, 1],
                                                     self.label_groups[self.index_data_label_shown],
                                                     ha='center', va='bottom', color='black')
                    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                           path_effects.Normal()]
                                          )
                else:
                    pass

        if self.click_coordinates_latent_space is not None:
            if self.is_select_latent_variable:
                self.ax_latent_space.scatter(self.click_coordinates_latent_space[0],
                                             self.click_coordinates_latent_space[1],
                                             marker='o', s=50, c='r',
                                             edgecolors='w', linewidths=2.0)
            else:
                self.ax_latent_space.scatter(self.click_coordinates_latent_space[0], self.click_coordinates_latent_space[1],
                                             marker="o", color="k", s=40, edgecolors='w', linewidths=2.0)

        if self.mesh_in_latent_space is not None:
            self._plot_wireframe(x=self.mesh_in_latent_space[:, :, 0],
                                 y=self.mesh_in_latent_space[:, :, 1],
                                 ax=self.ax_latent_space)
        else:
            pass

        if self.comment_latent_space is not None:
            self.ax_latent_space.text(1.0, 0.0,
                                      self.comment_latent_space,
                                      horizontalalignment='right',
                                      verticalalignment='top',
                                      transform=self.ax_latent_space.transAxes
                                      )

        if not self.is_show_ticks:
            self.ax_latent_space.tick_params(labelbottom=False,
                                             labelleft=False,
                                             labelright=False,
                                             labeltop=False)
            self.ax_latent_space.tick_params(bottom=False,
                                             left=False,
                                             right=False,
                                             top=False)

        self.fig.show()

    def _draw_data_space(self):
        self.ax_data_space.cla()
        self.ax_data_space.set_title(self.title_data_space)
        if self.grid_values_to_draw_data_space is not None:
            if self.grid_values_to_draw_data_space.ndim == 2:
                if self.grid_values_to_draw_data_space.shape[1] == 2:
                    from libs_tpm.tools.cmap2d_value_and_brightness import cmap2d_base_and_brightness
                    if self.is_latent_space_middle_color_zero:
                        max_grid_value = self.grid_values_to_draw_data_space[0].max()
                        min_grid_value = self.grid_values_to_draw_data_space[0].min()
                        vmin = -max(abs(max_grid_value), abs(min_grid_value))
                        vmax = max(abs(max_grid_value), abs(min_grid_value))
                    else:
                        vmin = None
                        vmax = None
                    if 'cmap' in self.params_imshow_latent_space.keys():
                        base_cmap = self.params_imshow_latent_space['cmap']
                    else:
                        base_cmap = None
                    grid_values_to_imshow = cmap2d_base_and_brightness(value=self.grid_values_to_draw_data_space[:, 0],
                                                                       brightness=self.grid_values_to_draw_data_space[:,
                                                                                  1],
                                                                       base_cmap=base_cmap,
                                                                       vmin=vmin, vmax=vmax)
                    grid_values_to_contour = self.grid_values_to_draw_data_space[:, 0]
                else:
                    if self.is_latent_space_middle_color_zero:
                        max_grid_value = self.grid_values_to_draw_data_space.max()
                        min_grid_value = self.grid_values_to_draw_data_space.min()
                        vmin = -max(abs(max_grid_value), abs(min_grid_value))
                        vmax = max(abs(max_grid_value), abs(min_grid_value))
                    else:
                        vmin = None
                        vmax = None
                    raise ValueError('invalid grid_value shape={}'.format(self.grid_values_to_draw_data_space.shape))
            elif self.grid_values_to_draw_data_space.ndim == 1:
                grid_values_to_imshow = self.grid_values_to_draw_data_space
                grid_values_to_contour = self.grid_values_to_draw_data_space
            else:
                raise ValueError('invalid grid_value ndim={}'.format(self.grid_values_to_draw_data_space.ndim))

            grid_points_3d = self.__unflatten_grid_array(self.grid_points_data_space,
                                                         self.n_grid_points_data_space)
            grid_values_to_imshow_3d = self.__unflatten_grid_array(grid_values_to_imshow,
                                                                   self.n_grid_points_data_space)
            grid_values_to_contour_3d = self.__unflatten_grid_array(grid_values_to_contour,
                                                                    self.n_grid_points_data_space)
            # set coordinate of axis
            any_index = 0
            if grid_points_3d[any_index, 0, 0] < grid_points_3d[any_index, -1, 0]:
                coordinate_ax_left = grid_points_3d[any_index, 0, 0]
                coordinate_ax_right = grid_points_3d[any_index, -1, 0]
            else:
                coordinate_ax_left = grid_points_3d[any_index, -1, 0]
                coordinate_ax_right = grid_points_3d[any_index, 0, 0]
                grid_values_to_imshow_3d = np.flip(grid_values_to_imshow_3d, axis=1).copy()

            if grid_points_3d[-1, any_index, 1] < grid_points_3d[0, any_index, 1]:
                coordinate_ax_bottom = grid_points_3d[-1, any_index, 1]
                coordinate_ax_top = grid_points_3d[0, any_index, 1]
            else:
                coordinate_ax_bottom = grid_points_3d[0, any_index, 1]
                coordinate_ax_top = grid_points_3d[-1, any_index, 1]
                grid_values_to_imshow_3d = np.flip(grid_values_to_imshow_3d, axis=0).copy()

            # extent = [left, right, bottom, top]
            self.ax_data_space.imshow(grid_values_to_imshow_3d,
                                      extent=[coordinate_ax_left,
                                              coordinate_ax_right,
                                              coordinate_ax_bottom,
                                              coordinate_ax_top
                                              ],
                                      **self.params_imshow_data_space
                                      )
            ctr = self.ax_data_space.contour(grid_points_3d[:, :, 0],
                                             grid_points_3d[:, :, 1],
                                             grid_values_to_contour_3d, 6, colors='k')
            self.ax_data_space.clabel(ctr)
        self.ax_data_space.scatter(self.member_features[self.mask_shown_member, 0],
                                   self.member_features[self.mask_shown_member, 1],
                                   **self.params_scatter_data_space)
        # if self.label_members is None:
        #     pass
        # else:
        #     for z, noise, label in zip(self.Z, self.noise_label, self.label_data):
        #         point_label = z + noise
        #         self.ax_data_space.text(point_label[0], point_label[1], label,
        #                                   ha='center', va='bottom', color='black')
        if not self.is_show_ticks:
            self.ax_data_space.tick_params(labelbottom=False,
                                             labelleft=False,
                                             labelright=False,
                                             labeltop=False)
            self.ax_data_space.tick_params(bottom=False,
                                             left=False,
                                             right=False,
                                             top=False)
        self.fig.show()

    def __calc_nearest_candidate(self, click_coordinates, candidates, retdist=False):
        distance = cdist(candidates, click_coordinates[None, :], metric='euclidean')
        index_nearest = np.argmin(distance.ravel())
        dist_min = distance.min()
        if retdist:
            return index_nearest, dist_min
        else:
            return index_nearest

    def __unflatten_grid_array(self, grid_array, n_grid_points):
        if grid_array.shape[0] == np.prod(n_grid_points):
            return np.squeeze(grid_array.reshape(np.append(n_grid_points, -1)))
        else:
            raise ValueError('arg shape {} is not consistent'.format(grid_array.shape))

    def _plot_wireframe(self, x: np.ndarray, y: np.ndarray, ax, **kwargs):
        from matplotlib.collections import LineCollection
        segs1 = np.stack((x, y), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))

