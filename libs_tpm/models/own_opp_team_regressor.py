from somf.libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from libs_tpm.models.ukr_for_kde import UKRForWeightedKDE as UKRKDE
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist
from tqdm import tqdm


class OwnTeamOppTeamRegressor():
    def __init__(self, win_team_bag_of_members, lose_team_bag_of_members,
                 win_team_performance, lose_team_performance, member_features,
                 params_lower_ukr, params_upper_ukr_kde, params_gpr,
                 init_upper_ukr_kde=None):
        if win_team_bag_of_members.shape != lose_team_bag_of_members.shape:
            raise ValueError('own_bag_of_members and opp_bga_of_members must be same size')
        self.n_games = win_team_bag_of_members.shape[0]
        # if is_own_win.ndim!=1 or is_own_win.shape[0]!=self.n_games:
        #     raise ValueError('is_own_win must be same length own_bag_of_members.shape[0]')

        # UKRに渡すパラメータをセットしてUKRのインスタンスを作成する
        self.params_lower_ukr = params_lower_ukr.copy()
        self.params_lower_ukr['X'] = member_features.copy()
        self.lower_ukr = UKR(**self.params_lower_ukr)

        # UKR for KDEに渡すgroup_featureを作りその他パラメータと共に辞書に渡す
        self.params_upper_ukr_kde = params_upper_ukr_kde.copy()
        team_features = np.concatenate([win_team_bag_of_members, lose_team_bag_of_members], axis=0)
        if 'weight_of_group' in params_upper_ukr_kde:
            raise ValueError("Do not set 'weight_of_group' in params_upper_ukr_kde")
        else:
            self.params_upper_ukr_kde['weight_of_group'] = team_features.copy()
        if init_upper_ukr_kde is None:
            pass
        elif init_upper_ukr_kde == 'mean_team_member_feature':
            if 'init' in params_upper_ukr_kde:
                raise ValueError('Already init is set in params_upper_ukr_kde')
            else:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                mean_team_member_feature = (team_features @ member_features) / team_features.sum(axis=1)[:, None]
                standardizer = StandardScaler()
                mean_team_member_feature = standardizer.fit_transform(mean_team_member_feature)
                pca = PCA(n_components=self.params_upper_ukr_kde['n_embedding'])
                init_value = pca.fit_transform(mean_team_member_feature)
                # scale for component 1 var 1
                init_value = (init_value / np.std(init_value[:, 0]))
                # scale by bandwidth
                init_value = init_value * self.params_upper_ukr_kde['bandwidth_nadaraya'] * 0.1
                self.params_upper_ukr_kde['init'] = init_value

        else:
            raise ValueError('invalid init_upper_ukr = {}'.format(init_upper_ukr_kde))
        self.num_groups = team_features.shape[0]

        # Regressorに渡すパラメータをセット
        self.win_team_performance = win_team_performance.copy()
        self.lose_team_performance = lose_team_performance.copy()
        self.training_performance = np.concatenate([self.win_team_performance,
                                                    self.lose_team_performance], axis=0)
        self.gpr = GaussianProcessRegressor(**params_gpr)

    def fit(self, nb_epoch_lower_ukr, eta_lower_ukr,
            nb_epoch_upper_ukr_kde, eta_upper_ukr_kde,
            lower_ukr_fit=None, upper_ukr_kde_fit=None):
        # fit lower ukr
        if lower_ukr_fit is None:
            self._fit_lower_ukr(nb_epoch_lower_ukr=nb_epoch_lower_ukr,
                                eta_lower_ukr=eta_lower_ukr)
            self.is_given_lower_ukr_fit = False
        elif isinstance(lower_ukr_fit, UKR):
            # 与えられたUKRが適切か簡易チェック
            if not np.all(lower_ukr_fit.X == self.lower_ukr.X):
                raise ValueError("Not expected lower_ukr_fit's X")
            elif lower_ukr_fit.bandwidth_gaussian_kernel != self.lower_ukr.bandwidth_gaussian_kernel:
                raise ValueError("Not expected lower_ukr_fit's bandwidth_gaussian_kernel")
            else:
                print('In OwnOppTeamRegressor: Use fit lower ukr')
                self.lower_ukr = lower_ukr_fit
                self.is_given_lower_ukr_fit = True
        else:
            raise ValueError('lower_ukr_fit must to be given fit ukr instance.')

        # fit upper ukr for kde
        if upper_ukr_kde_fit is None:
            self._fit_upper_ukr_kde(nb_epoch_ukr_kde=nb_epoch_upper_ukr_kde,
                                    eta_ukr_kde=eta_upper_ukr_kde)
            self.is_give_upper_ukr_kde_fit = False
        elif isinstance(upper_ukr_kde_fit, UKRKDE):
            # 簡易チェック
            if not np.all(upper_ukr_kde_fit.member_features == self.lower_ukr.Z):
                raise ValueError('upper_ukr_kde_fit.member_features is not matched laten variables of lower ukr')
            else:
                print('In OwnOppTeamRegressor: Use fit upper ukr for kde')
                self.upper_ukr_kde = upper_ukr_kde_fit
                self.is_give_upper_ukr_kde_fit = True
        else:
            raise ValueError('upper_ukr_kde_fit must to be given fit ukr for kde instance')

        # fit gpr
        self._fit_gpr()

    def _fit_lower_ukr(self, nb_epoch_lower_ukr, eta_lower_ukr):
        # ukrのfit
        self.lower_ukr.fit(nb_epoch=nb_epoch_lower_ukr, eta=eta_lower_ukr)

    def _fit_upper_ukr_kde(self, nb_epoch_ukr_kde, eta_ukr_kde):
        # ukr for kdeのfit
        self.params_upper_ukr_kde['member_features'] = self.lower_ukr.Z
        self.upper_ukr_kde = UKRKDE(**self.params_upper_ukr_kde)
        self.upper_ukr_kde.fit(n_epoch=nb_epoch_ukr_kde,
                               learning_rate=eta_ukr_kde)

    def _fit_gpr(self):
        # GPRegressionのfit
        ## GPRのfitの入力Xを作る
        ## 元々上位のUKRには自チームと相手チームのbag of membersを縦方向にconcatenateして入力しているため
        ## 半分の位置で二つに分ける
        own_latent_variables = self.upper_ukr_kde.Z[:self.n_games, :].copy()  # 自チームの潜在変数
        opp_latent_variables = self.upper_ukr_kde.Z[self.n_games:, :].copy()  # 相手チームの潜在変数
        # 回帰のTrainingに用いる自チームに相当する入力（潜在変数）と相手チームのそれを作成
        training_own_latent_variables = np.concatenate([own_latent_variables,
                                                        opp_latent_variables], axis=0)
        training_opp_latent_variables = np.concatenate([opp_latent_variables,
                                                        own_latent_variables], axis=0)
        # 最終的に回帰に渡すトレーニングセットの入力は自チームと相手チームのペア
        # 自チームに相当するownと相手チームに相当するoppの潜在変数を列方向にconcatenateする
        gpr_training_X = np.concatenate([training_own_latent_variables,
                                         training_opp_latent_variables], axis=1)
        ## fitしちゃう
        self.gpr.fit(X=gpr_training_X,
                     y=self.training_performance)

    def predict(self, own_team_bag_of_members, opp_team_bag_of_members,
                n_epoch, learning_rate, return_std=False, return_cov=False):
        own_latent_variables = self.upper_ukr_kde.transform(own_team_bag_of_members,
                                                            n_epoch=n_epoch,
                                                            learning_rate=learning_rate)
        opp_latent_variables = self.upper_ukr_kde.transform(opp_team_bag_of_members,
                                                            n_epoch=n_epoch,
                                                            learning_rate=learning_rate)
        return self._predict_from_two_latent_spaces(own_latent_variables,
                                                    opp_latent_variables,
                                                    return_std=return_std,
                                                    return_cov=return_cov)

    def _predict_from_two_latent_spaces(self, own_Zeta, opp_Zeta,
                                        return_std=False, return_cov=False):
        n_dim_latent_space = self.upper_ukr_kde.n_embedding
        return self.gpr.predict(X=np.concatenate([own_Zeta.reshape(-1, n_dim_latent_space),
                                                  opp_Zeta.reshape(-1, n_dim_latent_space)], axis=1),
                                return_std=return_std, return_cov=return_cov)

    def visualize(self, n_grid_points=30, cmap_feature=None, cmap_density=None, cmap_ccp=None,
                  label_member=None, label_feature=None, label_team=None, label_performance=None,
                  fig_size=None, is_member_cp_middle_color_zero=False, is_ccp_middle_color_zero=False,
                  params_init_lower_ukr=None, params_init_upper_ukr=None,
                  n_epoch_to_change_member=1500, learning_rate_to_change_member=0.001):
        # invalid check
        if self.lower_ukr.n_components != 2 or self.upper_ukr_kde.n_embedding != 2:
            raise ValueError('Now support only n_components = 2')

        # import necessary library to draw
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        # 相手チームのマップに関する変数や描画メソッドをまとめておく新しいukr for kdeインスタンスを作る
        self.own_ukr_kde = self.upper_ukr_kde  # こっちは参照渡し
        self.opp_ukr_kde = UKRKDE(**self.params_upper_ukr_kde)
        self.opp_ukr_kde.Z = self.upper_ukr_kde.Z.copy()

        # Create instance to draw opp member map
        import copy
        self.opp_lower_ukr = copy.deepcopy(self.lower_ukr)

        # 色付けに関わる変数の用意
        self.cmap_density = cmap_density
        self.cmap_feature = cmap_feature
        self.cmap_ccp = cmap_ccp
        self.is_ccp_middle_color_zero = is_ccp_middle_color_zero
        self.is_member_cp_middle_color_zero = is_member_cp_middle_color_zero

        # メンバー交代simulationのための変数の用意
        self.n_epoch_to_change_member =n_epoch_to_change_member
        self.learning_rate_to_change_member = learning_rate_to_change_member
        self.is_simulate_changing_member = False
        self.deleted_member_coordinates = None
        self.grid_simulated_latent_variables = None

        # figureやaxesの用意
        if fig_size is None:
            self.fig = plt.figure(figsize=(14, 10))
        else:
            self.fig = plt.figure(figsize=fig_size)
        ax_opp_member_latent_space = self.fig.add_subplot(2, 3, 6, aspect='equal')
        ax_lower_ukr_latent_space = self.fig.add_subplot(2, 3, 1, aspect='equal')
        ax_lower_ukr_feature_bars = self.fig.add_subplot(2, 3, 4)
        ax_own_team_latent_space = self.fig.add_subplot(2, 3, 2, aspect='equal')
        ax_opp_team_latent_space = self.fig.add_subplot(2, 3, 3, aspect='equal')
        self.ax_performance_bars = self.fig.add_subplot(2, 3, 5)
        self.fig.subplots_adjust(left=0.075, bottom=0.1, right=0.95, top=0.95, wspace=0.15, hspace=0.15)

        # 各インスタンスに描画に関わる変数をセットする
        if 'params_scatter' in params_init_lower_ukr.keys():
            pass
        else:
            params_init_lower_ukr['params_scatter'] = None
        self.lower_ukr._initialize_to_visualize(n_grid_points,
                                                label_data=label_member,
                                                label_feature=label_feature,
                                                title_latent_space='Own member map',
                                                title_feature_bars='Member feature',
                                                is_show_all_label_data=False,
                                                params_imshow={'cmap': cmap_feature,
                                                               'interpolation': 'spline16'},
                                                is_middle_color_zero=is_member_cp_middle_color_zero,
                                                is_show_ticks_latent_space=False,
                                                fig=self.fig,
                                                fig_size=None,
                                                ax_latent_space=ax_lower_ukr_latent_space,
                                                ax_feature_bars=ax_lower_ukr_feature_bars,
                                                **params_init_lower_ukr)
        self.opp_lower_ukr._initialize_to_visualize(n_grid_points,
                                                label_data=label_member,
                                                label_feature=label_feature,
                                                title_latent_space='Opp member map',
                                                title_feature_bars='Member feature',
                                                is_show_all_label_data=False,
                                                is_show_ticks_latent_space=False,
                                                params_imshow={'cmap': cmap_feature,
                                                               'interpolation': 'spline16'},
                                                is_middle_color_zero=is_member_cp_middle_color_zero,
                                                fig=self.fig,
                                                fig_size=None,
                                                ax_latent_space=ax_opp_member_latent_space,
                                                ax_feature_bars=None,
                                                **params_init_lower_ukr)
        # 下位のUKRの潜在空間を上位のUKRの観測空間
        params_init_upper_ukr['params_scatter_data_space'] = None
        self.own_ukr_kde._initialize_to_visualize(n_grid_points,
                                                  params_imshow_latent_space={'cmap': self.cmap_ccp,
                                                                              'interpolation': 'spline16'},
                                                  params_imshow_data_space={'interpolation': 'spline16'},
                                                  label_groups=label_team,
                                                  title_latent_space='Own team map',
                                                  title_data_space='Own member map',
                                                  fig=self.fig,
                                                  fig_size=None,
                                                  ax_latent_space=ax_own_team_latent_space,
                                                  ax_data_space=ax_lower_ukr_latent_space,
                                                  is_latent_space_middle_color_zero=is_ccp_middle_color_zero,
                                                  is_show_all_label_groups=False,
                                                  is_show_ticks=False,
                                                  **params_init_upper_ukr)
        self.opp_ukr_kde._initialize_to_visualize(n_grid_points,
                                                  params_imshow_latent_space={'cmap': self.cmap_ccp,
                                                                              'interpolation': 'spline16'},
                                                  params_imshow_data_space={'cmap': self.cmap_density,
                                                                            'interpolation': 'spline16'},
                                                  label_groups=label_team,
                                                  title_latent_space='Opposing team map',
                                                  title_data_space='Opposing member map',
                                                  fig=self.fig,
                                                  fig_size=None,
                                                  ax_latent_space=ax_opp_team_latent_space,
                                                  ax_data_space=ax_opp_member_latent_space,
                                                  is_latent_space_middle_color_zero=is_ccp_middle_color_zero,
                                                  is_show_all_label_groups=False,
                                                  is_show_ticks=False,
                                                  **params_init_upper_ukr)
        # 下位のUKRの潜在空間上の代表点を上位のUKRの観測空間上の代表点と合わせる
        self.lower_ukr._set_grid(grid_points=self.own_ukr_kde.grid_points_data_space,
                                 n_grid_points=self.own_ukr_kde.n_grid_points_data_space)

        # CCPに必要な変数を揃える
        if 'mesh_grid_mapping' in vars(self) and 'mesh_grid_precision':
            if self.mesh_grid_mapping.shape[0] == self.upper_ukr_kde.n_grid_points_latent_space.prod():
                pass
            else:
                raise ValueError('invalid previous mesh_grid_mapping or mesh_grid_precision')
        else:
            self.mesh_grid_mapping, mesh_grid_uncertainty = self._create_tensor_mapping_for_ccp(return_std=True)
            self.mesh_grid_precision = np.reciprocal(mesh_grid_uncertainty)
        self.selected_performance = None
        self.double_click_mapping = None
        if label_performance is None:
            self.label_performance = np.arange(self.training_performance.shape[1])
        else:
            self.label_performance = label_performance
        # CCPで不確定性を表示する際の輝度の幅を指定, 0.0は真っ黒になって判別できないため0.3未満は利用しない
        self.own_ukr_kde.set_used_bright_range([0.5, 1.0])
        self.opp_ukr_kde.set_used_bright_range([0.5, 1.0])

        # まず一度一通り描画しておく
        self.lower_ukr.set_comment_in_latent_space('Click in map')
        self.own_ukr_kde.set_comment_in_latent_space('Click in map')
        self.opp_ukr_kde.set_comment_in_latent_space('Click in map')
        self.opp_lower_ukr.set_comment_in_latent_space('Click in map')
        self._set_target_bars_from_two_team_latent_spaces()
        self.lower_ukr._draw_latent_space()
        self.lower_ukr._draw_feature_bars()
        self.own_ukr_kde._draw_latent_space()
        self.opp_ukr_kde._draw_latent_space()
        self.opp_lower_ukr._draw_latent_space()
        self._draw_target_bars()
        # connect figure and method defining action when latent space is clicked
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)
        self.fig.canvas.mpl_connect('motion_notify_event', self.__mouse_over_fig)
        plt.show()

    def __onclick_fig(self, event):
        self.is_initial_view = False
        if event.xdata is not None:
            click_coordinates = np.array([event.xdata, event.ydata])
            if event.inaxes == self.lower_ukr.ax_latent_space.axes:
                # 下位のUKRの潜在空間をクリックした時の挙動
                # If specific latent variable is selected in team map
                if self.own_ukr_kde.is_select_latent_variable:
                    self._set_to_simulate_changing_member(click_coordinates)
                else:
                    self.is_simulate_changing_member = False
                self.lower_ukr._set_feature_bar_from_latent_space(click_coordinates)


                # set the value to draw in own team map
                self.own_ukr_kde._set_latent_space_from_data_space(click_coordinates)
                self.own_ukr_kde.set_params_imshow_latent_space({'cmap': self.cmap_density})
                self.own_ukr_kde.is_latent_space_middle_color_zero = False

                # draw
                self.lower_ukr._draw_latent_space()
                self.lower_ukr._draw_feature_bars()
                self.own_ukr_kde._draw_latent_space()


            elif event.inaxes == self.lower_ukr.ax_feature_bars.axes:
                # 下位のUKRの特徴量barがクリックされた時
                self.lower_ukr._set_latent_space_from_feature_bar(click_coordinates)
                self.lower_ukr.set_params_imshow({'cmap': self.cmap_feature})
                self.lower_ukr.is_middle_color_zero = self.is_member_cp_middle_color_zero
                self.lower_ukr._draw_latent_space()
                self.lower_ukr._draw_feature_bars()
                annotation_text = 'Heat map of selected feature'
                self.lower_ukr.set_comment_in_latent_space(annotation_text)

            elif event.inaxes == self.own_ukr_kde.ax_latent_space.axes:
                if self.is_simulate_changing_member:
                    self._reset_simulation()
                # 下位の潜在空間に表示する値を計算
                self.own_ukr_kde._set_data_space_from_latent_space(click_coordinates)
                # その値を取得
                grid_values = self.own_ukr_kde.get_grid_values_to_draw_data_space()
                mask_shown_member = self.own_ukr_kde.mask_shown_member
                # 下位のUKRにセット
                self.lower_ukr._set_grid_values_to_draw(grid_values)
                self.lower_ukr.set_params_imshow({'cmap': self.cmap_density})
                self.lower_ukr.is_middle_color_zero = False
                self.lower_ukr.set_mask_latent_variables(mask=mask_shown_member)

                # if self.own_ukr_kde.is_select_latent_variable:
                #     print('Selecting latent variable in own team={}'.format(self.own_ukr_kde.index_team_selected))

                # set the value to draw opp team map
                self._set_opp_team_latent_space_from_own_team_latent_space()
                self.opp_ukr_kde.set_params_imshow_latent_space({'cmap': self.cmap_ccp})
                self.opp_ukr_kde.is_latent_space_middle_color_zero = self.is_ccp_middle_color_zero
                self._set_target_bars_from_two_team_latent_spaces()

                # set comment
                if self.own_ukr_kde.is_select_latent_variable:
                    annotation_text = 'Density of {}\nIf member is clicked, new team mode starts.'.format(
                        self.own_ukr_kde.label_groups[self.own_ukr_kde.index_team_selected]
                    )
                else:
                    annotation_text = 'Density of clicked own team'
                self.lower_ukr.set_comment_in_latent_space(annotation_text)

                # 描画
                self.lower_ukr._draw_latent_space()
                self.lower_ukr._draw_feature_bars()
                self.own_ukr_kde._draw_latent_space()
                self.opp_ukr_kde._draw_latent_space()
                self._draw_target_bars()



            elif event.inaxes == self.opp_ukr_kde.ax_latent_space.axes:
                # set the value to draw member map
                self.opp_ukr_kde._set_data_space_from_latent_space(click_coordinates)
                grid_values = self.opp_ukr_kde.get_grid_values_to_draw_data_space()
                mask_shown_member = self.opp_ukr_kde.mask_shown_member
                self.opp_lower_ukr._set_grid_values_to_draw(grid_values)
                self.opp_lower_ukr.set_params_imshow({'cmap': self.cmap_density})
                self.opp_lower_ukr.is_middle_color_zero = False
                self.opp_lower_ukr.set_mask_latent_variables(mask=mask_shown_member)

                # If simulate mode, update simulation result
                if self.is_simulate_changing_member:
                    self._set_to_simulate_changing_member(click_coordinates=self.deleted_member_coordinates)

                # set own team map and target bars
                self._set_own_team_latent_space_from_opp_team_latent_space()
                self.own_ukr_kde.set_params_imshow_latent_space({'cmap': self.cmap_ccp})
                self._set_target_bars_from_two_team_latent_spaces()

                # set comment
                if self.opp_ukr_kde.is_select_latent_variable:
                    annotation_text = 'Density of {}'.format(
                        self.opp_ukr_kde.label_groups[self.opp_ukr_kde.index_team_selected]
                    )
                else:
                    annotation_text = 'Density of clicked opp team'
                self.opp_lower_ukr.set_comment_in_latent_space(annotation_text)

                # 描画
                self.lower_ukr._draw_latent_space()
                self.lower_ukr._draw_feature_bars()
                self.own_ukr_kde._draw_latent_space()
                self.opp_ukr_kde._draw_latent_space()
                self.opp_lower_ukr._draw_latent_space()
                self._draw_target_bars()
            elif event.inaxes == self.opp_lower_ukr.ax_latent_space.axes:
                # 下位のUKRの潜在空間をクリックした時の挙動
                self.opp_lower_ukr._set_feature_bar_from_latent_space(click_coordinates)

                # set the value to draw in own team map
                self.opp_ukr_kde._set_latent_space_from_data_space(click_coordinates)
                self.opp_ukr_kde.set_params_imshow_latent_space({'cmap': self.cmap_density})
                self.opp_ukr_kde.is_latent_space_middle_color_zero = False

                # draw
                self.opp_lower_ukr._draw_latent_space()
                #self.opp_lower_ukr._draw_feature_bars()
                self.opp_ukr_kde._draw_latent_space()
            elif event.inaxes == self.ax_performance_bars.axes:
                self._set_two_team_latent_spaces_from_target_bars(click_coordinates)

                if self.is_simulate_changing_member:
                    self._set_to_simulate_changing_member(click_coordinates=self.deleted_member_coordinates)
                self.own_ukr_kde.set_params_imshow_latent_space({'cmap': self.cmap_ccp})
                self.opp_ukr_kde.set_params_imshow_latent_space({'cmap': self.cmap_ccp})
                self.own_ukr_kde._draw_latent_space()
                self.opp_ukr_kde._draw_latent_space()
                self._draw_target_bars()
                self.lower_ukr._draw_latent_space()

                # # The annotation in own team map
                # if self.selected_performance is not None:
                #     if self.own_ukr_kde.click_coordinates_latent_space is not None:
                #         if self.opp_ukr_kde.is_select_latent_variable:
                #             annotation_text = 'Heat map of {} (vs {})'.format(self.label_performance[self.selected_performance],
                #                                                               self.opp_ukr_kde.label_groups[self.opp_ukr_kde.index_team_selected]
                #                                                               )
                #         else:
                #             annotation_text = 'Heat map of {} (vs clicked team)'.format(self.label_performance[self.selected_performance])
                #     else:
                #         annotation_text = 'Heat map of {} (marginal)'.format(
                #             self.label_performance[self.selected_performance]
                #         )
                #     self.own_ukr_kde.ax_latent_space.text(1.0, 0.0,
                #                                           annotation_text,
                #                                           horizontalalignment='right',
                #                                           verticalalignment='top',
                #                                           transform=self.own_ukr_kde.ax_latent_space.transAxes
                #                                           )
                # # The annotation in opp team map
                # if self.selected_performance is not None:
                #     if self.own_ukr_kde.click_coordinates_latent_space is not None:
                #         # conditional
                #         if self.own_ukr_kde.is_select_latent_variable:
                #             annotation_text = 'Heat map of {} (vs {})'.format(self.label_performance[self.selected_performance],
                #                                                               self.own_ukr_kde.label_groups[self.own_ukr_kde.index_team_selected]
                #                                                               )
                #         else:
                #             annotation_text = 'Heat map of {} (vs clicked own team)'.format(self.label_performance[self.selected_performance])
                #     else:
                #         # marginal
                #         annotation_text = 'Heat map of {} (marginal)'.format(
                #             self.label_performance[self.selected_performance]
                #         )
                #     self.opp_ukr_kde.ax_latent_space.text(1.0, 0.0,
                #                                           annotation_text,
                #                                           horizontalalignment='right',
                #                                           verticalalignment='top',
                #                                           transform=self.opp_ukr_kde.ax_latent_space.transAxes
                #                                           )



    def __mouse_over_fig(self, event):
        if event.xdata is not None:
            # オーバーされた座標の取得
            over_coordinates = np.array([event.xdata, event.ydata])
            if event.inaxes == self.lower_ukr.ax_latent_space.axes:
                self.lower_ukr._set_shown_label_in_latent_space(over_coordinates)
                self.lower_ukr._draw_latent_space()
            elif event.inaxes == self.own_ukr_kde.ax_latent_space.axes:
                self.own_ukr_kde._set_shown_label_in_latent_space(over_coordinates)
                self.own_ukr_kde._draw_latent_space()
            elif event.inaxes == self.opp_ukr_kde.ax_latent_space.axes:
                self.opp_ukr_kde._set_shown_label_in_latent_space(over_coordinates)
                self.opp_ukr_kde._draw_latent_space()
            elif event.inaxes == self.opp_lower_ukr.ax_latent_space.axes:
                self.opp_lower_ukr._set_shown_label_in_latent_space(over_coordinates)
                self.opp_lower_ukr._draw_latent_space()
            else:
                pass

    def _create_tensor_mapping_for_ccp(self, return_std=False, return_cov=False):
        if return_std or return_cov:
            if 'noise_level' in self.gpr.kernel_.k1.__dict__.keys():
                white_level = self.gpr.kernel_.k1.noise_level
            elif 'noise_level' in self.gpr.kernel_.k2.__dict__.keys():
                white_level = self.gpr.kernel_.k2.noise_level
            else:
                raise ValueError("Can't find WhiteKernel!")
        if return_cov:
            raise ValueError('Now, not support return_cov={}'.format(return_cov))
        # prepare empty array
        mesh_grid_mapping = np.empty(
            (
                self.own_ukr_kde.n_grid_points_latent_space.prod(),
                self.opp_ukr_kde.n_grid_points_latent_space.prod(),
                self.training_performance.shape[1]
            )
        )

        if return_std:
            mesh_grid_std = np.empty(
                (
                    self.own_ukr_kde.n_grid_points_latent_space.prod(),
                    self.opp_ukr_kde.n_grid_points_latent_space.prod()
                )
            )
        if return_cov:
            mesh_grid_cov = np.empty(
                (
                    self.own_ukr_kde.n_grid_points_latent_space.prod(),
                    self.opp_ukr_kde.n_grid_points_latent_space.prod(),
                    self.opp_ukr_kde.n_grid_points_latent_space.prod()
                )
            )
        # insert value
        for i, own_grid_point in enumerate(tqdm(self.own_ukr_kde.grid_points_latent_space)):
            for j, opp_grid_point in enumerate(self.opp_ukr_kde.grid_points_latent_space):
                if return_std:
                    mesh_grid_mapping[i, j], val_cov_in_white = self._predict_from_two_latent_spaces(own_grid_point,
                                                                                                     opp_grid_point,
                                                                                                     return_std=False,
                                                                                                     return_cov=True)
                    mesh_grid_std[i, j] = np.sqrt(val_cov_in_white[0, 0] - white_level)
                elif return_cov:
                    mesh_grid_mapping[i, j], mesh_grid_cov[i, j] = self._predict_from_two_latent_spaces(own_grid_point,
                                                                                                        opp_grid_point,
                                                                                                        return_std=return_std,
                                                                                                        return_cov=return_cov)
                else:
                    mesh_grid_mapping[i, j] = self._predict_from_two_latent_spaces(own_grid_point, opp_grid_point,
                                                                                   return_std=return_std,
                                                                                   return_cov=return_cov)
        if return_std:
            return mesh_grid_mapping, mesh_grid_std
        elif return_cov:
            return mesh_grid_mapping, mesh_grid_cov
        else:
            return mesh_grid_mapping

    def _set_own_team_latent_space_from_opp_team_latent_space(self):
        # クリックされた座標の情報はself.opp_ukr_kde._set_data_space_from_latent_spaceで
        # 既にself.opp_ukr_kdeに与えてあるのでここで代入する必要はない
        if self.selected_performance is not None:
            if self.opp_ukr_kde.click_coordinates_latent_space is not None:
                # conditional component plane
                index = self.__calc_nearest_candidate(self.opp_ukr_kde.click_coordinates_latent_space,
                                                      self.opp_ukr_kde.grid_points_latent_space)
                print('in opp grid coordinate={}'.format(self.opp_ukr_kde.grid_points_latent_space[index]))
                grid_values = self.mesh_grid_mapping[:, index, self.selected_performance][:, None]
                grid_precision = self.mesh_grid_precision[:, index][:, None]
                if self.opp_ukr_kde.is_select_latent_variable:
                    annotation_text = 'Heat map of {} (vs {})'.format(self.label_performance[self.selected_performance],
                                                                      self.opp_ukr_kde.label_groups[
                                                                          self.opp_ukr_kde.index_team_selected]
                                                                      )
                else:
                    annotation_text = 'Heat map of {} (vs clicked opp team)'.format(
                        self.label_performance[self.selected_performance]
                    )

            else:
                # marginal component plane
                grid_values = np.mean(self.mesh_grid_mapping[:, :, self.selected_performance], axis=1)[:, None]
                grid_precision = np.mean(self.mesh_grid_precision[:, :], axis=1)[:,None]
                annotation_text = 'Heat map of {} (marginal)'.format(
                    self.label_performance[self.selected_performance]
                )
            set_grid_values = np.concatenate([grid_values, grid_precision], axis=1)
            self.own_ukr_kde.set_grid_values_to_draw_latent_space(set_grid_values)
            self.own_ukr_kde.set_comment_in_latent_space(annotation_text)
        else:
            pass

    def _set_opp_team_latent_space_from_own_team_latent_space(self):
        # クリックされた座標の情報はself.own_ukr_kde._set_data_space_from_latent_spaceで
        # 既にself.own_ukr_kdeに与えてあるのでここで代入する必要はない
        if self.selected_performance is not None:
            if self.own_ukr_kde.click_coordinates_latent_space is not None:
                # conditional
                index = self.__calc_nearest_candidate(self.own_ukr_kde.click_coordinates_latent_space,
                                                      self.own_ukr_kde.grid_points_latent_space)
                grid_values = self.mesh_grid_mapping[index, :, self.selected_performance][:, None]
                grid_precision = self.mesh_grid_precision[index, :][:, None]
                if self.own_ukr_kde.is_select_latent_variable:
                    annotation_text = 'Heat map of {} (vs {})'.format(self.label_performance[self.selected_performance],
                                                                      self.own_ukr_kde.label_groups[
                                                                          self.own_ukr_kde.index_team_selected]
                                                                      )
                else:
                    annotation_text = 'Heat map of {} (vs clicked own team)'.format(
                        self.label_performance[self.selected_performance])
            else:
                # marginal
                grid_values = np.mean(self.mesh_grid_mapping[:, :, self.selected_performance], axis=0)[:, None]
                grid_precision = np.mean(self.mesh_grid_precision[:, :], axis=0)[:, None]
                annotation_text = 'Heat map of {} (marginal)'.format(
                    self.label_performance[self.selected_performance]
                )
            # set specific feature of mapping and precision
            set_grid_values = np.concatenate([grid_values, grid_precision], axis=1)
            self.opp_ukr_kde.set_grid_values_to_draw_latent_space(set_grid_values)
            self.opp_ukr_kde.set_comment_in_latent_space(annotation_text)
        else:
            pass

    def _set_target_bars_from_two_team_latent_spaces(self):
        if self.own_ukr_kde.click_coordinates_latent_space is not None and self.opp_ukr_kde.click_coordinates_latent_space is not None:
            index_own = self.__calc_nearest_candidate(click_coordinates=self.own_ukr_kde.click_coordinates_latent_space,
                                                      candidates=self.own_ukr_kde.grid_points_latent_space)
            index_opp = self.__calc_nearest_candidate(click_coordinates=self.opp_ukr_kde.click_coordinates_latent_space,
                                                      candidates=self.opp_ukr_kde.grid_points_latent_space)
            self.double_click_mapping = self.mesh_grid_mapping[index_own, index_opp, :]
        elif self.own_ukr_kde.click_coordinates_latent_space is not None:
            index_own = self.__calc_nearest_candidate(click_coordinates=self.own_ukr_kde.click_coordinates_latent_space,
                                                      candidates=self.own_ukr_kde.grid_points_latent_space)
            self.double_click_mapping = np.mean(self.mesh_grid_mapping[index_own, :, :], axis=0)
        elif self.opp_ukr_kde.click_coordinates_latent_space is not None:
            index_opp = self.__calc_nearest_candidate(click_coordinates=self.opp_ukr_kde.click_coordinates_latent_space,
                                                      candidates=self.opp_ukr_kde.grid_points_latent_space)
            self.double_click_mapping = np.mean(self.mesh_grid_mapping[:, index_opp, :], axis=0)
        else:
            self.double_click_mapping = np.mean(self.mesh_grid_mapping, axis=(0, 1))

    def _set_two_team_latent_spaces_from_target_bars(self, click_coordinates):
        for i, bar in enumerate(self.performance_bars):
            if click_coordinates[0] > bar._x0 and click_coordinates[0] < bar._x1:
                self.selected_performance = i
        self._set_opp_team_latent_space_from_own_team_latent_space()
        self._set_own_team_latent_space_from_opp_team_latent_space()

    def _draw_target_bars(self):
        self.ax_performance_bars.cla()
        self.ax_performance_bars.set_title('Own team performance')
        if self.double_click_mapping is not None:
            self.performance_bars = self.ax_performance_bars.bar(self.label_performance, self.double_click_mapping)
            self.ax_performance_bars.set_title('Own team performance')
            if self.selected_performance is not None:
                self.performance_bars[self.selected_performance].set_color('r')
            # set ylim
            # to exclude outlier, lim value is given by 5 percentile and 95 percentile
            dim_min = np.argmin(self.training_performance.min(axis=0))  # calculate dimension has minimum value
            vmin = np.percentile(self.training_performance[:, dim_min], 5)  # value of 5 percentile
            dim_max = np.argmax(self.training_performance.max(axis=0))
            vmax = np.percentile(self.training_performance[:, dim_max], 95)
            self.ax_performance_bars.set_ylim(vmin, vmax)
        else:
            self.performance_bars = self.ax_performance_bars.bar(self.label_performance,
                                                                 np.zeros_like(self.label_performance))
        self.ax_performance_bars.set_xticklabels(labels=self.label_performance, rotation=270)
        if self.selected_performance is None:
            self.ax_performance_bars.text(1.0, 0.0, 'Click a stat you want to focus',
                                          horizontalalignment='right',
                                          verticalalignment='bottom',
                                          transform=self.ax_performance_bars.transAxes
                                          )
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


    def _set_to_simulate_changing_member(self, click_coordinates):
        member_features = self.own_ukr_kde.member_features[self.own_ukr_kde.mask_shown_member]
        index_nearest, distance = self.__calc_nearest_candidate(click_coordinates=click_coordinates,
                                                                candidates=member_features,
                                                                retdist=True)
        epsilon = 0.02 * np.abs(self.own_ukr_kde.grid_points_data_space.max() - self.own_ukr_kde.grid_points_data_space.min())
        if distance < epsilon and self.selected_performance is not None:
            self.is_simulate_changing_member = True
            self.deleted_member_coordinates = member_features[index_nearest]
            self.lower_ukr.set_scatter_cross(coordinates=self.deleted_member_coordinates)

            n_total_grid_points_in_member_map = self.own_ukr_kde.grid_points_data_space.shape[0]
            # Calculate team latent variables corresponds team changed selected member and all grid points
            if self.grid_simulated_latent_variables is None:
                if not np.isclose(self.own_ukr_kde.bag_of_shown_member.sum(), 1.0):
                    raise ValueError('sum of weights is not one: {}'.format(self.own_ukr_kde.bag_of_shown_member))
                # 残っているメンバーの重みを取得
                weights_minus_one = np.delete(self.own_ukr_kde.bag_of_shown_member, index_nearest)
                if weights_minus_one.sum() >= 1.0 or np.any(weights_minus_one < 0.0):
                    raise ValueError('invalid weights of shown member={}'.format(self.own_ukr_kde.bag_of_shown_member))
                # Get member features deleted selected member
                member_features_minus_one = np.delete(member_features, index_nearest, 0)
                # Create feature matrix added grid point
                # n_grid_points x n_member x dimension of data space
                grid_datasets = np.concatenate(
                    [
                        np.tile(
                            member_features_minus_one[None, :, :],
                            (n_total_grid_points_in_member_map, 1, 1)
                        ),
                        self.own_ukr_kde.grid_points_data_space[:, None, :]
                    ],
                    axis=1
                )
                weights = np.append(weights_minus_one, 1.0 - weights_minus_one.sum())
                # n_grid_points x n_member
                grid_weights = np.tile(weights[None, :],
                                       (n_total_grid_points_in_member_map, 1))
                # n_grid_points x latent dim of team map
                self.grid_simulated_latent_variables = self.own_ukr_kde.transform_from_datasets(
                    datasets=grid_datasets,
                    weights=grid_weights,
                    n_epoch=self.n_epoch_to_change_member,
                    learning_rate=self.learning_rate_to_change_member,
                    verbose=True
                )
            else:
                pass

            # set mesh in own team map
            self.own_ukr_kde.set_mesh_in_latent_space(
                mesh=self.__unflatten_grid_array(
                    grid_array=self.grid_simulated_latent_variables,
                    n_grid_points=self.own_ukr_kde.n_grid_points_data_space
                )
            )

            if self.opp_ukr_kde.click_coordinates_latent_space is not None:
                print('conditional simulation')
                # conditional
                tiled_opp_click_coordinates = np.tile(
                    self.opp_ukr_kde.click_coordinates_latent_space,
                    (n_total_grid_points_in_member_map, 1)
                )
                grid_predicted_performance = self._predict_from_two_latent_spaces(
                    own_Zeta=self.grid_simulated_latent_variables,
                    opp_Zeta=tiled_opp_click_coordinates,
                )[:, self.selected_performance]
            else:
                print('marginal simulation')
                # marginal
                mesh_grid_mapping = np.empty(
                    (
                        n_total_grid_points_in_member_map,
                        self.opp_ukr_kde.n_grid_points_latent_space.prod(),
                    )
                )
                for i, own_team_latent_variable in enumerate(tqdm(self.grid_simulated_latent_variables)):
                    for j, opp_grid_point in enumerate(self.opp_ukr_kde.grid_points_latent_space):
                        mesh_grid_mapping[i, j] = self._predict_from_two_latent_spaces(
                            own_Zeta=own_team_latent_variable,
                            opp_Zeta=opp_grid_point,
                            return_std=False,
                            return_cov=False
                        )[0, self.selected_performance]
                grid_predicted_performance = np.mean(mesh_grid_mapping, axis=1)

            self.lower_ukr.set_params_imshow(params={'cmap': self.cmap_ccp})
            self.lower_ukr.is_middle_color_zero = self.is_ccp_middle_color_zero
            self.lower_ukr._set_grid_values_to_draw(grid_values=grid_predicted_performance)
        else:
            self._reset_simulation()

    def _reset_simulation(self):
        self.is_simulate_changing_member = False
        self.deleted_member_coordinates = None
        self.grid_simulated_latent_variables = None
        self.lower_ukr.set_scatter_cross(None)
        self.own_ukr_kde.set_mesh_in_latent_space(None)

