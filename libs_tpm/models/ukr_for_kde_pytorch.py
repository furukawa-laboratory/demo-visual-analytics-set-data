import torch
import numpy as np
from tqdm import tqdm
import math


class UKRForWeightedKDEPyTorch():
    def __init__(self, weight_of_group, member_features, n_embedding,
                 bandwidth_kde, bandwidth_nadaraya,
                 is_compact, lambda_, evaluation_kldiv_method,
                 resolution_quadrature=None,
                 init='random', is_save_history=True):
        if weight_of_group.shape[1] != member_features.shape[0]:
            raise ValueError('group_features.shape[1] and member_features[0] must be match.')
        if np.any(weight_of_group < 0.0):
            raise ValueError('weight_of_group bust be non-negtive')

        # set input
        self.normalized_weight_of_group = torch.tensor(weight_of_group / weight_of_group.sum(axis=1)[:, None])
        self.member_features = torch.tensor(member_features)
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

        if evaluation_kldiv_method in ['jensen', 'quadrature_by_parts']:
            self.evaluation_kldiv_method = evaluation_kldiv_method
            if evaluation_kldiv_method == 'quadrature_by_parts':
                if isinstance(resolution_quadrature, int) and resolution_quadrature > 0:
                    self.resolution_quadrature = resolution_quadrature
                else:
                    raise ValueError(
                        'if kl divergence is evaluated using quadrature by parts, resolution_quadrature must be set integer')
        else:
            raise ValueError('{} is not supported to evaluate kl divergence'.format(evaluation_kldiv_method))

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = torch.tensor(np.random.normal(0,
                                                   self.bandwidth_nw * 0.5,
                                                   (self.n_groups, self.n_embedding)
                                                   ),
                                  requires_grad=True)
        elif isinstance(init, np.ndarray) and init.shape == (self.n_groups, self.n_embedding):
            self.Z = torch.tensor(init.copy(), requires_grad=True)
        else:
            raise ValueError("invalid init: {}".format(init))

        self.is_save_history = is_save_history
        if self.is_save_history:
            self.history = {}
        self._done_fit = False

    def fit(self, n_epoch, learning_rate, verbose=True):
        if self.is_save_history:
            self.history['z'] = np.zeros((n_epoch, self.n_groups, self.n_embedding))
            self.history['zvar'] = np.zeros((n_epoch, self.n_embedding))
            self.history['obj_func'] = np.zeros(n_epoch)
            self.history['obj_func_numpy'] = np.zeros(n_epoch)

        if verbose:
            bar = tqdm(range(n_epoch))
        else:
            bar = range(n_epoch)

        if self.evaluation_kldiv_method == 'jensen':
            all_members_sqeuclid = torch.cdist(self.member_features, self.member_features, p=2) ** 2.0
            cross_entropy_gaussians = 0.5 * (self.precision_kde * all_members_sqeuclid
                                             + self.n_features * (1.0
                                                                  + math.log(self.bandwidth_kde ** 2)
                                                                  + math.log(2 * math.pi))
                                             )
            upper_bound_cross_entropy_kdes = torch.einsum("tn,im,nm->ti",
                                                          self.normalized_weight_of_group,
                                                          self.normalized_weight_of_group,
                                                          cross_entropy_gaussians)
        elif self.evaluation_kldiv_method == 'quadrature_by_parts':
            from libs_tpm.models.kde import KDE
            from somf.libs.tools.create_zeta import create_zeta
            grid_points, self.step = create_zeta(zeta_min=self.member_features.min() - 2.0 * self.bandwidth_kde,
                                                      zeta_max=self.member_features.max() + 2.0 * self.bandwidth_kde,
                                                      latent_dim=self.n_features,
                                                      resolution=self.resolution_quadrature,
                                                      include_min_max=False, return_step=True)
            self.grid_points = torch.tensor(grid_points)
            # distance = grid_points[:, None, :] - self.member_features[None, :, :]
            # energy = -0.5 * torch.sum(distance * distance, dim=2)
            # gauss_func = torch.exp(self.precision_kde * energy)
            # kernel = gauss_func / math.sqrt((2.0 * np.pi * self.bandwidth_kde * self.bandwidth_kde) ** self.n_features)
            # data_densities = torch.sum(kernel[None, :, :] * self.normalized_weight_of_group[:, None, :], dim=2)
            self.data_densities = self._kde(x=self.grid_points,
                                      datasets=self.member_features,
                                      weights=self.normalized_weight_of_group)
            # kde = KDE()
            # kde.fit(dataset=self.member_features.detach().numpy(),
            #         bandwidth=self.bandwidth_kde,
            #         weights=self.normalized_weight_of_group.detach().numpy())
            # self.data_densities = torch.tensor(kde.pdf(representative_points))

        for epoch in bar:
            sqeuclidean_zz = torch.cdist(self.Z, self.Z, p=2) ** 2.0
            H = torch.exp(-0.5 * self.precision_nadaraya * sqeuclidean_zz)
            G = H.sum(1).unsqueeze(1)
            GInv = torch.reciprocal(G)
            R = H * GInv

            if self.evaluation_kldiv_method == 'jensen':
                obj_func = torch.sum(R * upper_bound_cross_entropy_kdes)
                obj_func_numpy = np.sum(R.detach().numpy() * upper_bound_cross_entropy_kdes.detach().numpy())
            elif self.evaluation_kldiv_method == 'quadrature_by_parts':
                approximated_densities = torch.mm(R, self.data_densities)
                obj_func = -torch.sum(self.data_densities * torch.log(approximated_densities)) * (self.step ** self.n_features)
                obj_func_numpy = -np.sum(
                    self.data_densities.detach().numpy() * np.log(approximated_densities.detach().numpy())) * (
                                             self.step ** self.n_features)
            obj_func += self.lambda_ * torch.sum(self.Z ** 2.0)
            obj_func.backward()

            with torch.no_grad():
                self.Z = self.Z - learning_rate * self.Z.grad
                if self.is_compact:
                    self.Z = torch.clamp(self.Z, -1.0, 1.0)
                else:
                    self.Z = self.Z - self.Z.mean(0)
            self.Z.requires_grad = True

            if self.is_save_history:
                self.history['z'][epoch] = self.Z.detach().numpy()
                self.history['zvar'][epoch] = torch.mean((self.Z - self.Z.mean(0)) ** 2, dim=0).detach().numpy()
                self.history['obj_func'][epoch] = obj_func.item()
                self.history['obj_func_numpy'][epoch] = obj_func_numpy

        self._done_fit = True
        if self.is_save_history:
            return self.history

    def _kde(self, x, datasets, weights):
        if weights.ndim == 1:
            normalized_weights = weights.reshape(1,-1)
        elif weights.ndim == 2:
            normalized_weights = weights
        else:
            raise ValueError('invalid weigths={}'.format(weights))
        normalized_weights = normalized_weights / normalized_weights.sum(dim=1)[:,None]
        distance = x[:, None, :] - datasets[None, :, :] # KxNxD
        energy = -0.5 * torch.sum(distance * distance, dim=2) # KxN
        gauss_func = torch.exp(self.precision_kde * energy)
        kernel = gauss_func / math.sqrt((2.0 * np.pi * self.bandwidth_kde * self.bandwidth_kde) ** self.n_features)
        densities = torch.einsum("kn,tn->tk", kernel, normalized_weights)
        return densities

    def transform(self, weights_of_group, n_epoch, learning_rate, verbose=True):
        if isinstance(weights_of_group, np.ndarray):
            weights_of_group = torch.tensor(weights_of_group)
        if self.is_save_history:
            self.history['znew'] = np.zeros((n_epoch,
                                             weights_of_group.shape[0],
                                             self.n_embedding))
            self.history['obj_func_znew'] = np.zeros(n_epoch)
        if verbose:
            bar = tqdm(range(n_epoch))
        else:
            bar = range(n_epoch)

        target_densities = self._kde(x=self.grid_points,
                                     datasets=self.member_features,
                                     weights=weights_of_group)
        cross_entropy = -torch.sum(target_densities[:,None,:] * torch.log(self.data_densities)[None,:,:],
                                   dim=2)
        nearest_data = torch.argmin(cross_entropy, dim=1)
        Znew = self.Z[nearest_data, :].clone().detach().requires_grad_(True)
        for epoch in bar:
            Delta = Znew[:, None, :] - self.Z.clone().detach()[None, :, :]
            sqdist = torch.sum(Delta ** 2, dim=2)
            H = torch.exp(-0.5 * self.precision_nadaraya * sqdist)
            G = torch.sum(H, dim=1)[:, None]
            GInv = torch.reciprocal(G)
            R = H * GInv
            model_densities = torch.mm(R, self.data_densities)
            obj_func = -torch.sum(target_densities * torch.log(model_densities))
            obj_func = obj_func * (self.step ** self.n_embedding)
            obj_func.backward()

            with torch.no_grad():
                Znew = Znew - learning_rate * Znew.grad
                if self.is_compact:
                    Znew = torch.clamp(Znew, -1.0, 1.0)
                else:
                    pass
            Znew.requires_grad = True

            if self.is_save_history:
                self.history['znew'][epoch] = Znew.detach().numpy()
                self.history['obj_func_znew'][epoch] = obj_func.item()

        return Znew.clone().detach().numpy()

