import numpy as np
from tqdm import tqdm
import scipy as sp


class GPLVM():
    def __init__(self, X: np.ndarray, n_components: int,
                 sqlength, beta_inv,
                 is_compact=False, init='random',
                 is_optimize_sqlength=True, is_optimize_beta_inv=True,
                 is_save_history=False,
                 how_calculate_inv='inv'):
        self.X = X.copy()
        self.S = X @ X.T
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_components = n_components
        self.sigma = np.log(sqlength)
        self.eta = np.log(beta_inv)
        self.is_compact = is_compact
        self.is_optimize_sqlength = is_optimize_sqlength
        self.is_optimize_beta_inv = is_optimize_beta_inv

        self.L = self.n_components
        self.D = self.n_features

        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0.0, 0.1 * np.sqrt(sqlength),
                                      (self.n_samples, self.n_components))
        elif isinstance(init, np.ndarray) and init.shape == (self.n_samples, self.n_components):
            self.Z = init.copy()
        else:
            raise ValueError('invalid init: {}'.format(init))

        self.is_save_history = is_save_history
        if how_calculate_inv in ['inv', 'cholesky']:
            self.how_calculate_inv = how_calculate_inv
        else:
            raise ValueError('invalid how_calculate_inv = {}'.format(how_calculate_inv))
        if is_save_history:
            self.history = {}
        self.is_done_fit = False
        self.is_done_calcF = False

    def fit(self, n_epoch, verbose=True,
            learning_rate_x=0.5,
            learning_rate_sigma=0.0025,
            learning_rate_eta=0.00005):
        self.nb_epoch = n_epoch

        if self.is_save_history:
            self.history['z'] = np.zeros((n_epoch, self.n_samples, self.n_components))
            self.history['obj_func'] = np.zeros(n_epoch)
            self.history['sqlength'] = np.zeros(n_epoch)
            self.history['length'] = np.zeros(n_epoch)
            self.history['beta_inv'] = np.zeros(n_epoch)
            self.history['mean_grad_norm'] = np.zeros(n_epoch)

        if verbose:
            bar = tqdm(range(n_epoch))
        else:
            bar = range(n_epoch)

        for epoch in bar:
            dict_grad = self.grad()
            dLdZ = dict_grad['z']
            self.Z += learning_rate_x * dLdZ
            if self.is_compact:
                self.Z = np.clip(self.Z, -1.0, 1.0)
            else:
                pass

            if self.is_optimize_sqlength:
                dLdsigma = dict_grad['sigma']
                self.sigma += learning_rate_sigma * dLdsigma
            if self.is_optimize_beta_inv:
                dLdeta = dict_grad['eta']
                self.eta += learning_rate_eta * dLdeta

            if self.is_save_history:
                self.history['z'][epoch] = self.Z.copy()
                self.history['obj_func'][epoch] = dict_grad['obj_func']
                self.history['sqlength'][epoch] = np.exp(self.sigma)
                self.history['length'][epoch] = np.sqrt(np.exp(self.sigma))
                self.history['beta_inv'][epoch] = np.exp(self.eta)
                self.history['mean_grad_norm'][epoch] = np.mean(np.sqrt(np.sum(np.square(dLdZ), axis=1)))

        self.is_done_fit = True

    def grad(self, return_obj_func=True):
        delta = self.Z[:, None, :] - self.Z[None, :, :]
        distance = np.sum(np.square(delta), axis=2)
        K = np.exp(-0.5 * distance / np.exp(self.sigma))
        K += np.exp(self.eta) * np.eye(self.n_samples)
        if self.how_calculate_inv == 'inv':
            Kinv = np.linalg.inv(K)
        elif self.how_calculate_inv == 'cholesky':
            Lower = sp.linalg.cholesky(K, lower=True)
            Kinv = sp.linalg.cho_solve((Lower, True),
                                       np.eye(Lower.shape[0]))
        else:
            raise ValueError('invalid how_calculate_inv={}'.format(self.how_calculate_inv))

        G = 0.5 * (Kinv @ self.S @ Kinv - self.n_features * Kinv)
        dKndXn = - K[:, :, None] * delta / np.exp(self.sigma)
        dLdX = 2.0 * np.einsum('nm,nml->nl', G, dKndXn)

        dKdsigma = 0.5 * (K - np.exp(self.eta) * np.eye(self.n_samples))
        dKdsigma = dKdsigma * distance / np.exp(self.sigma)

        dLdsigma = np.sum(G * dKdsigma)

        dKdeta = np.exp(self.eta) * np.eye(self.n_samples)
        dLdeta = np.sum(G * dKdeta)

        dict_grad = {}
        dict_grad['z'] = dLdX
        dict_grad['sigma'] = dLdsigma
        dict_grad['eta'] = dLdeta
        if return_obj_func:
            obj_func = (
                    -0.5 * self.n_samples * self.n_features * np.log(2.0 * np.pi)
                    - 0.5 * self.n_features * np.log(np.linalg.det(K))
                    - 0.5 * np.trace(Kinv @ self.S)
            )
            dict_grad['obj_func'] = obj_func

        return dict_grad

    def inverse_transform(self, Zstar, return_cov=False):
        return self._predict_by_gpr(Zstar=Zstar, return_cov=return_cov)

    def _predict_by_gpr(self, Zstar, return_cov, Z=None, sqlength=None, beta_inv=None):
        if Zstar.ndim == 1:
            if Zstar.shape[0] == self.n_components:
                Zstar = Zstar.reshape(1, -1).copy()
            else:
                raise ValueError('invalid shape Zstar={}'.format(Zstar.shape))

        elif Zstar.ndim == 2:
            if Zstar.shape[1] == self.n_components:
                pass
            else:
                raise ValueError('invalid shape Zstar={}'.format(Zstar.shape))
        else:
            raise ValueError('invalid Zstar={}'.format(Zstar))

        if Z is None:
            Z = self.Z.copy()
        elif isinstance(Z, np.ndarray):
            if Z.shape == self.Z.shape:
                pass
            else:
                raise ValueError('invalid Z={}'.format(Z))
        else:
            raise ValueError('invalid Z={}'.format(Z))

        if sqlength is None:
            sigma = self.sigma
        elif isinstance(sqlength, float):
            sigma = np.log(sqlength)
        else:
            raise ValueError('invalid sqlength={}'.format(sqlength))
        if beta_inv is None:
            eta = self.eta
        elif isinstance(beta_inv, float):
            eta = np.log(beta_inv)
        else:
            raise ValueError('invalid beta_inv={}'.format(beta_inv))

        delta_star = Zstar[:, None, :] - Z[None, :, :]
        distance_star = np.sum(np.square(delta_star), axis=2)
        Kstar = np.exp(-0.5 * distance_star / np.exp(sigma))
        delta = Z[:, None, :] - Z[None, :, :]
        distance = np.sum(np.square(delta), axis=2)
        K = np.exp(-0.5 * distance / np.exp(sigma))
        K += np.exp(eta) * np.eye(Z.shape[0])
        if self.how_calculate_inv == 'inv':
            Kinv = np.linalg.inv(K)
        elif self.how_calculate_inv == 'cholesky':
            Lower = sp.linalg.cholesky(K, lower=True)
            Kinv = sp.linalg.cho_solve((Lower, True),
                                       np.eye(Lower.shape[0]))
        else:
            raise ValueError('invalid how_calculate_inv={}'.format(self.how_calculate_inv))

        Y_star_mean = Kstar @ Kinv @ self.X
        if return_cov:
            delta_star_star = Zstar[:, None, :] - Zstar[None, :, :]
            distance_star_star = np.sum(np.square(delta_star_star), axis=2)
            Kstar_star = np.exp(-0.5 * distance_star_star / np.exp(sigma))

            Y_star_cov = Kstar_star - Kstar @ Kinv @ Kstar.T

            return Y_star_mean, Y_star_cov
        else:
            return Y_star_mean

    def set_z(self, Z):
        if isinstance(Z, np.ndarray):
            self.Z = Z.copy()
        else:
            raise ValueError('invalid Z={}'.format(Z))

    def calcF(self, resolution, size='auto', verbose=False):
        if self.is_done_calcF:
            pass
        else:
            if not self.is_done_fit:
                raise ValueError("fit is not done")
            if self.is_save_history is False:
                raise ValueError('is_save_history must be True')

            from somf.libs.tools.create_zeta import create_zeta
            self.resolution = resolution

            self.history['f'] = np.zeros(
                (
                    self.nb_epoch,
                    resolution ** self.n_components,
                    self.n_features
                )
            )

            if verbose:
                bar = tqdm(range(self.nb_epoch))
            else:
                bar = range(self.nb_epoch)

            for epoch in bar:
                Z = self.history['z'][epoch]
                sqlength = self.history['sqlength'][epoch]
                beta_inv = self.history['beta_inv'][epoch]
                if size == 'auto':
                    Xstar = create_zeta(Z.min(), Z.max(), self.n_components, resolution)
                else:
                    Xstar = create_zeta(size.min(), size.max(), self.n_components, resolution)

                self.history['f'][epoch] = self._predict_by_gpr(Zstar=Xstar,
                                                                return_cov=False,
                                                                Z=Z,
                                                                sqlength=sqlength,
                                                                beta_inv=beta_inv)

            self.is_done_calcF = True
