import numpy as np
from gmm_net.models.own_opp_team_regressor import OwnTeamOppTeamRegressor
from gmm_net.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from gmm_net.models.ukr_for_kde import UKRForWeightedKDE as UKRKDE
from gmm_net.models.gplvm import GPLVM
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings


class OwnTeamOppTeamPerformanceMultiLevelViewMM(OwnTeamOppTeamRegressor):
    def __init__(self,
                 win_team_bag_of_members: np.ndarray,
                 lose_team_bag_of_members: np.ndarray,
                 win_team_performance: np.ndarray,
                 lose_team_performance: np.ndarray,
                 member_features: np.ndarray,
                 params_lower_ukr: dict,
                 params_upper_ukr_kde: dict,
                 params_gplvm: dict,
                 params_gpr: dict,
                 is_save_history=False,
                 is_compact=True):
        '''
        Parameters
        ----------
        win_team_bag_of_members: np.ndarray
            shape = (n_games, n_members)
        lose_team_bag_of_members
            shape = (n_games, n_members)
        win_team_performance
            shape = (n_games, n_performance)
        lose_team_performance
            shape = (n_games, n_performance)
        member_features
            shape = (n_games, n_features)
        params_lower_ukr
            A dictionary of parameters about ukr to estimate latent variables of member features and
            mapping from member latent space to space of performance.
        params_upper_ukr_kde
            A dictionary of parameters about ukr for densities to estimate latent variables of team and
            mapping from team latent space to space of dinsities.
        params_gplvm
            A dictionary of parameters gplvm to estimate latent variables of performance and
            mapping from latent space to space of performance.
        params_gpr
            A dictionary of parameters GaussianProcessRegression to color own team latent space,
            opp one, performance bars. This GPR is to adjust lengthscale of RBF and beta inverse
            representing noise variance after learning latent variables of team.
            Therefore the kernel and other args can not to be passed.
        is_save_history: bool
        is_compact: bool
        '''
        # own opp team regressorと同様の__init__をまずは実行
        super().__init__(win_team_bag_of_members=win_team_bag_of_members,
                         lose_team_bag_of_members=lose_team_bag_of_members,
                         win_team_performance=win_team_performance,
                         lose_team_performance=lose_team_performance,
                         member_features=member_features,
                         params_lower_ukr=params_lower_ukr,
                         params_upper_ukr_kde=params_upper_ukr_kde,
                         params_gpr=params_gpr,
                         init_upper_ukr_kde=None)

        if self.lower_ukr.is_compact != is_compact:
            warnings.warn(
                "'is_compact' set in params_ukr = {} ".format(is_compact)
                + "and one set in this constructor ={} are not match. Is it OK?".format(is_compact)
            )
        self.n_teams = self.n_games * 2
        del self.gpr

        self.params_gplvm = params_gplvm.copy()
        self.params_gpr = params_gpr.copy()

        list_error_keys = ['is_compact', 'is_save_history', 'X']
        for key in list_error_keys:
            if key in self.params_upper_ukr_kde.keys():
                raise ValueError("Do not set '{}' in params_upper_ukr_kde".format(key))
            if key in self.params_gplvm.keys():
                raise ValueError("Do not set '{}' in params_gplvm".format(key))
        if 'n_components' in params_gplvm:
            raise ValueError("Do not set 'n_components' in params_gplvm")

        if 'kernel' in params_gpr.keys():
            raise ValueError('Do not specify kernel in params_gpr')

        for key in ['is_optimize_sqlength', 'is_optimize_beta_inv']:
            if self.params_gplvm[key]:
                raise ValueError(
                    "Not support {} = {}".format(
                        key,
                        self.params_gplvm[key]
                    )
                )
            else:
                pass

        self.is_compact = is_compact
        self.params_upper_ukr_kde['is_compact'] = is_compact
        self.params_gplvm['is_compact'] = is_compact

        self.is_save_history = is_save_history
        self.params_upper_ukr_kde['is_save_history'] = is_save_history
        self.params_gplvm['is_save_history'] = is_save_history

        if self.is_save_history:
            self.history = {}

    def fit(self, nb_epoch_lower_ukr, eta_lower_ukr,
            nb_epoch_multiview_mm, eta_multiview_mm,
            ratio_ukr_for_kde, lower_ukr_fit=None,
            verbose=True):
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

        # fit multiview manifold modeling
        self._fit_multiview_mm(nb_epoch=nb_epoch_multiview_mm,
                               learning_rate=eta_multiview_mm,
                               ratio_ukr_for_kde=ratio_ukr_for_kde,
                               verbose=verbose)

        # set gplvm parameter to gpr parameter
        kernel = (
                RBF(length_scale=np.sqrt(np.exp(self.gplvm.sigma)))
                + WhiteKernel(noise_level=np.exp(self.gplvm.eta))
        )
        self.params_gpr['kernel'] = kernel
        self.gpr = GaussianProcessRegressor(**self.params_gpr)
        self._fit_gpr()

    def _fit_multiview_mm(self, nb_epoch, learning_rate, ratio_ukr_for_kde, verbose):
        if ratio_ukr_for_kde >= 0.0 and ratio_ukr_for_kde <= 1.0:
            pass
        else:
            raise ValueError('Must be 0.0 <= ratio_ukr_for_kde <= 1.0')
        # set up ukr for kde
        self.params_upper_ukr_kde['member_features'] = self.lower_ukr.Z.copy()
        self.upper_ukr_kde = UKRKDE(**self.params_upper_ukr_kde)
        self.Z = self.upper_ukr_kde.Z.copy()

        # setup gplvm
        self.params_gplvm['X'] = self.training_performance
        self.params_gplvm['init'] = self._duplicate_to_input_gplvm(array=self.Z)
        self.params_gplvm['n_components'] = self.upper_ukr_kde.n_embedding * 2
        self.gplvm = GPLVM(**self.params_gplvm)

        if self.is_save_history:
            self.history['z'] = np.zeros((nb_epoch,
                                          self.n_teams,
                                          self.upper_ukr_kde.n_embedding))
            self.history['obj_func'] = np.zeros(nb_epoch)
            self.history['obj_func_ukr'] = np.zeros(nb_epoch)
            self.history['nega_obj_func_gplvm'] = np.zeros(nb_epoch)
            self.history['mean_grad_norm_ukr'] = np.zeros(nb_epoch)
            self.history['mean_grad_norm_gplvm'] = np.zeros(nb_epoch)
            self.history['mean_grad_norm'] = np.zeros(nb_epoch)

        self.nb_epoch = nb_epoch
        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        # update latent variables iteratively
        for epoch in bar:
            grad_ukr, obj_func_ukr = self.upper_ukr_kde.grad(return_obj_func_value=True)
            mean_grad_norm_ukr = np.mean(np.sqrt(np.sum(np.square(grad_ukr), axis=1)))

            dict_grad_gplvm = self.gplvm.grad(return_obj_func=True)
            grad_gplvm = dict_grad_gplvm['z']
            obj_func_gplvm = dict_grad_gplvm['obj_func']
            nega_obj_func_gplvm = -0.5 * obj_func_gplvm

            # mean to calculate value to update
            mean_grad_gplvm = self._unduplicate_to_input_ukr(grad_gplvm)
            nega_mean_grad_gplvm = -mean_grad_gplvm
            mean_grad_norm_gplvm = np.mean(np.sqrt(np.sum(np.square(nega_mean_grad_gplvm), axis=1)))
            if mean_grad_norm_gplvm > mean_grad_norm_ukr:
                nega_mean_grad_gplvm /= mean_grad_norm_gplvm
                nega_mean_grad_gplvm *= mean_grad_norm_ukr
                min_mean_grad_norm = mean_grad_norm_ukr
            else:
                grad_ukr /= mean_grad_norm_gplvm
                grad_ukr *= mean_grad_norm_gplvm
                min_mean_grad_norm = mean_grad_norm_gplvm

            dFdZ = ratio_ukr_for_kde * grad_ukr + (1.0 - ratio_ukr_for_kde) * nega_mean_grad_gplvm
            self.Z = self.Z - learning_rate * dFdZ
            if self.is_compact:
                self.Z = np.clip(self.Z, -1.0, 1.0)
            else:
                pass

            if self.is_save_history:
                self.history['z'][epoch] = self.Z
                self.history['obj_func'][epoch] = (
                        ratio_ukr_for_kde * obj_func_ukr
                        + (1.0 - ratio_ukr_for_kde) * nega_obj_func_gplvm
                )
                self.history['obj_func_ukr'][epoch] = obj_func_ukr
                self.history['nega_obj_func_gplvm'][epoch] = nega_obj_func_gplvm
                self.history['mean_grad_norm_ukr'][epoch] = ratio_ukr_for_kde * min_mean_grad_norm
                self.history['mean_grad_norm_gplvm'][epoch] = (1.0 - ratio_ukr_for_kde ) * min_mean_grad_norm
                self.history['mean_grad_norm'][epoch] = np.mean(np.sqrt(np.sum(np.square(dFdZ), axis=1)))

            # set Z instances to prepare next iteration
            self.upper_ukr_kde.set_z(self.Z)
            self.gplvm.set_z(self._duplicate_to_input_gplvm(self.Z))

    def _duplicate_to_input_gplvm(self, array: np.ndarray):
        if array.ndim != 2:
            raise ValueError('input array must be 2 dim')

        if array.shape[0] == self.n_teams:
            array_win = array[:self.n_games, :].copy()
            array_lose = array[self.n_games:, :].copy()
            array_own = np.concatenate([array_win,
                                        array_lose], axis=0)
            array_opp = np.concatenate([array_lose,
                                        array_win], axis=0)
            array_duplicated = np.concatenate([array_own, array_opp], axis=1)
            return array_duplicated
        else:
            raise ValueError(
                'if array.shape[0] must match number of games * 2'
            )
    def _unduplicate_to_input_ukr(self, array: np.ndarray):
        if array.ndim != 2:
            raise ValueError('input array must be 2 dim')

        array_own_win = array[:self.n_games, :self.upper_ukr_kde.n_embedding].copy()
        array_own_lose = array[self.n_games:, :self.upper_ukr_kde.n_embedding].copy()
        array_opp_lose = array[:self.n_games, self.upper_ukr_kde.n_embedding:].copy()
        array_opp_win = array[self.n_games:, self.upper_ukr_kde.n_embedding:].copy()
        unduplicated_array_win = 0.5 * (array_own_win + array_opp_win)
        unduplicated_array_lose = 0.5 * (array_own_lose + array_opp_lose)
        unduplicated_array = np.concatenate([unduplicated_array_win,
                                             unduplicated_array_lose], axis=0)
        return unduplicated_array
