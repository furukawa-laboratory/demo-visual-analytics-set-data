import torch
import numpy as np
from tqdm import tqdm
import math
class GPLVMPyTorch():
    def __init__(self,
                 Y: np.ndarray,
                 LatentDim: int,
                 sqlength: float,
                 beta_inv: float,
                 X: np.ndarray,
                 is_save_history=True,
                 is_compact=True):
        self.Y = torch.tensor(Y.copy(), dtype=torch.float64)
        self.n_components = LatentDim
        self.n_features = Y.shape[1]
        self.n_samples = Y.shape[0]

        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X.copy(),
                                  requires_grad=True,
                                  dtype=torch.float64)
        else:
            raise ValueError('initial X must be input')

        self.sigma = torch.tensor(math.log(sqlength),
                                  requires_grad=True,
                                  dtype=torch.float64)
        self.alpha = torch.tensor(math.log(beta_inv),
                                  requires_grad=True,
                                  dtype=torch.float64)

        self.S = torch.mm(self.Y, self.Y.transpose(1, 0))
        self.is_save_history = is_save_history
        self.is_compact = is_compact
        if is_save_history:
            self.history = {}

    def fit(self, n_epoch=100, epsilonX=0.5,
            epsilon_length=0.0025, epsilon_alpha=0.00005,
            verbose=True):
        if self.is_save_history:
            self.history['X'] = np.zeros((n_epoch, self.n_samples, self.n_components))
            self.history['sigma'] = np.zeros(n_epoch)
            self.history['sigma_grad'] = np.zeros(n_epoch)
            self.history['alpha'] = np.zeros(n_epoch)
            self.history['obj_func'] = np.zeros(n_epoch)

        if verbose:
            bar = tqdm(range(n_epoch))
        else:
            bar = range(n_epoch)
        for epoch in bar:
            self.K = self._kernel(self.X,
                                  self.X,
                                  sigma=self.sigma,
                                  alpha=self.alpha)
            self.Kinv = torch.inverse(self.K)
            obj_func = (
                    -0.5 * self.n_samples * self.n_features * math.log(2.0 * math.pi)
                    -0.5 * self.n_features * torch.log(torch.det(self.K))
                    -0.5 * torch.trace(torch.mm(self.Kinv, self.S))
            )
            obj_func.backward()


            self.X.data =  self.X.data + epsilonX * self.X.grad.data
            self.sigma.data = self.sigma.data + epsilon_length * self.sigma.grad.data
            sigma_grad = self.sigma.grad.data
            self.alpha.data = self.alpha.data + epsilon_alpha * self.alpha.grad.data

            if self.is_save_history:
                self.history['X'][epoch] = self.X.detach().numpy()
                self.history['obj_func'][epoch] = obj_func.item()
                self.history['alpha'][epoch] = math.exp(self.alpha.item())
                self.history['sigma'][epoch] = math.exp(self.sigma.item())
                self.history['sigma_grad'][epoch] = sigma_grad.item()

            self.X.grad.data.zero_()
            self.sigma.grad.data.zero_()
            self.alpha.grad.data.zero_()
            # with torch.no_grad():
            #     self.X = self.X + epsilonX * self.X.grad
            #     sigma_grad = self.X.grad
            #     if self.is_compact:
            #         self.X = torch.clamp(self.X, -1.0, 1.0)
            #     else:
            #         pass
            #     self.ln_sqlength = self.ln_sqlength + (epsilon_length * self.ln_sqlength.grad)
            #     self.ln_alpha = self.ln_alpha + epsilon_alpha * self.ln_alpha.grad
            # self.X.requires_grad = True
            # self.ln_sqlength.requires_grad = True
            # self.ln_alpha.requires_grad = True




    def _kernel(self, X1, X2, sigma, alpha):
        #distance = torch.cdist(X1, X2, p=2) ** 2.0
        distance = torch.sum((X1[:, None, :] - X2[None, :, :])**2.0, dim=2)
        gauss = torch.exp(-0.5 * distance / torch.exp(sigma))
        K = gauss + torch.exp(alpha) * torch.eye(self.n_samples,
                                                 dtype=torch.float64)
        return K
