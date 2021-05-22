import numpy as np
from tqdm import tqdm
import torch
import pandas as pd


class GradMinimizerBase():
    def __init__(self, energy_fn, protein, num_steps=1000, log_interval=10):
        self.energy_fn = energy_fn
        self.protein = protein

        self.optimizer = None

        self.x_best = self.protein.coords
        self.energy_best = protein.get_energy(energy_fn).item()

        self.sample = []
        self.sample_energy = []

        self.num_steps = num_steps
        self.log_interval = log_interval

    def _step(self):
        raise NotImplementedError

    def run(self):
        for i in tqdm(range(self.num_steps)):
            energy = self._step()

            current_energy = energy.detach().item()
            if current_energy < self.energy_best:
                self.energy_best = current_energy
                self.x_best = self.protein.coords.detach().clone()

            if i % self.log_interval == 0:
                self.sample.append(self.protein.coords.detach().cpu().clone())
                self.sample_energy.append(current_energy)
                print(f'Step:{i}, Energy:{current_energy:.2f}')


class GradMinimizerCartesian(GradMinimizerBase):
    def __init__(self, energy_fn, protein, lr=3e-2, momentum=0.9, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)

        # params = {"lr": 3e-2, "momentum": 0.0}
        params = {"lr": lr, "momentum": momentum}

        x = self.protein.coords

        x.requires_grad_()
        # self.optimizer = torch.optim.Adam([x], **params)
        self.optimizer = torch.optim.SGD([x], **params)

    def _step(self):
        self.optimizer.zero_grad()
        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        # energy.backward(retain_graph=True)
        if torch.isnan(self.protein.coords.grad).sum() > 0:
            print('coords_grad is nan')
        else:
            self.optimizer.step()
        return energy


class GradMinimizerInternal(GradMinimizerBase):
    def __init__(self, energy_fn, protein, lr=3e-3, momentum=0.9, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)

        # params = {"lr": 3e-3, "momentum": 0.9}
        params = {"lr": lr, "momentum": momentum}

        self.protein.update_internal_from_cartesian()
        x = self.protein.coords_int
        x.requires_grad_()
        # self.optimizer = torch.optim.Adam([x], **params)
        self.optimizer = torch.optim.SGD([x], **params)

    def _step(self):

        self.optimizer.zero_grad()

        self.protein.update_cartesian_from_internal()
        energy = self.protein.get_energy(self.energy_fn)

        energy.backward()
        # energy.backward(retain_graph=True)
        # print(self.protein.coords_int.grad)
        self.optimizer.step()
        return energy


class GradMinimizerMixed(GradMinimizerBase):
    def __init__(self, energy_fn, protein, lr=3e-3, momentum=0.9, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)

        # self.params_int = {"lr": 1e-1, "momentum": 0.9}
        # self.params = {"lr": 1e-1, "momentum": 0.9}
        self.params_int = {"lr": lr, "momentum": momentum}
        self.params = {"lr": lr, "momentum": momentum}
        self.cart_per_int = 0

        self.protein.update_internal_from_cartesian()

        # x = self.protein.coords_int
        # y = self.protein.coords
        # x.requires_grad_()
        # y.requires_grad_()
        # self.optimizer_x = torch.optim.SGD([x], **params)
        # self.optimizer_y = torch.optim.SGD([y], **params)

    def _step(self):
        lr = self.params['lr']
        lr_int = self.params_int['lr']

        self.protein.coords_int.requires_grad_()
        self.protein.update_cartesian_from_internal()

        # self.protein.coords is not leaf, so we need to set retain_grad=True.
        self.protein.coords.retain_grad()

        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        grad_coords = self.protein.coords.grad.clone()

        with torch.no_grad():
            # self.protein.coords_int -= lr_int * self.protein.coords_int.grad
            self.protein.coords_int = self.protein.coords_int - lr_int * self.protein.coords_int.grad
            self.protein.update_cartesian_from_internal()
            # self.protein.coords -= lr * grad_coords
            self.protein.coords = self.protein.coords - lr * grad_coords
            self.protein.update_internal_from_cartesian()

        # for i in range(self.cart_per_int):
        #     self.protein.coords.requires_grad_()
        #
        #     energy = self.protein.get_energy(self.energy_fn)
        #     energy.backward()
        #
        #     grad_coords = self.protein.coords.grad.clone()
        #
        #     if torch.isnan(self.protein.coords.grad).sum() > 0:
        #         print('coords_grad is nan')
        #     else:
        #         with torch.no_grad():
        #             self.protein.coords = self.protein.coords - lr * grad_coords

        # self.protein.coords.requires_grad_()
        # energy = self.protein.get_energy(self.energy_fn)
        # energy.backward()
        #
        # with torch.no_grad():
        #     self.protein.coords -= lr * self.protein.coords.grad
        #     self.protein.update_internal_from_cartesian()

        return energy


class GradMinimizerIntFast(GradMinimizerBase):
    def __init__(self, energy_fn, protein, lr=3e-3, momentum=0.9, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.params = {"lr": lr, "momentum": momentum}
        self.protein.update_internal_from_cartesian()

    def _step(self):
        # use fast calculation of dx and dz
        self.protein.coords.requires_grad_()
        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        gradx = self.protein.coords.grad.clone()
        gradz = self.protein.get_gradz_from_gradx(self.protein.coords, gradx)
        # print(gradz)
        with torch.no_grad():
            dz = -1.0 * self.params['lr'] * gradz
            # print(dz.max(dim=0), dz.min(dim=0))
            dx = self.protein.get_dx_from_dz(self.protein.coords, dz)
            self.protein.coords = self.protein.coords + dx
            self.protein.update_internal_from_cartesian()

        return energy


class GradMinimizerMixFast(GradMinimizerBase):
    def __init__(self, energy_fn, protein, lr=3e-3, momentum=0.9, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.params = {"lr": lr, "momentum": momentum}
        self.protein.update_internal_from_cartesian()

    def _step(self):
        # use fast calculation of dx and dz
        lr = self.params['lr']
        self.protein.coords.requires_grad_()
        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        gradx = self.protein.coords.grad.clone()
        # use the cartesian step to update the coords and calculate gradz using the updated coords
        dx_cart = -1.0 * lr * gradx
        coords_c = self.protein.coords + dx_cart
        gradz = self.protein.get_gradz_from_gradx(coords_c, gradx)
        # print(gradz)
        with torch.no_grad():
            dz = -1.0 * self.params['lr'] * gradz
            # print(dz.max(dim=0), dz.min(dim=0))
            dx = self.protein.get_dx_from_dz(self.protein.coords, dz)
            # apply both the cartesian step and the internal step
            self.protein.coords = self.protein.coords + dx_cart + dx
            self.protein.update_internal_from_cartesian()

        return energy


class GradMinimizerProfile():
    def __init__(self, energy_fn, protein, num_steps=1000, log_interval=10):
        self.energy_fn = energy_fn
        self.protein = protein

        self.energy_best = protein.get_energy(energy_fn).item()

        self.sample = []
        self.sample_energy = []

        self.num_steps = num_steps
        self.log_interval = log_interval

        self.params = {"lr": 1e-2, "momentum": 0.9}

        profile_tr = self.protein.profile
        df = pd.read_csv('data/aa_freq.csv')
        aa_freq = df['freq'].values / df['freq'].sum()
        self.aa_freq = torch.tensor(aa_freq, dtype=torch.float, device=profile_tr.device)

        self.profile = profile_tr * self.aa_freq / (1 - profile_tr)

        self.x_best = self.profile

    def _step(self):
        self.profile.requires_grad_()

        self.protein.profile = self.profile / (self.profile + self.aa_freq)

        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        # print(self.profile.grad.min(), self.protein.profile.grad.max())
        # print(self.profile.grad)

        with torch.no_grad():
            if torch.isnan(self.profile.grad).sum() > 0:
                print('profile_grad is nan')
            else:
                self.profile -= self.params['lr'] * self.profile.grad
                # clip profile, normalize profile
                self.profile = torch.clamp(self.profile, min=0, max=1.0)
                noise = torch.rand_like(self.profile, device=self.profile.device) * 0.001

                self.profile += noise

                self.profile = self.profile / self.profile.sum(dim=-1, keepdim=True)

        return energy

    def run(self):
        for i in tqdm(range(self.num_steps)):
            energy = self._step()

            current_energy = energy.detach().item()
            if current_energy < self.energy_best:
                self.energy_best = current_energy
                self.x_best = self.profile.detach().clone()

            if i % self.log_interval == 0:
                self.sample.append(self.profile.detach().cpu().clone())
                self.sample_energy.append(current_energy)
                print(f'Step:{i}, Energy:{current_energy:.2f}')


