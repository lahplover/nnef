import numpy as np
from tqdm import tqdm
import torch


class DynamicsBase():
    def __init__(self, energy_fn, protein, num_steps=100000, log_interval=10):
        self.energy_fn = energy_fn
        self.protein = protein
        self.int_scale = 0.1 * torch.tensor([[0.1, 1.0, 1.0]], device=protein.coords.device)
        self.cart_scale = 50.0

        self.x_best = self.protein.coords
        self.energy_best = protein.get_energy(energy_fn).item()

        # self.params = {"alpha": 3e-2, "beta": 3e-2}

        self.sample = []
        self.sample_energy = []

        self.num_steps = num_steps
        self.log_interval = log_interval
        # x = self.protein.coords
        # x.requires_grad_()
        # self.optimizer = torch.optim.SGD([x], lr=1e-2, momentum=0.0)

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


class Dynamics(DynamicsBase):
    def __init__(self, energy_fn, protein, lr=3e-2, t_noise=3e-2, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.params = {"alpha": lr, "beta": t_noise}
        # move_map is a weights matrix with shape (N,); protein.coords has shape (N, 3)
        # fixed beads have weights=0; movable beads have weights=1.0;
        self.move_map = None

    def _step(self):
        # self.optimizer.zero_grad()
        self.protein.coords.requires_grad_()

        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        # print(self.protein.coords.grad[10])

        grad_coords = self.protein.coords.grad.clone()
        # print(grad_coords.max().item(), grad_coords.min().item())
        if self.move_map is not None:
            grad_coords = grad_coords * self.move_map[:, None]

        if torch.isnan(grad_coords).sum() > 0:
            print('coords_grad is nan')
            with torch.no_grad():
                noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
                self.protein.coords = self.protein.coords + self.params['beta'] * noise
        else:
            with torch.no_grad():
                noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
                self.protein.coords = self.protein.coords - self.params['alpha'] * grad_coords + self.params['beta'] * noise
                # grad_noise_scale = (self.params['alpha'] * grad_coords) / (self.params['beta'] * noise)
                # print(grad_coords)
                # print(torch.mean(torch.abs(grad_noise_scale)).item(), torch.median(torch.abs(grad_noise_scale)).item())
                # self.protein.coords -= self.params['alpha'] * grad_coords + self.params['beta'] * noise

        # self.optimizer.step()
        return energy

    # def _step(self):
    #     self.protein.coords.requires_grad_()
    #
    #     energy = self.protein.get_energy(self.energy_fn)
    #     energy.backward()
    #     grad_coords = self.protein.coords.grad.clone()
    #
    #     if torch.isnan(self.protein.coords.grad).sum() > 0:
    #         print('coords_grad is nan')
    #     else:
    #         with torch.no_grad():
    #             noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
    #             print(grad_coords.max().item(), grad_coords.min().item())
    #             self.protein.coords -= self.params['alpha'] * grad_coords + self.params['beta'] * noise
    #     return energy


class DynamicsInternal(DynamicsBase):
    def __init__(self, energy_fn, protein, lr=3e-2, t_noise=3e-2, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.protein.update_internal_from_cartesian()
        self.params = {"alpha": lr, "beta": t_noise}

    def _step(self):
        self.protein.coords_int.requires_grad_()

        self.protein.update_cartesian_from_internal()
        energy = self.protein.get_energy(self.energy_fn)

        energy.backward()

        grad_coords = self.protein.coords_int.grad
        # print(grad_coords.max().item(), grad_coords.min().item())

        if torch.isnan(grad_coords).sum() > 0:
            print('coords_grad is nan')
            with torch.no_grad():
                noise = torch.randn_like(self.protein.coords_int, device=self.protein.coords_int.device)
                dz = self.params['beta'] * noise * self.int_scale
                self.protein.coords_int = self.protein.coords_int + dz
        else:
            with torch.no_grad():
                noise = torch.randn_like(self.protein.coords_int, device=self.protein.coords_int.device)
                dz = self.int_scale * (-1.0 * self.params['alpha'] * grad_coords + self.params['beta'] * noise)
                self.protein.coords_int = self.protein.coords_int + dz

        return energy


class DynamicsMixed(DynamicsBase):
    def __init__(self, energy_fn, protein, lr=3e-2, t_noise=3e-2, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.protein.update_internal_from_cartesian()
        self.params = {"alpha": lr, "beta": t_noise}

    def _step(self):
        lr = self.params['alpha']
        t_noise = self.params['beta']
        # run one step in Cartesian space
        self.protein.coords_int.requires_grad_()
        self.protein.update_cartesian_from_internal()
        # self.protein.coords is not leaf, so we need to set retain_grad=True.
        self.protein.coords.retain_grad()
        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        grad_coords_int = self.protein.coords_int.grad.clone()
        grad_coords = self.protein.coords.grad.clone()
        with torch.no_grad():
            noise_int = torch.randn_like(self.protein.coords_int, device=self.protein.coords_int.device)
            if (torch.isnan(grad_coords_int).sum() > 0) | (torch.isnan(grad_coords).sum() > 0):
                print('coords_grad is nan')
                dz = t_noise * noise_int * self.int_scale
                self.protein.coords_int = self.protein.coords_int + dz
                self.protein.update_cartesian_from_internal()
            else:
                dz = self.int_scale * (-1.0 * lr * grad_coords_int + t_noise * noise_int)
                self.protein.coords_int = self.protein.coords_int + dz
                self.protein.update_cartesian_from_internal()
                noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
                self.protein.coords = self.protein.coords - lr * grad_coords + t_noise * noise
                self.protein.update_internal_from_cartesian()

        # # run one step in Cartesian space
        # self.protein.coords.requires_grad_()
        # energy = self.protein.get_energy(self.energy_fn)
        # energy.backward()
        # grad_coords = self.protein.coords.grad.clone()
        # with torch.no_grad():
        #     if torch.isnan(grad_coords).sum() > 0:
        #         print('coords_grad is nan')
        #         noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
        #         self.protein.coords = self.protein.coords + t_noise * noise
        #     else:
        #         noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
        #         self.protein.coords = self.protein.coords - lr * grad_coords + t_noise * noise
        #     self.protein.update_internal_from_cartesian()

        return energy


class DynamicsIntFast(DynamicsBase):
    def __init__(self, energy_fn, protein, lr=3e-2, t_noise=3e-2, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.protein.update_internal_from_cartesian()
        self.params = {"alpha": lr, "beta": t_noise}

    def _step(self):
        # use fast calculation of dx and dz
        self.protein.coords.requires_grad_()
        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        gradx = self.protein.coords.grad.clone()
        gradz = self.protein.get_gradz_from_gradx(self.protein.coords, gradx)
        # print(gradz)
        with torch.no_grad():
            noise = torch.randn_like(self.protein.coords_int, device=self.protein.coords_int.device)
            dz = self.int_scale * (-1.0 * self.params['alpha'] * gradz + self.params['beta'] * noise)
            # print(dz.max(dim=0), dz.min(dim=0))
            dx = self.protein.get_dx_from_dz(self.protein.coords, dz)
            self.protein.coords = self.protein.coords + dx
            self.protein.update_internal_from_cartesian()

        return energy


class DynamicsMixFast(DynamicsBase):
    def __init__(self, energy_fn, protein, lr=3e-2, t_noise=3e-2, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.protein.update_internal_from_cartesian()
        self.params = {"alpha": lr, "beta": t_noise}

    def _step(self):
        lr = self.params['alpha']
        t_noise = self.params['beta']

        self.protein.coords.requires_grad_()
        energy = self.protein.get_energy(self.energy_fn)
        energy.backward()
        gradx = self.protein.coords.grad.clone()
        with torch.no_grad():
            # apply the cartesian step to update the coords
            noise = torch.randn_like(self.protein.coords, device=self.protein.coords.device)
            dx_cart = self.cart_scale * (-1.0 * lr * gradx + t_noise * noise)
            coords_c = self.protein.coords + dx_cart
            # calculate gradz using the updated coords
            gradz = self.protein.get_gradz_from_gradx(coords_c, gradx)
            # print(gradz)
            # apply the internal step
            noise_int = torch.randn_like(self.protein.coords_int, device=self.protein.coords_int.device)
            dz = self.int_scale * (-1.0 * lr * gradz + t_noise * noise_int)
            # print(dz.max(dim=0), dz.min(dim=0))
            # calculate dx at the updated coords -- coords_c
            # dx = self.protein.get_dx_from_dz(self.protein.coords, dz)
            dx = self.protein.get_dx_from_dz(coords_c, dz)
            self.protein.coords = coords_c + dx
            self.protein.update_internal_from_cartesian()

        return energy


