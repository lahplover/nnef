import numpy as np
from tqdm import tqdm
import torch
from .move import MoveICOne, MoveSeq


class AnnealBase():
    def __init__(self, energy_fn, protein, T_max=0.1, T_min=0.01, L=1000):
        self.T_max = T_max
        self.T_min = T_min
        self.T = T_max
        self.L = L
        self.num_steps = 0

        self.protein = protein
        self.energy_fn = energy_fn

        self.x_best = None
        self.energy_best = protein.get_energy(energy_fn).item()

        self.move = None
        self.sample = []
        self.sample_energy = []

    def cool_down(self):
        raise NotImplementedError

    def move_step(self, x):
        raise NotImplementedError

    def run(self):
        # run annealing
        x_current, energy_current = self.x_best, self.energy_best
        device = x_current.device

        while self.T > self.T_min:
            climb_up = 0
            climb_down = 0
            for k in tqdm(range(self.L)):
                x_new = self.move_step(x_current)
                if torch.isnan(x_new).sum() > 0:
                    print('x_new is nan')
                    continue
                energy_new = self.protein.get_energy(self.energy_fn).item()
                # print(energy_new)

                # Metropolis
                energy_diff = energy_new - energy_current
                if energy_diff < 0:
                    x_current, energy_current = x_new, energy_new
                    climb_down += 1
                    self.sample.append(x_current.cpu())
                    self.sample_energy.append(energy_current)

                elif np.exp(-energy_diff / self.T) > np.random.rand():
                    x_current, energy_current = x_new, energy_new
                    climb_up += 1
                    self.sample.append(x_current.cpu())
                    self.sample_energy.append(energy_current)

                if energy_new < self.energy_best:
                    self.x_best, self.energy_best = x_new, energy_new

            self.num_steps += 1
            # if energy_best < energy_native:
            #     break
            print(f'climb up fraction: {climb_up * 1.0 / self.L}')
            print(f'climb down fraction: {climb_down * 1.0 / self.L}')
            print(f'T: {self.T}, energy best: {self.energy_best}, energy current: {energy_current}')
            print(f'total steps: {self.num_steps}')
            # print(self.x_best)
            rg2, _ = self.protein.get_rad_gyration(self.x_best)
            print(f'current best rg2 radius of gyration square: {rg2}')

            self.cool_down()


class AnnealCoords(AnnealBase):
    def __init__(self, energy_fn, protein, mode='CB', loop=None, ic_move_std=2, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.x_best = self.protein.coords
        self.move = MoveICOne(mode, loop, ic_move_std)

    def cool_down(self):
        self.T = self.T * 0.9
        # self.T = self.T_max / (np.log2(self.num_steps + 1))

    def move_step(self, x_current):
        x_new = self.move(x_current.clone())
        self.protein.update_coords(x_new)
        return x_new


class AnnealFrag(AnnealBase):
    def __init__(self, energy_fn, protein, frag=None, use_rg=True, T_rg2=50, **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.x_best = self.protein.coords
        self.frag_pos, self.frag_int = frag
        self.use_rg = use_rg
        self.T_rg2 = T_rg2
        self.rg2_best = None

    def cool_down(self):
        self.T = self.T * 0.9
        # self.T_rg2 = self.T_rg2 * 0.8
        # self.T = self.T_max / (np.log2(self.num_steps + 1))

    def move_step(self, x_current):
        x_current_int = self.protein.cartesian_to_internal(x_current)
        N, frag_len, _ = self.frag_int.size()
        i = torch.randint(0, N, (1,))[0]  # frag starting position should exclude the last few residues
        pos_i = self.frag_pos[i]
        if pos_i+frag_len-1 == x_current_int.size(0):
            x_current_int[pos_i:pos_i + frag_len] = self.frag_int[i][0:-1]
        else:
            x_current_int[pos_i:pos_i+frag_len] = self.frag_int[i]
        self.protein.update_coords_internal(x_current_int)
        self.protein.update_cartesian_from_internal()
        x_new = self.protein.coords
        return x_new

    def run(self):
        # run annealing
        x_current, energy_current = self.x_best, self.energy_best
        device = x_current.device
        rg2_level, _ = self.protein.get_rad_gyration(x_current)
        # print(f'initial radius of gyration square: {rg2_current}')

        while self.T > self.T_min:
            climb_up = 0
            climb_down = 0
            n_energy_calculation = 0
            for k in tqdm(range(self.L)):
                x_new = self.move_step(x_current)
                if torch.isnan(x_new).sum() > 0:
                    print('x_new is nan')
                    continue
                if self.use_rg:
                    # check collision and radius of gyration
                    # rg2_level may not be the rg2 of the current coordinates,
                    # because the coordinates may be rejected by energy calculation.
                    rg2_new, collision = self.protein.get_rad_gyration(x_new)
                    if collision:
                        continue
                    rg2_diff = (rg2_new - rg2_level).item()
                    self.T_rg2 = min(rg2_level.item() / 10.0, 60.0)
                    if rg2_diff < 0:
                        rg2_level = rg2_new
                    elif np.exp(-rg2_diff / self.T_rg2) > np.random.rand():
                        rg2_level = rg2_new
                    else:
                        continue

                energy_new = self.protein.get_energy(self.energy_fn).item()
                n_energy_calculation += 1

                # Metropolis
                energy_diff = energy_new - energy_current
                if energy_diff < 0:
                    x_current, energy_current = x_new, energy_new
                    climb_down += 1
                    self.sample.append(x_current.cpu())
                    self.sample_energy.append(energy_current)

                elif np.exp(-energy_diff / self.T) > np.random.rand():
                    x_current, energy_current = x_new, energy_new
                    climb_up += 1
                    self.sample.append(x_current.cpu())
                    self.sample_energy.append(energy_current)

                if energy_new < self.energy_best:
                    self.x_best, self.energy_best = x_new, energy_new

            self.num_steps += 1
            # if energy_best < energy_native:
            #     break
            if self.use_rg:
                self.rg2_best, _ = self.protein.get_rad_gyration(self.x_best)
                print(f'best radius of gyration square: {self.rg2_best}')
                print(f'rg2 level: {rg2_level}')
            print(f'climb up fraction: {climb_up * 1.0 / self.L}')
            print(f'climb down fraction: {climb_down * 1.0 / self.L}')
            print(f'energy calculation fraction: {n_energy_calculation * 1.0 / self.L}')
            print(f'T: {self.T}, energy best: {self.energy_best}, energy current: {energy_current}')
            print(f'total steps: {self.num_steps}')
            self.cool_down()


class AnnealSeq(AnnealBase):
    def __init__(self, energy_fn, protein, seq_move_type='mutate_one', **kwargs):
        super().__init__(energy_fn, protein, **kwargs)
        self.x_best = self.protein.profile
        self.move = MoveSeq(seq_move_type)

    def cool_down(self):
        # self.T = self.T * 0.9
        self.T = self.T_max / (np.log2(self.num_steps + 1))

    def move_step(self, x_current):
        x_new = self.move(x_current.clone())
        self.protein.update_profile(x_new)
        return x_new

