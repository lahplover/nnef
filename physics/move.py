import numpy as np
import h5py
import torch


def rotate_coords(coords, i):
    """
    giving cartesian coordinates, a selected central residue C, rotate the system to the internal ref.
    internal ref: C is at origin; A, B are the previous two residues. C-B along negative x. C-B-A in x-y plane.
    :param coords:
    :return:
    """
    coords = coords - coords[i]
    c1 = coords[i-1]
    c2 = coords[i-2]

    z = torch.cross(c1, c2, dim=-1)
    x = -c1
    y = torch.cross(z, x, dim=-1)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    R = torch.cat([x, y, z], dim=-1).view(3, 3)  # (3, 3)

    coords = torch.matmul(R[None, :, :], coords[:, :, None]).squeeze()

    return coords


class SampleICNext():
    def __init__(self, mode, ic_move_std=2):
        self.mode = mode
        data_c_next = h5py.File(f'data/training_30_{mode}_c-next_sample.h5', 'r')
        self.c_next_sample = torch.tensor(data_c_next['coords_internal'][()])
        self.ic_move_std = ic_move_std

    def __call__(self):
        # get a random x-y-z for coords of next residue,
        j = torch.randint(0, self.c_next_sample.size(0), (1,))[0]
        c_next = self.c_next_sample[j] + torch.rand((3,)) * 0.001  # add a small random noise
        # TODO: add a random permute to the angles?
        return c_next

    def random_coords_int(self, n):
        # get n random r-theta-phi for coords of next residue,
        j = torch.randint(0, self.c_next_sample.size(0), (n,))
        c_next = self.c_next_sample[j] + torch.rand((n, 3)) * 0.001  # add a small random noise
        r = torch.norm(c_next, dim=-1)
        theta = torch.acos(c_next[:, 0] / r)
        phi = torch.atan2(c_next[:, 2], c_next[:, 1])  # atan2 considers the quadrant
        coords_int = torch.stack((r, theta, phi), dim=1)
        return coords_int

    def extend_coords_int(self, n):
        # get n average r-theta-phi for coords of next residue,
        c_next = self.c_next_sample
        r = torch.norm(c_next, dim=-1).mean()
        theta = torch.acos(c_next[:, 0] / r).mean()
        phi = torch.atan2(c_next[:, 2], c_next[:, 1]).mean()  # atan2 considers the quadrant
        print(f'average internal c_next: r={r}, theta={theta}, phi={phi}')
        coords_int = torch.tensor([r, theta, phi]).repeat((n, 1))
        return coords_int

    def small_move_int(self, c_next):
        # do small perturbations to the r, theta, phi internal coordinates.
        r = torch.norm(c_next, dim=-1)
        theta = torch.acos(c_next[0] / r)
        phi = torch.atan2(c_next[2], c_next[1])  # atan2 considers the quadrant
        # a = torch.rand((3,)) * 2 - 1  # random number (-1, 1)
        a = torch.randn((3,))  # random number N(0, 1)
        r = r + 0.001 * a[0]
        theta = theta + np.pi / 180.0 * self.ic_move_std * a[1]  # standard deviation = 2 degree
        phi = phi + np.pi / 180.0 * self.ic_move_std * a[2]
        # get xyz by reversing the above r-theta-phi calculations;
        x = r * torch.cos(theta)
        y = r * torch.sin(theta) * torch.cos(phi)  # tan(phi) = z/y
        z = r * torch.sin(theta) * torch.sin(phi)
        c_next_new = torch.tensor([x, y, z], device=c_next.device)
        return c_next_new


class MoveICOne():
    def __init__(self, mode, loop, ic_move_std):
        # If loop is set, move only the flexible loop region. loop is the index of the residues.
        self.sample_ic = SampleICNext(mode, ic_move_std=ic_move_std)
        self.loop = loop
        self.sample_method = 'small_move'  # small_move or random_sample

    def _rotation_matrix(self, c_next_current, c_next_new, device):
        # Instead, apply rotations and shifts along r direction to the residues after c_next;
        # u is the rotation axis, theta is the rotation angle
        c_next_new_norm = torch.norm(c_next_new)
        c_next_current_norm = torch.norm(c_next_current)
        # u should be a unit vector
        u = torch.cross(c_next_current, c_next_new)
        u /= torch.norm(u)
        cos_theta = torch.dot(c_next_current, c_next_new) / (c_next_new_norm * c_next_current_norm)
        sin_theta = (1 - cos_theta ** 2) ** 0.5

        ux, uy, uz = u[0], u[1], u[2]
        u_cross_product_matrix = torch.tensor([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], device=device)
        u_outer_product = torch.ger(u, u)
        id_matrix = torch.eye(3, device=device)
        R = cos_theta * id_matrix + \
            sin_theta * u_cross_product_matrix + \
            (1 - cos_theta) * u_outer_product
        return R

    def __call__(self, coords):
        """
        get a random residue i, move residue i+1 and the residues after it.
        coords -- (N, 3)
        :return:
        """
        device = coords.device

        if self.loop is None:
            i = torch.randint(2, coords.size(0) - 1, (1,), device=device)[0]
        else:
            # sample a loop residue
            i = torch.randint(0, self.loop.size(0), (1,), device=device)[0]
            i = self.loop[i]

        coords_new = rotate_coords(coords, i)
        c_next_current = coords_new[i + 1]

        # get a new bond connecting residue i and i+1
        if self.sample_method == 'small_move':
            c_next_new = self.sample_ic.small_move_int(c_next_current)
        elif self.sample_method == 'random_sample':
            c_next_new = self.sample_ic().to(device)
        else:
            raise ValueError('sample_method should be small_move / random_sample')

        R = self._rotation_matrix(c_next_current, c_next_new, device)
        coords_new[i + 1:] = torch.matmul(R[None, :, :], coords_new[i + 1:, :, None]).squeeze()
        # shift along new r direction
        c_next_new_norm = torch.norm(c_next_new)
        c_next_current_norm = torch.norm(c_next_current)
        delta_xyz = c_next_new / c_next_new_norm * (c_next_new_norm - c_next_current_norm)
        coords_new += delta_xyz

        return coords_new


def init_coords(coords, mode):
    sample_ic = SampleICNext(mode)

    num = coords.size(0)
    coords[3:] = 0
    for i in range(2, num-1):
        coords[:i+1] = rotate_coords(coords[:i+1], i)
        c_next = sample_ic()
        coords[i+1] = c_next.to(coords.device)
    return coords


class MoveSeq():
    def __init__(self, seq_move_type):
        self.seq_move_type = seq_move_type

    def __call__(self, seq):
        if self.seq_move_type == 'mutate_one':
            # mutate one residue
            n = seq.size(0)
            i = np.random.randint(0, n)
            seq[i] = np.random.randint(0, 20)
        elif self.seq_move_type == 'swap_one':
            # swap two residue
            n = seq.size(0)
            i, j = np.random.randint(0, n, 2)
            m = seq[i].item()
            n = seq[j].item()
            seq[i] = n
            seq[j] = m
        else:
            raise ValueError('seq move type should be mutate_one / swap_one')

        return seq




