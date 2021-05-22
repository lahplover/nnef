import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from model import LocalEnergyCE
import pandas as pd


class EnergyFun(nn.Module):
    def __init__(self, model, args, residue_sum=False, return_loss_terms=False):
        super().__init__()
        self.energy_fn = LocalEnergyCE(model, args)

        # energy_fn returns the mean loss of batch. if residue_sum is True, use the sum of loss in the batch.
        self.residue_sum = residue_sum
        self.return_loss_terms = return_loss_terms
        self.debug = args.debug
        if self.debug:
            self.writer = SummaryWriter('runs/fold/')
            self.counter = 0

    def forward(self, seq, coords, start_id, res_counts):
        loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts = self.energy_fn.forward(seq, coords, start_id, res_counts)

        energy = loss_r + loss_angle + loss_profile + loss_start_id + loss_res_counts

        if self.debug:
            self.writer.add_scalar('profile_loss', loss_profile.item(), self.counter)
            self.writer.add_scalar('coords_radius_loss', loss_r.item(), self.counter)
            self.writer.add_scalar('coords_angle_loss', loss_angle.item(), self.counter)
            self.writer.add_scalar('start_id_loss', loss_start_id.item(), self.counter)
            self.writer.add_scalar('res_counts_loss', loss_res_counts.item(), self.counter)
            self.writer.add_scalar('total_loss', energy.item(), self.counter)
            self.counter += 1

        if self.residue_sum:
            energy = energy * coords.size(0)

        if self.return_loss_terms:
            return energy, loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts
        else:
            return energy


class ProteinBase:
    k = 10  # besides the 5 residues in the central segment, use another k nearest neighbors
    use_graph_net = False
    ref_scale = 400.0
    use_ref = False
    energy_model_type = None

    def __init__(self):
        df = pd.read_csv('data/aa_freq_alpha-beta-train.csv')
        self.energy_ref = torch.tensor(df['e_ref'].values, dtype=torch.float)


class Protein(ProteinBase):
    """
    A protein and its energies.
    This class compute the energy of a protein chain.
    group_num, coords, seq_feature should have the same first dimension.
    The protein chain should NOT have missing residues.
    In this case, group_num is from 1 to coords.shape[0].
    The nearest 4 residues in sequence for g are simply g-2, g-1, g+1, g+2.
    Internal coordinate system: C is at origin; A, B are the previous two residues.
    C-B along negative x. C-B-A in x-y plane.
    """

    def __init__(self, seq, coords, profile, protein_id=None):
        super().__init__()
        self.seq = seq  # (N,)
        self.coords = coords  # (N, 3)
        self.coords_int = None
        self.profile = profile  # (N, E) for evolution profile, (N,) for residues
        # reference energy of unfold state
        self.energy_ref = self.energy_ref.to(self.coords.device)
        self.energy_seq = torch.mean(self.energy_ref[profile]) * self.ref_scale
        self.protein_id = protein_id

    def update_coords(self, coords):
        self.coords = coords

    def update_profile(self, profile):
        self.profile = profile
        self.energy_seq = torch.mean(self.energy_ref[profile]) * self.ref_scale

    def update_coords_internal(self, coords_int):
        self.coords_int = coords_int

    def update_cartesian_from_internal(self):
        coords_ref = self.coords[:3].detach()
        self.coords = self.internal_to_cartesian(coords_ref, self.coords_int)

    def update_internal_from_cartesian(self):
        self.coords_int = self.cartesian_to_internal(self.coords)

    @staticmethod
    def _get_internal_unit_vectors(c1, c2):
        # c1 is minus X-axis, c1 c2 are X-Y plane.
        z = torch.cross(c1, c2, dim=-1)
        x = -c1
        y = torch.cross(z, x, dim=-1)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        return x, y, z

    def cartesian_to_internal(self, coords):
        # convert cartesian to internal,
        # coords -- (N, 3), internal (N-3, 3)
        c0 = coords[2:-1, :]  # c residue is the (0, 0, 0) of internal coordinate
        c1 = coords[1:-2, :] - c0  # c-1 residue
        c2 = coords[0:-3, :] - c0  # c-2 residue
        c_next = coords[3:, :] - c0  # c+1 residue
        x, y, z = self._get_internal_unit_vectors(c1, c2)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        c_next = torch.matmul(R, c_next[:, :, None]).squeeze(dim=-1)
        # theta is the angle to X-axis, phi is the angle in Y-Z plane
        r = torch.norm(c_next, dim=-1)
        theta = torch.acos(c_next[:, 0] / r)
        phi = torch.atan2(c_next[:, 2], c_next[:, 1])  # atan2 considers the quadrant
        coords_int = torch.stack((r, theta, phi), dim=1)
        return coords_int

    def cartesian_to_c_next(self, coords):
        # convert cartesian to c_next in internal coordinate system,
        # coords -- (N, 3), c_next (N-3, 3)
        c0 = coords[2:-1, :]  # c residue is the (0, 0, 0) of internal coordinate
        c1 = coords[1:-2, :] - c0  # c-1 residue
        c2 = coords[0:-3, :] - c0  # c-2 residue
        c_next = coords[3:, :] - c0  # c+1 residue
        x, y, z = self._get_internal_unit_vectors(c1, c2)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        c_next = torch.matmul(R, c_next[:, :, None]).squeeze(dim=-1)
        return c_next

    def internal_to_cartesian(self, coords_ref, coords_int):
        # convert internal to cartesian,
        # coords_ref: (3, 3), using coordinates of the first 3 residues as reference.
        N = coords_int.shape[0] + 3
        coords = torch.zeros((N, 3), dtype=coords_int.dtype, device=coords_int.device)
        coords[0:3] = coords_ref
        for i in range(2, N - 1):
            # calculate the unit vectors (x, y, z) of internal coordinate system i
            c1 = coords[i - 1] - coords[i]
            c2 = coords[i - 2] - coords[i]
            x, y, z = self._get_internal_unit_vectors(c1, c2)
            # rotate the internal coordinates of the next point to the reference coordinate system
            r, theta, phi = coords_int[i - 2]
            cos_angle, sin_angle = torch.cos(theta), torch.sin(theta)
            cos_torsion, sin_torsion = torch.cos(phi), torch.sin(phi)
            c_next = r * (cos_angle * x +
                          cos_torsion * sin_angle * y +
                          sin_torsion * sin_angle * z)
            coords[i + 1] = c_next + coords[i]
        return coords

    @staticmethod
    def get_jacobian_dxdz(coords):
        """
        dx/dr: Increasing or decreasing the bond length r extends or retracts all downstream coordinates
        along the bonds axis ur;
        dx/d_theta: moving a bond angle theta drives circular motion of all downstream coordinates
        around the normal vector n_theta;
        dx/d_phi: moving a torsion angle phi drives circular motion of all downstream coordinates around vector ux.
        """
        N = coords.size(0)
        ur = coords[3:] - coords[2:-1]  # (N-3, 3)
        ur = ur / torch.norm(ur, dim=-1, keepdim=True)

        ux = coords[2:-1] - coords[1:-2]  # (N-3, 3)
        ux = ux / torch.norm(ux, dim=-1, keepdim=True)

        n_theta = torch.cross(ux, ur, dim=-1)  # (N-3, 3)
        n_theta = n_theta / torch.norm(n_theta, dim=-1, keepdim=True)

        # each xi (dim0) is subtracted by all xj (dim1).
        xij = coords[None, 3:, :] - coords[2:-1, None, :]   # (N-3, N-3, 3)
        # each xi only move its downstream xj, so only keep the upper right triangle of the matrix.
        mask = torch.triu(torch.ones((N - 3, N - 3), device=coords.device))[:, :, None]

        dxdr = mask * ur[:, None, :].expand((-1, N - 3, -1))  # (N-3, N-3, 3)
        dxdtheta = mask * torch.cross(n_theta[:, None, :].expand((-1, N-3, -1)), xij, dim=-1)  # (N-3, N-3, 3)
        dxdphi = mask * torch.cross(ux[:, None, :].expand((-1, N-3, -1)), xij, dim=-1)  # (N-3, N-3, 3)

        return dxdr, dxdtheta, dxdphi

    @staticmethod
    def multiply_jacobian_dz(dz, dxdr, dxdtheta, dxdphi):
        """
        dx = Jacobian dxdz * dz,  (N-3, 3)
        dz (N-3, 3),
        dxdr, dxdtheta, dxdphi, (N-3, N-3, 3)
        """
        dr = dz[:, None, 0:1]  # (N-3, 1, 1)
        dtheta = dz[:, None, 1:2]  # (N-3, 1, 1)
        dphi = dz[:, None, 2:3]  # (N-3, 1, 1)

        # for each bead j, sum the moves caused by the dr / dtheta / dphi of all previous bead i.
        dx_r = torch.sum(dr * dxdr, dim=0)  # (N-3, 3)
        dx_theta = torch.sum(dtheta * dxdtheta, dim=0)  # (N-3, 3)
        dx_phi = torch.sum(dphi * dxdphi, dim=0)  # (N-3, 3)

        dx = dx_r + dx_theta + dx_phi
        return dx

    def get_dx_from_dz(self, coords, dz):
        """
        coords: (N, 3); dz: (N-3, 3)
        use the improved Euler method to calculate dx.
        x_p = x_t + dx_t/dz * dz
        dx = 0.5 * (dx_t/dz + dx_p/dz) * dz
        """
        dxdr, dxdtheta, dxdphi = self.get_jacobian_dxdz(coords)
        dx_1 = self.multiply_jacobian_dz(dz, dxdr, dxdtheta, dxdphi)
        coords_p = coords.clone()
        coords_p[3:] = coords_p[3:] + dx_1
        dxdr_p, dxdtheta_p, dxdphi_p = self.get_jacobian_dxdz(coords_p)
        dx_2 = self.multiply_jacobian_dz(dz, dxdr_p, dxdtheta_p, dxdphi_p)
        dx = 0.5 * (dx_1 + dx_2)  # (N-3, 3)
        # add 3 rows of zeros, (N-3, 3) -> (N, 3)
        head_zeros = torch.zeros((3, 3), device=dx.device)
        dx = torch.cat([head_zeros, dx], dim=0)
        return dx

    def get_gradz_from_gradx(self, coords, gradx):
        """
        coords: (N, 3); gradx: (N, 3)
        gradz: (N-3, 3)
        Each r / theta / phi of residue i will change the x, y, z of all following residues,
        so in back-propagation, grad_i should sum gradients from the x-y-z of grad_j (j>i).
        """
        dxdr, dxdtheta, dxdphi = self.get_jacobian_dxdz(coords)
        gradx = gradx[3:]
        grad_r = torch.sum(gradx[None, :, :] * dxdr, dim=(1, 2))  # (1, N-3, 3) * (N-3, N-3, 3) -> (N-3)
        grad_theta = torch.sum(gradx[None, :, :] * dxdtheta, dim=(1, 2))  # (1, N-3, 3) * (N-3, N-3, 3) -> (N-3)
        grad_phi = torch.sum(gradx[None, :, :] * dxdphi, dim=(1, 2))  # (1, N-3, 3) * (N-3, N-3, 3) -> (N-3)
        gradz = torch.stack([grad_r, grad_theta, grad_phi], dim=1)
        return gradz

    @staticmethod
    def get_distmap(coords):
        # coords: (N, 3); distmap: (N, N)
        x_map = coords[:, None, :] - coords[None, :, :]
        distmap = torch.norm(x_map, dim=-1)
        # distmap = torch.sqrt(torch.sum((coords[:, None, :] - coords[None, :, :])**2, dim=-1))
        return distmap

    @staticmethod
    def get_rad_gyration(coords):
        x_map = coords[:, None, :] - coords[None, :, :]
        distmap = torch.norm(x_map, dim=-1)
        rg2 = torch.mean(distmap**2)
        diag_idx = torch.arange(coords.shape[0])
        distmap[diag_idx, diag_idx] = 1.0
        idx = (distmap < 1.0)
        n = torch.sum(idx).item()
        collision = False
        if n > 0:
            collision = True
        return rg2, collision

    @staticmethod
    def check_coords(coords):
        # check if any two beads are too close
        x_map = coords[:, None, :] - coords[None, :, :]
        distmap = torch.norm(x_map, dim=-1)
        diag_idx = torch.arange(coords.shape[0])
        distmap[diag_idx, diag_idx] = 1.0
        idx = (distmap < 1e-1)
        n = torch.sum(idx).item()
        if n > 0:
            print(f'{n} beads pairs are too close.')
            print(distmap[idx])
        # check if any 3 adjacent beads are in a line
        c1 = coords[1:-1, :] - coords[:-2, :]
        c2 = coords[2:, :] - coords[1:-1, :]
        z = torch.cross(c1, c2, dim=-1)
        # check if c1 and c2 are along the same line
        z_norm = torch.norm(z, dim=1)
        n2 = z_norm[z_norm < 1e-6].shape[0]
        if n2 > 0:
            print(f'{n} beads triples are co-line.')

    @staticmethod
    def _local_rotation_matrix(c1, c2):
        # c1, c2 size (N, 3)
        z = torch.cross(c1, c2, dim=-1)
        # check if c1 and c2 are along the same line
        z_norm = torch.norm(z, dim=1)
        idx = (z_norm < 1e-6)
        if c2[idx].shape[0] > 0:
            c2[idx] = c2[idx] + 1e-4
        x = c1
        y = torch.cross(z, x, dim=-1)
        x = x / torch.norm(x, dim=1, keepdim=True)
        y = y / torch.norm(y, dim=1, keepdim=True)
        z = z / torch.norm(z, dim=1, keepdim=True)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        return R

    def get_local_struct(self):
        group_coords = self.coords
        group_profile = self.profile
        # done in parallel for all residues. group_coords: (N, 3), group_profile: (N, E)
        device = group_coords.device
        num = group_coords.size(0)
        group_num = torch.arange(num, device=device)
        gc = group_num[2:-2]   # (num-4)
        g1 = group_num[1:-3]
        g2 = group_num[3:-1]
        g3 = group_num[:-4]
        g4 = group_num[4:]

        # index of the central segment
        g_seg = torch.stack((gc, g1, g2, g3, g4)).transpose(0, 1)  # (N-4, 5)
        # get group numbers EXCLUDE the central segment
        g_others = torch.arange(num, device=device).repeat(1, num-4)
        g_others = F.pad(g_others, (0, num-4)).view(num-4, num+1)[:, 5:].flatten()
        g_others = g_others[:-(num-4)].view(num-4, num-5)   # (N-4, N-5)

        coords_others = group_coords[g_others] - group_coords[gc][:, None, :]  # (N-4, N-5, 3)
        distance = torch.norm(coords_others, dim=-1)  # (N-4, N-5)

        # get number of residues within 8A, 10A, 12A
        count_8a = (distance < 8).sum(dim=-1)  # (N-4,)
        count_10a = (distance < 10).sum(dim=-1)
        count_12a = (distance < 12).sum(dim=-1)
        res_counts = torch.cat([count_8a[:, None], count_10a[:, None], count_12a[:, None]], dim=-1)  # (N-4, 3)

        if self.k > 0:
            # calculate the index of the nearest k neighbors
            topk_dist, topk_arg = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)  # topk_arg size (N-4, k)
            g_topk = torch.gather(g_others, dim=1, index=topk_arg)  # (N-4, k)

            # re-order the topk index, g_topk and topk_dist is the group num and dist of the topk residues.
            g_topk_sorted, g_topk_sort_arg = torch.sort(g_topk, dim=-1)  # sort by group number
            # reorder dist_topk, so dist_topk_sorted is the distance sorted by group number
            dist_topk_sorted = torch.gather(topk_dist, dim=1, index=g_topk_sort_arg)
            # make a mask for each segment g_mask (N-4, k, k)
            g_topk_diff = g_topk_sorted[:, None, :] - g_topk_sorted[:, :, None]
            g_mask_ref = torch.arange(self.k, device=device)
            g_mask_ref = g_mask_ref - g_mask_ref[:, None]
            g_mask = torch.zeros(g_topk_sorted.size(0), self.k, self.k, device=device)
            g_mask[g_topk_diff == g_mask_ref] = 1
            # calculate the mean distance of each segment
            dist_masked = dist_topk_sorted[:, None, :] * g_mask
            dist_topk_mean = torch.sum(dist_masked, dim=-1) / torch.sum(g_mask, dim=-1)
            # reorder the groups by the segment distance and group number.
            dist_gnum_topk = (dist_topk_mean * 10 ** 6).long() * num + g_topk_sorted
            dist_gnum_topk_arg = torch.argsort(dist_gnum_topk, dim=-1)
            gnum_dist_topk_sorted = torch.gather(g_topk_sorted, dim=1, index=dist_gnum_topk_arg)

            # concat the index of the central segment and the k neighbors
            # g_local = torch.cat((g_seg, g_topk), dim=1)  # (N-4, 5+k)
            g_local = torch.cat((g_seg, gnum_dist_topk_sorted), dim=1)  # (N-4, 5+k)
        else:
            g_local = g_seg
        # extract the profile and coordinates using the index
        profile_local = group_profile[g_local]  # (N-4, 5+k, E)
        coords_local = group_coords[g_local] - group_coords[gc][:, None, :]  # (N-4, 5+k, 3)

        # rotate the local coordinates
        rotate_mat = self._local_rotation_matrix(coords_local[:, 1, :], coords_local[:, 2, :])
        # (N-4, 1, 3, 3) * (N-4, 5+k, 3, 1) -> (N-4, 5+k, 3, 1)
        coords_local = torch.matmul(rotate_mat[:, None, :, :], coords_local[:, :, :, None]).squeeze()

        start_id = torch.zeros_like(g_local)
        start_id[:, 1:5] = 1
        if self.k > 1:
            start_id[:, 6:][g_local[:, 6:] - g_local[:, 5:-1] == 1] = 1

        res_counts = res_counts.to(torch.float)

        return profile_local, coords_local, start_id, res_counts

    @staticmethod
    def _local_cartesian_to_radian(coords_local):
        # r = torch.norm(coords_local, dim=-1)  # (N-4, 5+k)
        r = torch.norm(coords_local[:, 1:], dim=-1)  # (N-4, 5+k-1)  # exclude the origin
        zr = coords_local[:, 1:, 2] / r
        zr = torch.clamp(zr, min=-1.0 + 1e-6, max=1.0 - 1e-6)  # otherwise, acos(1.00001) results NaN
        theta = torch.acos(zr)  # exclude the origin
        # theta = torch.acos(coords_local[:, 1:, 2] / r[:, 1:])  # exclude the origin
        phi = torch.atan2(coords_local[:, 1:, 1], coords_local[:, 1:, 0])  # atan2 considers the quadrant,
        r = F.pad(r, (1, 0), value=0)
        theta = F.pad(theta, (1, 0), value=0)
        phi = F.pad(phi, (1, 0), value=0)
        coords = torch.stack((r, theta, phi), dim=2)  # (N-4, 5+k, 3)
        return coords

    def get_energy(self, energy_fun):
        profile_local, coords_local, start_id, res_counts = self.get_local_struct()
        coords_local = self._local_cartesian_to_radian(coords_local)
        energy = energy_fun.forward(profile_local, coords_local, start_id, res_counts)
        if self.use_ref:
            energy = energy - self.energy_seq
        return energy

    def get_residue_energy(self, energy_fun):
        # seq & profile have add up to seq_feature
        profile_local, coords_local, start_id, res_counts = self.get_local_struct()
        coords_local = self._local_cartesian_to_radian(coords_local)
        num = coords_local.size(0)
        residue_energy = np.zeros(num)
        for i in range(num):
            residue_energy[i] = energy_fun.forward(profile_local[i:i+1], coords_local[i:i+1],
                                                   start_id[i:i+1], res_counts[i:i+1])
        return residue_energy

    def get_local_struct_phy(self):
        # local structures for graph net input
        group_coords = self.coords
        group_profile = self.profile
        # done in parallel for all residues. group_coords: (N, 3), group_profile: (N, E)
        device = group_coords.device
        num = group_coords.size(0)
        group_num = torch.arange(num, device=device)
        gc = group_num[2:-2]   # (num-4)
        g1 = group_num[1:-3]
        g2 = group_num[3:-1]
        g3 = group_num[:-4]
        g4 = group_num[4:]

        # index of the central segment
        g_seg = torch.stack((gc, g1, g2, g3, g4)).transpose(0, 1)  # (N-4, 5)
        # get group numbers EXCLUDE the central segment
        g_others = torch.arange(num, device=device).repeat(1, num-4)
        g_others = F.pad(g_others, (0, num-4)).view(num-4, num+1)[:, 5:].flatten()
        g_others = g_others[:-(num-4)].view(num-4, num-5)   # (N-4, N-5)

        coords_others = group_coords[g_others] - group_coords[gc][:, None, :]  # (N-4, N-5, 3)
        distance = torch.norm(coords_others, dim=-1)  # (N-4, N-5)

        # calculate the index of the nearest k neighbors
        topk_dist, topk_arg = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)  # topk_arg size (N-4, k)
        g_topk = torch.gather(g_others, dim=1, index=topk_arg)  # (N-4, k)

        # re-order the topk index, g_topk is the group num of the topk residues.
        g_topk_sorted, _ = torch.sort(g_topk, dim=-1)  # sort by group number

        # concat the index of the central segment and the k neighbors
        g_local = torch.cat((g_seg, g_topk_sorted), dim=1)  # (N-4, 5+k)

        # extract the profile and coordinates using the index
        profile_local = group_profile[g_local]  # (N-4, 5+k, E)
        coords_local = group_coords[g_local] - group_coords[gc][:, None, :]  # (N-4, 5+k, 3)

        # rotate the local coordinates
        rotate_mat = self._local_rotation_matrix(coords_local[:, 1, :], coords_local[:, 2, :])
        # (N-4, 1, 3, 3) * (N-4, 5+k, 3, 1) -> (N-4, 5+k, 3, 1)
        coords_local = torch.matmul(rotate_mat[:, None, :, :], coords_local[:, :, :, None]).squeeze()

        edge_type = g_local[:, None, :] - g_local[:, :, None]
        idx = (edge_type == 1) | (edge_type == -1) | (edge_type == 0)
        edge_type[~idx] = 2
        idx = (edge_type == 1) | (edge_type == -1)
        edge_type[idx] = 1
        return profile_local, coords_local, edge_type


class ProteinComplex(ProteinBase):
    """
    A protein complex and its energies.
    This class compute the energy of a few protein chains.
    group_num, coords, seq_feature should have the same first dimension.
    Each protein chain should NOT have missing residues.
    For the first chain group_num is from 1 to coords.shape[0].
    The the next chain group_num starts with 10 + group_num of the last residue in previous chain.
    The nearest 4 residues in sequence for g are simply g-2, g-1, g+1, g+2.
    Internal representation is not required for protein complex.
    """
    def __init__(self, chain, seq, coords, profile, protein_id=None):
        super().__init__()
        self.chain = chain  # (N,)  [0, 0, 0, 0, ... 1, 1, 1...]
        self.num_chain = torch.unique(chain).size(0)  # number of chain, eg. 2
        self.seq = seq  # (N,)
        self.coords = coords  # (N, 3)
        self.profile = profile  # (N, E)

        self.protein_id = protein_id
        # self.k = 10

    def update_coords(self, coords):
        self.coords = coords

    def update_profile(self, profile):
        self.profile = profile

    @staticmethod
    def _local_rotation_matrix(c1, c2):
        # c1, c2 size (N, 3)
        z = torch.cross(c1, c2, dim=-1)
        x = c1
        y = torch.cross(z, x, dim=-1)
        x = x / torch.norm(x, dim=1, keepdim=True)
        y = y / torch.norm(y, dim=1, keepdim=True)
        z = z / torch.norm(z, dim=1, keepdim=True)
        R = torch.cat([x, y, z], dim=-1).view(-1, 3, 3)  # (N, 3, 3)
        return R

    def setup_group_num(self, device):
        # num_chain = self.num_chain
        chain = self.chain

        group_num = torch.arange(chain.shape[0], device=device)
        # there is a bug.  The end of one chain and the start of next chain are 'connected' in group_num.
        # It will put them into the same segment if they are all neighbors of one residue.
        # for i in range(num_chain):
        #     idx = (chain == i)
        #     group_num[idx] += i * 10  # add a gap of 10 between chains

        gc_list = []
        g_seg_list = []
        g_others_list = []

        # for i in range(num_chain):
        for i in torch.unique(chain):
            idx = (chain == i)
            group_num_i = group_num[idx]
            # if the chain is shorter than 6 residues, ignore it.
            if len(group_num_i) < 6:
                continue
            gc = group_num_i[2:-2]  # (N_i-4)
            g1 = group_num_i[1:-3]
            g2 = group_num_i[3:-1]
            g3 = group_num_i[:-4]
            g4 = group_num_i[4:]

            gc_list.append(gc)
            # index of the central segment
            g_seg = torch.stack((gc, g1, g2, g3, g4)).transpose(0, 1)  # (N_i-4, 5)
            g_seg_list.append(g_seg)

            # get group numbers EXCLUDE the central segment
            num = group_num[idx].shape[0]
            g_others = torch.arange(num, device=device).repeat(1, num - 4)
            g_others = F.pad(g_others, (0, num - 4)).view(num - 4, num + 1)[:, 5:].flatten()
            g_others = g_others[:-(num - 4)].view(num - 4, num - 5)  # (N_i-4, N_i-5)
            g_others = g_others + group_num_i[0]

            g_other_chain = group_num[~idx].repeat(num - 4, 1)
            g_others = torch.cat((g_others, g_other_chain), dim=1)  # (N_i-4, N-5)
            g_others_list.append(g_others)

        gc = torch.cat(gc_list)
        g_seg = torch.cat(g_seg_list)
        g_others = torch.cat(g_others_list)

        return gc, g_seg, g_others

    def get_local_struct(self):
        group_coords = self.coords
        group_profile = self.profile
        # done in parallel for all residues. group_coords: (N, 3), group_profile: (N, E)
        device = group_coords.device
        num = self.coords.size(0)

        gc, g_seg, g_others = self.setup_group_num(device)

        coords_others = group_coords[g_others] - group_coords[gc][:, None, :]  # (N-4, N-5, 3)
        distance = torch.norm(coords_others, dim=-1)  # (N-4, N-5)

        # get number of residues within 8A, 10A, 12A
        count_8a = (distance < 8).sum(dim=-1)  # (N-4,)
        count_10a = (distance < 10).sum(dim=-1)
        count_12a = (distance < 12).sum(dim=-1)
        res_counts = torch.cat([count_8a[:, None], count_10a[:, None], count_12a[:, None]], dim=-1)  # (N-4, 3)

        # calculate the index of the nearest k neighbors
        topk_dist, topk_arg = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)  # topk_arg size (N-4, k)
        g_topk = torch.gather(g_others, dim=1, index=topk_arg)  # (N-4, k)

        # re-order the topk index, g_topk and topk_dist is the group num and dist of the topk residues.
        g_topk_sorted, g_topk_sort_arg = torch.sort(g_topk, dim=-1)  # sort by group number
        dist_topk_sorted = torch.gather(topk_dist, dim=1, index=g_topk_sort_arg)  # reorder dist_topk

        g_topk_diff = g_topk_sorted[:, None, :] - g_topk_sorted[:, :, None]
        g_mask_ref = torch.arange(self.k, device=device)
        g_mask_ref = g_mask_ref - g_mask_ref[:, None]
        g_mask = torch.zeros(g_topk_sorted.size(0), self.k, self.k, device=device)
        g_mask[g_topk_diff == g_mask_ref] = 1

        dist_masked = dist_topk_sorted[:, None, :] * g_mask
        dist_topk_mean = torch.sum(dist_masked, dim=-1) / torch.sum(g_mask, dim=-1)

        dist_gnum_topk = (dist_topk_mean * 10 ** 6).long() * num + g_topk_sorted
        dist_gnum_topk_arg = torch.argsort(dist_gnum_topk, dim=-1)
        gnum_dist_topk_sorted = torch.gather(g_topk_sorted, dim=1, index=dist_gnum_topk_arg)

        # concat the index of the central segment and the k neighbors
        # g_local = torch.cat((g_seg, g_topk), dim=1)  # (N-4, 5+k)
        g_local = torch.cat((g_seg, gnum_dist_topk_sorted), dim=1)  # (N-4, 5+k)

        # extract the profile and coordinates using the index
        profile_local = group_profile[g_local]  # (N-4, 5+k, E)
        coords_local = group_coords[g_local] - group_coords[gc][:, None, :]  # (N-4, 5+k, 3)

        # rotate the local coordinates
        rotate_mat = self._local_rotation_matrix(coords_local[:, 1, :], coords_local[:, 2, :])
        # (N-4, 1, 3, 3) * (N-4, 5+k, 3, 1) -> (N-4, 5+k, 3, 1)
        coords_local = torch.matmul(rotate_mat[:, None, :, :], coords_local[:, :, :, None]).squeeze()

        start_id = torch.zeros_like(g_local)
        start_id[:, 1:5] = 1
        start_id[:, 6:][g_local[:, 6:] - g_local[:, 5:-1] == 1] = 1

        res_counts = res_counts.to(torch.float)

        return profile_local, coords_local, start_id, res_counts

    @staticmethod
    def _local_cartesian_to_radian(coords_local):
        r = torch.norm(coords_local, dim=-1)  # (N-4, 5+k)
        theta = torch.acos(coords_local[:, 1:, 2] / r[:, 1:])  # exclude the origin
        phi = torch.atan2(coords_local[:, 1:, 1], coords_local[:, 1:, 0])  # atan2 considers the quadrant,
        theta = F.pad(theta, (1, 0), value=0)
        phi = F.pad(phi, (1, 0), value=0)
        coords = torch.stack((r, theta, phi), dim=2)  # (N-4, 5+k, 3)
        return coords

    def get_energy(self, energy_fun):
        # seq & profile have add up to seq_feature
        profile_local, coords_local, start_id, res_counts = self.get_local_struct()
        coords_local = self._local_cartesian_to_radian(coords_local)
        energy = energy_fun.forward(profile_local, coords_local, start_id, res_counts)
        return energy

    def get_residue_energy(self, energy_fun):
        # seq & profile have add up to seq_feature
        profile_local, coords_local, start_id, res_counts = self.get_local_struct()
        coords_local = self._local_cartesian_to_radian(coords_local)
        num = coords_local.size(0)
        residue_energy = np.zeros(num)
        for i in range(num):
            residue_energy[i] = energy_fun.forward(profile_local[i:i+1], coords_local[i:i+1],
                                                   start_id[i:i+1], res_counts[i:i+1])
        return residue_energy




