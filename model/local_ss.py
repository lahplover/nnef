import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import pandas as pd


"""
LocalEnergy is the energy of a protein. LocalTransformer is the transformer block. 
In local_ss.py, we separate structure and sequence. 
First calculate P(structure), then calculate P(sequence|structure). 
"""


class LocalTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.embedding = SeqEmbedding(args)
        self.seq_type = args.seq_type
        self.residue_type_num = args.residue_type_num
        if self.seq_type == 'residue':
            self.aa_embedding = nn.Embedding(self.residue_type_num, embedding_dim=args.embed_size)
        else:
            self.linear_seq = nn.Linear(self.residue_type_num, args.embed_size)

        self.seq_start_embed = nn.Embedding(1, embedding_dim=args.embed_size)
        self.pos_x_embed = nn.Embedding(args.seq_len+1, embedding_dim=args.embed_size // 4)
        self.pos_s_embed = nn.Embedding(args.seq_len+1, embedding_dim=args.embed_size // 4)

        self.start_id_embed = nn.Embedding(2, embedding_dim=args.embed_size // 4)
        # self.linear = nn.Linear(args.embed_size + 15, args.dim)
        # self.linear = nn.Linear(args.embed_size + 3, args.dim)
        self.linear_x = nn.Linear(2 * (args.embed_size // 4) + 3, args.dim)
        self.linear_s = nn.Linear(args.embed_size + args.embed_size // 4, args.dim)

        hidden = args.dim
        out_hidden = 2 * args.dim
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=args.attn_heads,
                dim_feedforward=hidden*4,
                dropout=args.dropout)

        decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden, nhead=args.attn_heads,
                dim_feedforward=hidden*4,
                dropout=args.dropout)

        mask = torch.tril(torch.ones((args.seq_len+1, args.seq_len+1), device=args.device))
        mask = mask.masked_fill((mask == 0), float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('mask', mask)

        self._encoder1 = nn.TransformerEncoder(encoder_layer, args.n_layers // 2)

        self._encoder2 = nn.TransformerEncoder(encoder_layer, args.n_layers // 2)

        self._decoder = nn.TransformerDecoder(decoder_layer, args.n_layers)

        self.connect_dim = 2
        self.out_r_dim = args.mixture_r * 3
        self.out_angle_dim = args.mixture_angle * 6
        if self.seq_type == 'residue':
            self.out_profile_dim = self.residue_type_num
        else:
            self.out_profile_dim = args.mixture_seq * 3 * self.residue_type_num
        # self.output_dim = self.num_mixture * 69  # 3 for r, 6 for theta-phi, 60 for profiles
        self.out_x_dim = self.out_r_dim + self.out_angle_dim + self.connect_dim
        # self.out_env_dim = args.mixture_res_counts * 9  # for res counts within 3 spheres
        self.out_s_dim = self.out_profile_dim

        self.linear_out_x = nn.Sequential(
            nn.Linear(hidden, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, self.out_x_dim)
        )

        self.linear_out_s = nn.Sequential(
            nn.Linear(hidden, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, self.out_s_dim)
        )

        # self.linear_res_counts = nn.Sequential(
        #     nn.Linear(hidden, out_hidden),
        #     nn.ReLU(),
        #     nn.Linear(out_hidden, out_hidden),
        #     nn.ReLU(),
        #     nn.Linear(out_hidden, self.out_env_dim)
        # )

    def forward(self, seq, coords, start_id):
        # s_feature = self.embedding(seq, start_id)
        # x_feature = self._coords_feature(coords)
        # sx = torch.cat((s_feature, x_feature), -1)
        if self.seq_type == 'residue':
            seq_feature = self.aa_embedding(seq)
        else:
            seq_feature = self.linear_seq(seq)
        seq_start_feature = self.seq_start_embed.weight.expand((seq_feature.size(0), -1, -1))
        # print(seq_start_feature.shape, seq_feature.shape)
        seq_feature = torch.cat((seq_start_feature, seq_feature), dim=1)  # (N, L, E) -> (N, L+1, E)
        # print(seq_feature.shape, seq_feature[:10, 0])

        start_id_feature = self.start_id_embed(start_id)
        # pos_feature = self.pos_embed.weight[None, :, :]
        # print(seq_feature.shape, start_id_feature.shape, pos_feature.shape)
        # s_feature = seq_feature + start_id_feature + pos_feature
        pos_x_feature = self.pos_x_embed.weight.expand((coords.size(0), -1, -1))
        pos_s_feature = self.pos_s_embed.weight.expand((seq_feature.size(0), -1, -1))

        # sx = torch.cat((s_feature, coords), -1)
        s_feature = torch.cat((seq_feature, pos_s_feature), -1)
        x_feature = torch.cat((coords, start_id_feature, pos_x_feature), -1)
        # sx = torch.cat((seq_feature, start_id_feature, pos_feature, coords), -1)
        s_feature = self.linear_s(s_feature)
        x_feature = self.linear_x(x_feature)

        s_feature = s_feature.transpose(0, 1)  # (N, L, E) -> (L, N, E)
        x_feature = x_feature.transpose(0, 1)  # (N, L, E) -> (L, N, E)

        code_x = self._encoder1(x_feature, mask=self.mask)
        # TODO: add sparsity loss of code_x? and maybe also for features of each layers?

        out_x = self._encoder2(code_x, mask=self.mask)
        out_x = out_x.transpose(0, 1)  # (L, N, E) -> (N, L, E)

        out_s = self._decoder(s_feature, code_x, tgt_mask=self.mask)
        out_s = out_s.transpose(0, 1)  # (L, N, E) -> (N, L, E)

        out_x = self.linear_out_x(out_x)
        out_s = self.linear_out_s(out_s)

        # z_r = out[:, :, 0:self.out_r_dim]
        # z_angles = out[:, :, self.out_r_dim:self.out_r_dim+self.out_angle_dim_dim]
        # z_profiles = out[:, :, -self.out_profile_dim:]
        #
        # return z_r, z_angles, z_profiles

        return out_x, out_s


class LocalEnergy(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.m_r = args.mixture_r
        self.m_angle = args.mixture_angle
        self.m_seq = args.mixture_seq
        self.m_res = args.mixture_res_counts
        self.residue_type_num = args.residue_type_num

        self.start_id_loss = nn.CrossEntropyLoss()

        self.profile_prob = args.profile_prob
        self.profile_loss_lamda = args.profile_loss_lamda
        self.coords_angle_loss_lamda = args.coords_angle_loss_lamda

    def forward(self, seq, coords, start_id):
        # seq = seq.to(self.device)  # (N, L)
        # coords = coords.to(self.device)  # (N, L, 3)
        # start_id = start_id.to(self.device)  # (N, L)
        # target_res_counts = res_counts.to(self.device)  # (N, 3)

        seq_len = seq.shape[-1]

        input_seq = seq[:, :-1]
        input_coords = coords  # (N, L-1, 3)
        input_start_id = start_id

        target_seq = seq
        target_coords = coords[:, 1:, :]

        out_x, out_s = self.model(input_seq, input_coords, input_start_id)

        if seq_len > 6:
            target_start_id = start_id[:, 6:]  # ignore the first 6 positions
            out_start_id = out_x[:, 5:-1, -2:].transpose(1, 2)  # ignore the first 5 positions
            # print('out_start_id', out_start_id.shape)
            loss_start_id = self.start_id_loss(out_start_id, target_start_id)
        else:
            loss_start_id = 0

        loss_r, loss_angle, loss_profile = self.get_mixture_loss(out_x[:, :-1, :-2], out_s, target_coords, target_seq)

        loss_angle *= self.coords_angle_loss_lamda
        loss_profile *= self.profile_loss_lamda

        return loss_r, loss_angle, loss_profile, loss_start_id

    def get_mixture_coef(self, out_x, out_s):
        """
        m is the number of mixture.
        Returns the Mixture Density Network distribution params.
        This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
        """
        m_r, m_angle, m_seq, m_res = self.m_r, self.m_angle, self.m_seq, self.m_res

        N, L, _ = out_x.size()
        idx1, idx2 = 3*m_r, 3*m_r+6*m_angle
        z_r = out_x[:, :, 0:idx1].reshape((N, L, m_r, 3))
        z_angle = out_x[:, :, idx1:idx2].reshape((N, L, m_angle, 6))

        z_profile = out_s.reshape((N, L+1, self.residue_type_num, m_seq, 3))

        # softmax all the mixture weights:
        r_pi = F.softmax(z_r[:, :, :, 0], dim=-1)
        angle_pi = F.softmax(z_angle[:, :, :, 0], dim=-1)
        profile_pi = F.softmax(z_profile[:, :, :, :, 0], dim=-1)

        r_mu = z_r[:, :, :, 1]
        angle_mu1 = z_angle[:, :, :, 1]
        angle_mu2 = z_angle[:, :, :, 2]
        profile_mu = z_profile[:, :, :, :, 1]
        # profile_mu = F.softmax(z_profile[:, :, :, :, 1], dim=2)

        # exponentiate the sigmas and also make corr between -1 and 1.
        r_sigma = torch.exp(z_r[:, :, :, 2])
        angle_sigma1 = torch.exp(z_angle[:, :, :, 3])
        angle_sigma2 = torch.exp(z_angle[:, :, :, 4])
        # profile_sigma = torch.exp(z_profile[:, :, :, :, 2])
        profile_sigma = torch.exp(z_profile[:, :, :, :, 2]) + 0.002

        angle_corr = torch.tanh(z_angle[:, :, :, 5])

        m_coef = [r_pi, r_mu, r_sigma,
                  angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr,
                  profile_pi, profile_mu, profile_sigma]

        return m_coef

    def get_mixture_loss(self, out_x, out_s, target_coords, target_seq):
        r, theta, phi = target_coords[:, :, 0:1], target_coords[:, :, 1:2], target_coords[:, :, 2:3]  # (N, L, 1)
        profile = target_seq[:, :, :, None]  # (N, L, 20, 1)

        m_coef = self.get_mixture_coef(out_x, out_s)

        r_pi, r_mu, r_sigma, \
        angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr, \
        profile_pi, profile_mu, profile_sigma = m_coef

        def normal_1d(x, mu, sigma):
            norm = (2*np.pi)**0.5 * sigma
            exp = torch.exp(-0.5 * (x - mu)**2 / sigma**2)
            return exp / norm

        def normal_2d(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            z1 = (x1 - mu1)**2 / sigma1**2
            z2 = (x2 - mu2)**2 / sigma2**2
            z12 = (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            z = z1 + z2 - 2 * rho * z12
            exp = torch.exp(-z/(2*(1-rho**2)))
            norm = 2 * np.pi * sigma1 * sigma2 * (1-rho**2)**0.5
            return exp / norm

        def log_weighted_sum(x, weight):
            x = torch.sum(x * weight, dim=-1)
            # return -torch.log(x + 1e-6)  # x may be much smaller than 1e-6
            return -torch.log(x + 1e-9)

        loss_r = normal_1d(r, r_mu, r_sigma)
        loss_r = log_weighted_sum(loss_r, r_pi)
        loss_r = torch.mean(loss_r)

        loss_profile = normal_1d(profile, profile_mu, profile_sigma)
        loss_profile = log_weighted_sum(loss_profile, profile_pi)  # (N, L, 20)
        if self.profile_prob:
            # instead of simply summing, sum the 20 AA types using the target profile as weights.
            # Thus high-prob residues will have larger contributions to loss
            loss_profile = torch.sum(loss_profile * profile.squeeze(), dim=-1)  # (N, L)
            loss_profile = torch.mean(loss_profile)
        else:
            N, L, _ = loss_profile.size()
            loss_profile = loss_profile[profile[:, :, :, 0] > 0.005]
            # loss_profile = torch.sum(loss_profile, dim=-1)  # sum the 20 AA types
            loss_profile = torch.sum(loss_profile) / float(N*L)

        # bring phi-mu into (-pi, pi)
        phi = phi.repeat([1, 1, self.m_angle])  # (N, L, mix_phi)
        assert (phi.shape == angle_mu2.shape)
        idx1 = (angle_mu2 > -np.pi) & (angle_mu2 < np.pi) & (phi - angle_mu2 > np.pi)
        idx2 = (angle_mu2 > -np.pi) & (angle_mu2 < np.pi) & (phi - angle_mu2 < -np.pi)
        # phi[idx1] -= 2 * np.pi
        # phi[idx2] += 2 * np.pi
        phi2 = torch.ones(phi.shape, device=phi.device)
        phi2[:] = phi[:]
        phi2[idx1] = phi[idx1] - 2 * np.pi
        phi2[idx2] = phi[idx2] + 2 * np.pi

        # ignore angles of the first two residues in the central segment
        loss_angle1 = normal_2d(theta[:, 2:], phi2[:, 2:], angle_mu1[:, 2:], angle_mu2[:, 2:],
                                angle_sigma1[:, 2:], angle_sigma2[:, 2:], angle_corr[:, 2:])
        loss_angle1 = log_weighted_sum(loss_angle1, angle_pi[:, 2:])
        loss_angle1 = torch.mean(loss_angle1)
        # phi of the second residue in the central segment
        loss_angle2 = normal_1d(phi2[:, 2], angle_mu2[:, 2], angle_sigma2[:, 2])  # (N, 1)
        loss_angle2 = log_weighted_sum(loss_angle2, angle_pi[:, 2])
        loss_angle2 = torch.mean(loss_angle2)

        loss_angle = loss_angle1 + loss_angle2

        return loss_r, loss_angle, loss_profile


class LocalEnergyCE(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.m_r = args.mixture_r
        self.m_angle = args.mixture_angle

        self.random_ref = args.random_ref
        self.smooth_gaussian = args.smooth_gaussian
        self.smooth_r = args.smooth_r
        self.smooth_angle = args.smooth_angle

        self.reduction = args.reduction

        self.seq_loss = nn.CrossEntropyLoss(reduction='none')

        if self.random_ref:
            df = pd.read_csv('data/aa_freq.csv')
            self.aa_freq = df['freq'].values / df['freq'].sum()

        self.start_id_loss = nn.CrossEntropyLoss(reduction='none')

        self.profile_loss_lamda = args.profile_loss_lamda
        self.coords_angle_loss_lamda = args.coords_angle_loss_lamda
        if args.use_position_weights:
            # position weights for each residues in the building blocks
            device = torch.device(args.device)
            position_weights = torch.ones((1, args.seq_len + 1), device=device)
            position_weights[:, 0:5] *= args.cen_seg_loss_lamda
            position_weights[:, 5:] *= args.oth_seg_loss_lamda
            self.position_weights = position_weights
        else:
            self.position_weights = None

    def forward(self, seq, coords, start_id, res_counts):
        # seq = seq.to(self.device)  # (N, L)
        # coords = coords.to(self.device)  # (N, L, 3)
        # start_id = start_id.to(self.device)  # (N, L)
        # target_res_counts = res_counts.to(self.device)  # (N, 3)

        seq_len = seq.shape[-1]

        input_seq = seq[:, :-1]  # (N, L-1, 3)
        input_coords = coords  # (N, L, 3)
        input_start_id = start_id

        target_seq = seq
        target_coords = coords[:, 1:, :]

        out_x, out_s = self.model(input_seq, input_coords, input_start_id)

        # out_x[:, -1, :] is not used, out_x[:, :, -2:] are dimensions for connectivity.
        loss_r, loss_angle = self.get_mixture_loss(out_x[:, :-1, :-2], target_coords)

        loss_seq = self.seq_loss(out_s.transpose(1, 2), target_seq)  # (N, L)

        if self.random_ref:
            aa_freq = torch.tensor(self.aa_freq, dtype=torch.float, device=target_seq.device)
            seq_prob = aa_freq[target_seq]  # (N, L)
            loss_seq_ref = torch.log(seq_prob)
            loss_seq = loss_seq + loss_seq_ref

        # print(out_s[0].transpose(0, 1).argmax(dim=-1), target_seq[0])

        if seq_len > 6:
            target_start_id = start_id[:, 6:]  # ignore the first 6 positions
            out_start_id = out_x[:, 5:-1, -2:].transpose(1, 2)  # ignore the first 5 positions
            # print('out_start_id', out_start_id.shape)
            loss_start_id = self.start_id_loss(out_start_id, target_start_id)
            if self.position_weights is not None:
                # loss_start_id = loss_start_id * self.position_weights[:, 5:-1]  # (N, L-6) * (1, L-6)
                loss_start_id = loss_start_id * self.position_weights[:, 6:]  # (N, L-6) * (1, L-6)
            loss_start_id = torch.sum(loss_start_id, dim=-1)
            if self.reduction != 'keep_batch_dim':
                loss_start_id = torch.mean(loss_start_id)
        else:
            loss_start_id = torch.tensor([0], dtype=torch.float, device=seq.device)

        # only sum the seq dimension (N, L) -> (N)
        if self.position_weights is not None:
            loss_seq = loss_seq * self.position_weights  # (N, L) * (1, L)
        loss_seq = torch.sum(loss_seq, dim=-1)

        if self.reduction != 'keep_batch_dim':
            loss_seq = torch.mean(loss_seq)

        loss_angle *= self.coords_angle_loss_lamda
        loss_seq *= self.profile_loss_lamda

        loss_res_counts = torch.tensor([0], dtype=torch.float, device=seq.device)

        return loss_r, loss_angle, loss_seq, loss_start_id, loss_res_counts

    def get_mixture_coef(self, out):
        """
        m is the number of mixture.
        Returns the Mixture Density Network distribution params.
        This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
        """
        m_r, m_angle = self.m_r, self.m_angle

        N, L, _ = out.size()
        idx1, idx2 = 3*m_r, 3*m_r+6*m_angle
        z_r = out[:, :, 0:idx1].reshape((N, L, m_r, 3))
        z_angle = out[:, :, idx1:idx2].reshape((N, L, m_angle, 6))

        # softmax all the mixture weights:
        # TODO: add eps?
        r_pi = F.softmax(z_r[:, :, :, 0], dim=-1)
        angle_pi = F.softmax(z_angle[:, :, :, 0], dim=-1)

        r_mu = z_r[:, :, :, 1]
        angle_mu1 = z_angle[:, :, :, 1]
        angle_mu2 = z_angle[:, :, :, 2]

        # exponentiate the sigmas
        r_sigma = torch.exp(z_r[:, :, :, 2])
        angle_sigma1 = torch.exp(z_angle[:, :, :, 3])
        angle_sigma2 = torch.exp(z_angle[:, :, :, 4])

        if self.smooth_gaussian:
            r_sigma = r_sigma + self.smooth_r  # default is 0.3 AA
            angle_sigma1 = angle_sigma1 + np.pi / 180.0 * self.smooth_angle  # default is 15 degree
            angle_sigma2 = angle_sigma2 + np.pi / 180.0 * self.smooth_angle

        # make corr between -1 and 1.
        angle_corr = torch.tanh(z_angle[:, :, :, 5])
        # angle_corr = 1.0 or -1.0 wil cause nan in angle loss
        angle_corr = torch.clamp(angle_corr, max=0.99, min=-0.99)

        m_coef = [r_pi, r_mu, r_sigma,
                  angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr
                  ]

        return m_coef

    def get_mixture_loss(self, out, target_coords):
        r, theta, phi = target_coords[:, :, 0:1], target_coords[:, :, 1:2], target_coords[:, :, 2:3]  # (N, L, 1)

        m_coef = self.get_mixture_coef(out)

        r_pi, r_mu, r_sigma, \
        angle_pi, angle_mu1, angle_mu2, angle_sigma1, angle_sigma2, angle_corr = m_coef

        def normal_1d(x, mu, sigma):
            norm = (2*np.pi)**0.5 * sigma
            exp = torch.exp(-0.5 * (x - mu)**2 / sigma**2)
            return exp / norm

        def normal_2d(x1, x2, mu1, mu2, sigma1, sigma2, rho):
            # eps = 1e-6
            z1 = (x1 - mu1)**2 / sigma1**2
            z2 = (x2 - mu2)**2 / sigma2**2
            z12 = (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
            z = z1 + z2 - 2 * rho * z12
            exp = torch.exp(-z / (2 * (1 - rho ** 2)))
            # exp = torch.exp(-z/(2*(1-rho**2)+eps))
            norm = 2 * np.pi * sigma1 * sigma2 * (1-rho**2)**0.5
            norm_2d = exp / norm
            # norm_2d = exp / (norm + eps)
            return norm_2d

        def log_weighted_sum(x, weight):
            x = torch.sum(x * weight, dim=-1)
            # return -torch.log(x + 1e-6)  # x may be much smaller than 1e-6
            return -torch.log(x + 1e-9)

        loss_r = normal_1d(r, r_mu, r_sigma)
        loss_r = log_weighted_sum(loss_r, r_pi)  # (N, L)

        if self.random_ref:
            loss_r_ref = torch.log((r[:, :, 0]/8)**2)  # (N, L), r is normalized by 8A. prob ~ 4*pi * r^2
            loss_r = loss_r + loss_r_ref  # -log(P/Pref) = -log(P) + log(Pref)

        # bring phi-mu into (-pi, pi)
        phi = phi.repeat([1, 1, self.m_angle])  # (N, L, mix_phi)
        assert (phi.shape == angle_mu2.shape)
        idx1 = (angle_mu2 > -np.pi) & (angle_mu2 < np.pi) & (phi - angle_mu2 > np.pi)
        idx2 = (angle_mu2 > -np.pi) & (angle_mu2 < np.pi) & (phi - angle_mu2 < -np.pi)
        phi2 = torch.ones(phi.shape, device=phi.device)
        phi2[:] = phi[:]
        phi2[idx1] = phi[idx1] - 2 * np.pi
        phi2[idx2] = phi[idx2] + 2 * np.pi
        # phi[idx1] -= 2 * np.pi
        # phi[idx2] += 2 * np.pi

        # ignore angles of the first two residues in the central segment
        loss_angle1 = normal_2d(theta[:, 2:], phi2[:, 2:], angle_mu1[:, 2:], angle_mu2[:, 2:],
                                angle_sigma1[:, 2:], angle_sigma2[:, 2:], angle_corr[:, 2:])
        loss_angle1 = log_weighted_sum(loss_angle1, angle_pi[:, 2:])  # (N, L-2)
        # phi of the second residue in the central segment
        # the second residue should be phi2[:, 1]
        loss_angle2 = normal_1d(phi2[:, 1], angle_mu2[:, 1], angle_sigma2[:, 1])
        loss_angle2 = log_weighted_sum(loss_angle2, angle_pi[:, 1])  # (N)
        # print(loss_angle1.shape, loss_angle2.shape)

        if self.random_ref:
            loss_angle1_ref = torch.log(torch.abs(torch.sin(theta[:, 2:, 0])))  # (N, L), theta prob ~ r * sin(theta)
            loss_angle1 = loss_angle1 + loss_angle1_ref  # -log(P/Pref) = -log(P) + log(Pref)

        # only sum the seq dimension (N, L) -> (N)
        if self.position_weights is not None:
            # loss_r = loss_r * self.position_weights[:, :-1]  # (N, L) * (1, L)
            # loss_angle1 = loss_angle1 * self.position_weights[:, 2:-1]  # (N, L-2) * (1, L-2)
            # loss_angle2 = loss_angle2 * self.position_weights[:, 1]
            loss_r = loss_r * self.position_weights[:, 1:]  # (N, L) * (1, L)
            loss_angle1 = loss_angle1 * self.position_weights[:, 3:]  # (N, L-2) * (1, L-2)
            loss_angle2 = loss_angle2 * self.position_weights[:, 2]

        loss_r = torch.sum(loss_r, dim=-1)
        loss_angle1 = torch.sum(loss_angle1, dim=-1)
        loss_angle = loss_angle1 + loss_angle2

        if self.reduction != 'keep_batch_dim':
            loss_r = torch.mean(loss_r)  # (N) -> 1
            loss_angle = torch.mean(loss_angle)

        return loss_r, loss_angle


