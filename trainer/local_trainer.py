import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from optim import ScheduledOptim
from model import LocalEnergyCE


class LocalGenTrainer:
    def __init__(self, writer, model, device, args):
        self.model = model
        self.energy_fn = LocalEnergyCE(model, args)

        self.device = device

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        # self.optim = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, init_lr=args.lr,
                                             n_warmup_steps=args.n_warmup_steps,
                                             steps_decay_scale=args.steps_decay_scale)

        self.log_freq = args.log_interval
        self.writer = writer

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def step(self, data):
        seq, coords, start_id, res_counts = data

        seq = seq.to(self.device)  # (N, L)
        coords = coords.to(self.device)  # (N, L, 3)
        start_id = start_id.to(self.device)  # (N, L)
        res_counts = res_counts.to(self.device)  # (N, 3)

        loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts = self.energy_fn.forward(seq, coords, start_id, res_counts)
        return loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts

    def train(self, epoch, data_loader, flag='Train'):
        for i, data in tqdm(enumerate(data_loader)):
            loss_r, loss_angle, loss_profile, loss_start_id, loss_res_counts = self.step(data)
            loss = loss_r + loss_angle + loss_profile + loss_start_id + loss_res_counts

            if flag == 'Train':
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            len_data_loader = len(data_loader)
            if flag == 'Train':
                log_freq = self.log_freq
            else:
                log_freq = 1
            if i % log_freq == 0:
                self.writer.add_scalar(f'{flag}/profile_loss', loss_profile.item(), epoch * len_data_loader + i)
                self.writer.add_scalar(f'{flag}/coords_radius_loss', loss_r.item(), epoch * len_data_loader + i)
                self.writer.add_scalar(f'{flag}/coords_angle_loss', loss_angle.item(), epoch * len_data_loader + i)
                self.writer.add_scalar(f'{flag}/start_id_loss', loss_start_id.item(), epoch * len_data_loader + i)
                self.writer.add_scalar(f'{flag}/res_counts_loss', loss_res_counts.item(), epoch * len_data_loader + i)
                self.writer.add_scalar(f'{flag}/total_loss', loss.item(), epoch * len_data_loader + i)

                print(f'{flag} epoch {epoch} Iter: {i} '
                      f'profile_loss: {loss_profile.item():.3f} '
                      f'coords_radius_loss: {loss_r.item():.3f} '
                      f'coords_angle_loss: {loss_angle.item():.3f} '
                      f'start_id_loss: {loss_start_id.item():.3f} '
                      f'res_counts_loss: {loss_res_counts.item():.3f} '
                      f'total_loss: {loss.item():.3f} ')

    def test(self, epoch, data_loader, flag='Test'):
        self.model.eval()
        torch.set_grad_enabled(False)

        self.train(epoch, data_loader, flag=flag)

        self.model.train()
        torch.set_grad_enabled(True)



