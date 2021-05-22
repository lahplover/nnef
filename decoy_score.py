import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import options
# from dataset import DatasetPDBProfile
from dataset import DatasetLocalGenOS
from trainer.local_gen_os_trainer import LocalGenTrainer
from physics.protein_os import Protein
# from physics.protein_os import Protein, EnergyFun, ProteinBase
from model import LocalTransformer
from utils import load_protein_decoy
from physics.grad_minimizer import GradMinimizerCartesian
from utils import test_setup


parser = options.get_decoy_parser()
args = options.parse_args_and_arch(parser)

#################################################
if args.static_decoy:
    writer = SummaryWriter('./runs/test/')
    device = torch.device(args.device)
    model = LocalTransformer(args)
    model.load_state_dict(torch.load(f'{args.load_exp}/models/model.pt', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    trainer = LocalGenTrainer(writer, model, device, args)
else:
    # energy_fn = EnergyFun(model, args)
    # ProteinBase.k = args.seq_len - 4
    device, model, energy_fn, ProteinBase = test_setup(args)

#################################################

# decoy_set = '3DRobot_set'
# decoy_set = 'casp11'
decoy_set = args.decoy_set
decoy_loss_dir = args.decoy_loss_dir

if not os.path.exists(f'data/decoys/{decoy_set}/{decoy_loss_dir}'):
    os.system(f'mkdir -p data/decoys/{decoy_set}/{decoy_loss_dir}')

if decoy_set == '3DRobot_set':
    # pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/pdb_profile_diff.txt')['pdb'].values
    # pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/pdb_profile_diff_match.txt')['pdb'].values
    pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/pdb_no_missing_residue.csv')['pdb'].values
elif decoy_set == 'casp11':
    # pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/pdb_list_new.txt')['pdb'].values
    pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/no_missing_residue.txt')['pdb'].values
elif (decoy_set == 'casp13') | (decoy_set == 'casp14'):
    pdb_list = pd.read_csv(f'data/decoys/{decoy_set}/pdb_list.txt')['pdb'].values
else:
    raise ValueError('decoy_set should be casp11 / 3DRobot_set / casp13')

# pdb_list = ['1HF2A']
for pdb_id in tqdm(pdb_list):
    if os.path.exists(f'data/decoys/{decoy_set}/{decoy_loss_dir}/{pdb_id}_decoy_loss.csv'):
        continue
    if decoy_set == '3DRobot_set':
        # df = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.txt', sep='\s+')
        df = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.csv')
        decoy_list = df['NAME'].values
    elif decoy_set == 'casp11':
        df = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/list.txt', header=None, names=['NAME'])
        decoy_list = df['NAME'].values
    elif (decoy_set == 'casp13') | (decoy_set == 'casp14'):
        df = pd.read_csv(f'data/decoys/{decoy_set}/{pdb_id}/flist.txt')
        decoy_list = df['pdb'].values
    else:
        raise ValueError('decoy_set should be casp11 / 3DRobot_set / casp13 / casp14')
    loss_all = []

    if args.static_decoy:
        for decoy in decoy_list:
            data_path = f'data/decoys/{decoy_set}/{pdb_id}/{decoy[:-4]}_local_rot_CA.h5'
            if not os.path.exists(data_path):
                loss_all.append(999)
                continue
            test_data = h5py.File(data_path, 'r')
            test_dataset = DatasetLocalGenOS(test_data, args)
            datasampler = SequentialSampler(test_dataset)
            data_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=datasampler)

            for i, data in enumerate(data_loader):
                loss_terms = trainer.step(data)
                loss = 0
                for loss_i in loss_terms:
                    loss += loss_i.item()

                loss_all.append(loss)
    else:
        for decoy in decoy_list:
            if (decoy_set == 'casp13') | (decoy_set == 'casp14'):
                decoy_id = decoy
            else:
                decoy_id = decoy[:-4]
            seq, coords_native, profile = load_protein_decoy(pdb_id, decoy_id, args.mode, device, args)

            protein = Protein(seq, coords_native, profile)
            energy = protein.get_energy(energy_fn).item()
            # print('energy:', energy)
            # residue_energy = protein.get_residue_energy(energy_fn)
            # print(residue_energy)

            if args.relax:
                minimizer = GradMinimizerCartesian(energy_fn, protein, num_steps=args.relax_steps)
                minimizer.run()
                energy = minimizer.energy_best
                print('energy relaxed:', energy)
            loss_all.append(energy)

    print(pdb_id, loss_all[0])
    loss_all = np.array(loss_all)
    df['loss'] = loss_all
    df.to_csv(f'data/decoys/{decoy_set}/{decoy_loss_dir}/{pdb_id}_decoy_loss.csv', index=False)

