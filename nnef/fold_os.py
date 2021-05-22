import numpy as np
import pandas as pd
import torch
from physics.protein_os import Protein
import options
from utils import write_pdb, write_pdb_sample, transform_profile, load_protein
from physics.anneal import AnnealCoords, AnnealFrag
# from physics.move import SampleICNext
from physics.grad_minimizer import *
from physics.dynamics import *
import os
import mdtraj as md
from utils import test_setup
import h5py


#################################################
parser = options.get_fold_parser()
args = options.parse_args_and_arch(parser)

device, model, energy_fn, ProteinBase = test_setup(args)

# position_weights = torch.zeros((1, args.seq_len + 1), device=device)
# position_weights[:, 0:5] = 1
# energy_fn.energy_fn.position_weights = position_weights


#################################################
data_path = 'data/fold/cullpdb_val_deep'
protein_sample = pd.read_csv(f'{data_path}/sample.csv')
pdb_selected = protein_sample['pdb'].values
np.random.shuffle(pdb_selected)

fold_engine = args.fold_engine
mode = args.mode
# sample_ic = SampleICNext(mode)
exp_id = args.load_exp[-5:]
save_dir = args.save_dir
# if not os.path.exists(f'data/fold/{exp_id}'):
#     os.mkdir(f'data/fold/{exp_id}')
if not os.path.exists(f'data/fold/{save_dir}'):
    os.mkdir(f'data/fold/{save_dir}')


for pdb_id in pdb_selected:

    seq, coords_native, profile = load_protein(data_path, pdb_id, mode, device, args)

    protein_native = Protein(seq, coords_native, profile)
    energy_native = protein_native.get_energy(energy_fn).item()
    print('energy_native:', energy_native)
    rg2, collision = protein_native.get_rad_gyration(coords_native)
    print('native radius of gyration square:', rg2.item())
    # residue_energy = protein_native.get_residue_energy(energy_fn)
    # print(residue_energy)
    # write_pdb(seq, coords_native, pdb_id, 'native', exp_id)

    protein = Protein(seq, coords_native.clone(), profile.clone())
    if args.random_init:
        # random_coords_int = sample_ic.random_coords_int(len(seq)-3).to(device)
        # protein.update_coords_internal(random_coords_int)
        # extend_coords_int = torch.tensor([[5.367, 1.6, 0.0]], device=device).repeat((len(seq)-3, 1))
        extend_coords_int = torch.tensor([[5.367, 0.1, 0.0]], device=device).repeat((len(seq)-3, 1))
        protein.update_coords_internal(extend_coords_int)
        protein.update_cartesian_from_internal()
    coords_init = protein.coords

    energy_init = protein.get_energy(energy_fn).item()
    print('energy_init:', energy_init)
    # write_pdb(seq, coords_init, pdb_id, f'init_{mode}', exp_id)

    if fold_engine == 'anneal':
        # simulated annealing
        torch.set_grad_enabled(False)
        if args.anneal_type == 'int_one':
            annealer = AnnealCoords(energy_fn, protein, mode=mode, ic_move_std=args.ic_move_std,
                                    T_max=args.T_max, T_min=args.T_min, L=args.L)
        elif args.anneal_type == 'frag':
            frag_file = h5py.File(f'data/fragment/{pdb_id}/{pdb_id}_int.h5', 'r')
            query_pos = torch.tensor(frag_file['query_pos'][()], device=device)
            frag_int = torch.tensor(frag_file['coords_int'][()], device=device)
            annealer = AnnealFrag(energy_fn, protein, frag=(query_pos, frag_int), use_rg=args.use_rg,
                                  T_max=args.T_max, T_min=args.T_min, L=args.L)
        else:
            raise ValueError('anneal_type should be int_one / frag.')
        annealer.run()
        coords_best = annealer.x_best
        energy_best = annealer.energy_best
        sample = annealer.sample
        sample_energy = annealer.sample_energy
    elif fold_engine == 'grad':
        if args.x_type == 'cart':
            minimizer = GradMinimizerCartesian(energy_fn, protein, lr=args.lr, num_steps=args.L)
        elif args.x_type == 'internal':
            minimizer = GradMinimizerInternal(energy_fn, protein, lr=args.lr, num_steps=args.L, momentum=0.0)
        elif args.x_type == 'int_fast':
            minimizer = GradMinimizerIntFast(energy_fn, protein, lr=args.lr, num_steps=args.L)
        elif args.x_type == 'mixed':
            minimizer = GradMinimizerMixed(energy_fn, protein, lr=args.lr, num_steps=args.L)
        elif args.x_type == 'mix_fast':
            minimizer = GradMinimizerMixFast(energy_fn, protein, lr=args.lr, num_steps=args.L)
        else:
            raise ValueError('x_type should be cart / internal / mixed / int_fast / mix_fast.')

        minimizer.run()
        coords_best = minimizer.x_best
        energy_best = minimizer.energy_best
        sample = minimizer.sample
        sample_energy = minimizer.sample_energy
    elif fold_engine == 'dynamics':
        if args.x_type == 'cart':
            minimizer = Dynamics(energy_fn, protein, num_steps=args.L, lr=args.lr, t_noise=args.T_max)
        elif args.x_type == 'internal':
            minimizer = DynamicsInternal(energy_fn, protein, num_steps=args.L, lr=args.lr, t_noise=args.T_max)
        elif args.x_type == 'int_fast':
            minimizer = DynamicsIntFast(energy_fn, protein, num_steps=args.L, lr=args.lr, t_noise=args.T_max)
        elif args.x_type == 'mixed':
            minimizer = DynamicsMixed(energy_fn, protein, num_steps=args.L, lr=args.lr, t_noise=args.T_max)
        elif args.x_type == 'mix_fast':
            minimizer = DynamicsMixFast(energy_fn, protein, num_steps=args.L, lr=args.lr, t_noise=args.T_max)
        else:
            raise ValueError('x_type should be cart / internal / mixed / int_fast / mix_fast.')

        minimizer.run()
        coords_best = minimizer.x_best
        energy_best = minimizer.energy_best
        sample = minimizer.sample
        sample_energy = minimizer.sample_energy
    else:
        raise ValueError('fold_engine should be anneal / grad / dynamics')

    # protein.update_coords(coords_best)
    # residue_energy = protein.get_residue_energy(energy_fn)
    # print(residue_energy)
    # write_pdb(seq, coords_best, pdb_id, f'best_{mode}', exp_id)

    # save sampled structures
    sample = [coords_native.cpu(), coords_best.cpu(), coords_init.cpu()] + sample
    sample_energy = [energy_native, energy_best, energy_init] + sample_energy
    # write_pdb_sample(seq, sample, pdb_id, 'sample', exp_id)
    # pd.DataFrame({'sample_energy': sample_energy}).to_csv(f'data/fold/{exp_id}/{pdb_id}_energy.csv', index=False)
    write_pdb_sample(seq, sample, pdb_id, 'sample', save_dir)

    # compute RMSD,
    sample_xyz = torch.stack(sample, 0).cpu().detach().numpy()
    print(sample_xyz.shape)
    t = md.Trajectory(xyz=sample_xyz, topology=None)
    t = t.superpose(t, frame=0)
    write_pdb_sample(seq, t.xyz, pdb_id, 'sample2', save_dir)
    sample_rmsd = md.rmsd(t, t, frame=0)  # computation will change sample_xyz;

    print(f'best RMSD: {sample_rmsd[1]}')
    df = pd.DataFrame({'sample_energy': sample_energy,
                       'sample_rmsd': sample_rmsd})
    df.to_csv(f'data/fold/{save_dir}/{pdb_id}_energy.csv', index=False)





