import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp


def select_small_sample():
    df = pd.read_csv('data/hhsuite_CB_cullpdb_val_no_missing_residue.csv')
    df = df.sample(100)
    df.to_csv('data/decoys/decoys_complex/hhsuite_CB_cullpdb_val_no_missing_residue_sample2.csv', index=False)


def prepare_decoy_complex():
    # prepare decoys of pseudo protein complex
    # cut the small protein in the middle, add a chain number, save new bead file
    df = pd.read_csv('data/decoys/decoys_complex/hhsuite_CB_cullpdb_val_no_missing_residue_sample2.csv')
    pdb_list = df['pdb'].values
    for pdb in pdb_list:
        df = pd.read_csv(f'../hhsuite/hhsuite_beads/hhsuite/{pdb}_bead.csv')
        num_res = df.shape[0]
        n_cut = np.random.randint(int(num_res*0.4), int(num_res*0.6))
        chain = np.zeros(num_res, dtype=np.int)
        chain[n_cut:] = 1
        df['chain'] = chain
        df_a = df[:n_cut]
        df_b = df[n_cut:]
        df.to_csv(f'data/decoys/decoys_complex/{pdb}_ab.csv', index=False)
        df_a.to_csv(f'data/decoys/decoys_complex/{pdb}_a.csv', index=False)
        df_b.to_csv(f'data/decoys/decoys_complex/{pdb}_b.csv', index=False)









