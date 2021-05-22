import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def casp_match():
    """
    match the CASP structures and the PDBs using the seq_index in PDBs files.
    This method does not work because the sequences used in CASP are different from the sequences in PDB files.
    """
    casp_set = '11'
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    casp_df = pd.read_csv(f'data/decoys/casp/casp/casp11.csv')
    casp_id_pdb = {x: y for x, y in zip(casp_df['casp11_id'], casp_df['pdb'])}
    casp_id_chain = {x: y for x, y in zip(casp_df['casp11_id'], casp_df['chain'])}

    casp_id_matched = []
    casp = pd.read_csv(f'data/decoys/casp11/pdb_list.txt')['pdb'].values
    for casp_id in tqdm(casp):
        # pdb_list = pd.read_csv(f'data/decoys/casp13/{casp_id}/flist.txt')['pdb'].values
        # df_casp = pd.read_csv(f'data/decoys/casp13/{casp_id}/{pdb_list[0]}_bead.csv')

        pdb_list = pd.read_csv(f'data/decoys/casp11/{casp_id}/list.txt',
                               header=None, names=['NAME'])['NAME'].values
        df_casp = pd.read_csv(f'data/decoys/casp11/{casp_id}/{pdb_list[0][:-4]}_bead.csv')

        try:
            pdb_id = casp_id_pdb[casp_id]
            chain_id = casp_id_chain[casp_id]
        except KeyError:
            print(f'{casp_id} do not have PDB chain data')
            continue

        pdb_data_path = f'data/decoys/casp/casp/{pdb_id}_{chain_id}_bead.csv'
        if not os.path.exists(pdb_data_path):
            print(f'{casp_id} do not have PDB chain data')
            continue
        df_pdb = pd.read_csv(pdb_data_path)

        group_num = df_pdb['group_num'].values  # id in protein sequence
        if group_num.max() >= df_casp['group_name'].shape[0]:
            print(f'{casp_id}: casp sequence is too short.')
            continue

        seq_pdb = ''.join(df_pdb['group_name'].apply(lambda x: vocab_dict[x]).values)
        seq_casp = ''.join(df_casp['group_name'].values[group_num])
        if seq_pdb == seq_casp:
            casp_id_matched.append(casp_id)
        else:
            print(f'{casp_id}: casp sequence do not match pdb sequence.')
            break


def check_casp13_missing_residues():
    df = pd.read_csv('data/decoys/casp/casp13.csv')
    pdb_name = df['pdb'].values
    chain_name = df['chain'].values

    no_missing_res_list = []
    seq_len_list = []
    idx = np.zeros(df.shape[0])
    for i, pdb in tqdm(enumerate(pdb_name)):
        pdb_chain = pdb_name[i] + '_' + chain_name[i]
        data_path = f'data/decoys/casp/{pdb_chain}_bead.csv'
        if not os.path.exists(data_path):
            continue
        df_i = pd.read_csv(data_path)
        group_num_pdb = df_i['group_num_pdb'].values
        group_num = df_i['group_num'].values
        if type(group_num_pdb[0]) == str:
            continue
        num_res_pdb = group_num_pdb[-1] - group_num_pdb[0] + 1
        if group_num.shape[0] == num_res_pdb:
            no_missing_res_list.append(pdb)
            seq_len_list.append(group_num.shape[0])
            idx[i] = 1
    df2 = df[idx == 1.0].copy()
    df2.to_csv('data/decoys/casp/casp13_no_missing_residues.csv', index=False)

