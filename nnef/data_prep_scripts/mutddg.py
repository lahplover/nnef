import pandas as pd
import numpy as np
from Bio.PDB import Selection, PDBParser
from tqdm import tqdm
import os


def prepare_ddg():
    df = pd.read_csv('skempi_v2.csv', sep=';')
    kd_wt = df['Affinity_wt_parsed'].values
    kd_mut = df['Affinity_mut_parsed'].values
    # temp = df['Temperature'].values

    idx = ~(np.isnan(kd_wt) | np.isnan(kd_mut))
    df = df[idx]

    df['pdb'] = df['#Pdb'].apply(lambda x: x[:4])
    kd_wt = df['Affinity_wt_parsed'].values
    kd_mut = df['Affinity_mut_parsed'].values
    df['ddg'] = (8.314/4184) * (273.15 + 25.0) * np.log(kd_mut / kd_wt)

    # ddg = df['ddg'].values
    df2 = df[['pdb', 'Mutation(s)_cleaned', 'ddg']]
    df2.to_csv('skempi_v2_ddg.csv', sep=';', index=False, float_format='%.4f')


def extract_beads(pdb_id):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', f'data/mutddg/PDBs/{pdb_id}.pdb')
    residue_list = Selection.unfold_entities(structure, 'R')

    ca_center_list = []
    cb_center_list = []
    res_name_list = []
    res_num_list = []
    chain_list = []

    for res in residue_list:
        if res.get_resname() not in vocab_aa:
            # raise ValueError('protein has non natural amino acids')
            continue
        chain_list.append(res.parent.id)
        res_name_list.append(vocab_dict[res.get_resname()])
        res_num_list.append(res.id[1])

        try:
            ca_center_list.append(res['CA'].get_coord())
        except KeyError:
            print(f'{pdb_id}, {res} missing CA')
            return 0
        if res.get_resname() != 'GLY':
            try:
                cb_center_list.append(res['CB'].get_coord())
            except KeyError:
                print(f'{pdb_id}, {res} missing CB')
                return 0
        else:
            cb_center_list.append(res['CA'].get_coord())

    ca_center = np.vstack(ca_center_list)
    cb_center = np.vstack(cb_center_list)

    df = pd.DataFrame({'chain_id': chain_list,
                       'group_num': res_num_list,
                       'group_name': res_name_list,
                       'x': ca_center[:, 0],
                       'y': ca_center[:, 1],
                       'z': ca_center[:, 2],
                       'xcb': cb_center[:, 0],
                       'ycb': cb_center[:, 1],
                       'zcb': cb_center[:, 2]})

    # assign "chain" number for the energy calculation
    chain = np.zeros(df.shape[0], dtype=np.int)
    chain_id = df['chain_id'].values
    group_num = df['group_num'].values
    count = 0
    chain_0 = chain_id[0]
    group_0 = group_num[0]

    # if type(group_0) is str:
    #     print(pdb_id, 'group_num has string')

    for i in range(1, df.shape[0]):
        chain_i = chain_id[i]
        group_i = group_num[i]
        if (chain_i == chain_0) & (group_i == group_0 + 1):
            group_0 += 1
        else:
            count += 1
            chain_0 = chain_i
            group_0 = group_i
        chain[i] = count
    df['chain'] = chain

    df.to_csv(f'data/mutddg/PDBs/{pdb_id}_bead.csv', index=False)
    return 1


def extract_mutddg():
    df = pd.read_csv('data/mutddg/skempi_v2_ddg.csv', sep=';')
    pdb_list = df['pdb'].unique()

    failed_pdb_list = []
    for pdb_id in tqdm(pdb_list):
        if os.path.exists(f'data/mutddg/PDBs/{pdb_id}_bead.csv'):
            continue
        result = extract_beads(pdb_id)
        if result == 0:
            failed_pdb_list.append(pdb_id)

    ind = np.zeros(df.shape[0], dtype=np.int)
    pdb = df['pdb'].values
    for i in range(ind.shape[0]):
        if pdb[i] in failed_pdb_list:
            ind[i] = 1
    df2 = df[ind == 0]
    df2.to_csv('data/mutddg/skempi_v2_ddg_fil.csv', sep=';', index=False)

    df_fail = pd.DataFrame({'pdb_missing_atom': failed_pdb_list})
    df_fail.to_csv('data/mutddg/pdb_missing_atom.csv', index=False)


def two_body_interface():
    df = pd.read_csv('data/mutddg/skempi_v2_ddg_fil.csv', sep=';')
    pdb_list = df['pdb'].unique()

    pdb_dict = {}
    for pdb_id in pdb_list:
        df_beads = pd.read_csv(f'data/mutddg/PDBs/{pdb_id}_bead.csv')
        chain = df_beads['chain']
        num_chain = df_beads['chain'].nunique()
        if num_chain <= 2:
            # make sure each chain is long enough for energy calculation
            if (chain[chain == 0].shape[0] > 15) & (chain[chain == 1].shape[0] > 15):
                pdb_dict[pdb_id] = 0
            else:
                pdb_dict[pdb_id] = 1
        else:
            pdb_dict[pdb_id] = 1

    two_body_idx = df['pdb'].apply(lambda x: pdb_dict[x])
    df2 = df[two_body_idx == 0]
    df2.to_csv('data/mutddg/skempi_v2_ddg_fil_twobody.csv', sep=';', index=False)




