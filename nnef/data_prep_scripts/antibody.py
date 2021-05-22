import pandas as pd
import numpy as np
from Bio.PDB import Selection, PDBParser
from tqdm import tqdm
import os


def extract_beads(pdb_id):
    # pdb_id = 'wtf'
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', f'data/mutddg/antibody/{pdb_id}.pdb')
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

    df.to_csv(f'data/mutddg/antibody/{pdb_id}_bead.csv', index=False)
    return 1


def find_mutations():
    df1 = pd.read_csv('wtf_bead.csv')
    df2 = pd.read_csv('m29f_bead.csv')
    chain_id1 = df1['chain_id'].values
    group_num1 = df1['group_num'].values
    chain_id2 = df2['chain_id'].values
    group_num2 = df2['group_num'].values
    seq1 = df1['group_name'].values
    seq2 = df2['group_name'].values
    for i in range(df1.shape[0]):
        if seq1[i] != seq2[i]:
            # print(chain_id1[i], group_num1[i], chain_id2[i], group_num2[i], seq1[i], seq2[i])
            print(f'{seq1[i]}{chain_id1[i]}{group_num1[i]}{seq2[i]}')











