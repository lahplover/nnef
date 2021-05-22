import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from Bio.PDB import Selection, PDBParser
import matplotlib.pyplot as pl
from scipy import stats


def extract_beads(pdb_id):
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', f'data/stability/pdb/pdb/{pdb_id}')
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
        ca_center_list.append(res['CA'].get_coord())
        if res.get_resname() != 'GLY':
            cb_center_list.append(res['CB'].get_coord())
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

    df.to_csv(f'data/stability/pdb/beads/{pdb_id}.csv', index=False)


def extract_beads_batch(pdb_list):
    for pdb_id in tqdm(pdb_list):
        extract_beads(pdb_id)


def extract_beads_all():
    pdb_list = pd.read_csv(f'data/stability/pdb/flist.txt')['pdb'].values
    count = len(pdb_list)
    num_cores = 40
    batch_size = count // num_cores + 1
    idx_list = np.arange(count)

    batch_list = []
    for i in range(0, count, batch_size):
        batch = idx_list[i:i+batch_size]
        batch_list.append(pdb_list[batch])

    # setup the multi-processes
    with mp.Pool(processes=num_cores) as pool:
        pool.map(extract_beads_batch, batch_list)


def stability_score():
    df_list = []
    for rd in ['rd1', 'rd2', 'rd3', 'rd4']:
        df = pd.read_csv(f'{rd}_stability_scores', sep='\t')
        df2 = df[['name', 'stabilityscore']]
        df_list.append(df2)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv('rd1234_stability_score.csv', index=False)


def plot_stability_energy():
    # data_file = 'exp61/energy.csv'
    data_file = 'rosetta/rosetta_energy.csv'

    df_sta = pd.read_csv('rd1234_stability_score.csv')
    df_ene = pd.read_csv(data_file)
    pdb_type = df_ene['pdb'].apply(lambda x: x.split('_')[0])
    rd = df_ene['pdb'].apply(lambda x: x.split('_')[1])  # design round 1-4

    sta_dict = {x: y for x, y in zip(df_sta['name'], df_sta['stabilityscore'])}

    sta_score = []
    for x in df_ene['pdb']:
        try:
            sta_score.append(sta_dict[x])
        except KeyError:
            sta_score.append(999)
    sta_score = np.array(sta_score)

    idx = (sta_score != 999) & (sta_score < 100000)
    sta_score2 = sta_score[idx]
    energy = df_ene['energy'].values[idx]
    pdb_t = pdb_type.values[idx]
    rd = rd.values[idx]

    for t in ['EEHEE', 'EHEE', 'HEEH', 'HHH']:
        pl.figure()
        idx = (pdb_t == t)
        # idx = (pdb_t == t) & (rd == 'rd1')
        pl.plot(energy[idx], sta_score2[idx], 'b.')
        pl.xlabel('energy score')
        # pl.ylabel('protease stability score')
        pl.ylabel('stability score')
        pl.title(t)
        pl.savefig(data_file[:-3] + t + '.pdf')

        print(t, stats.pearsonr(energy[idx], sta_score2[idx]))




