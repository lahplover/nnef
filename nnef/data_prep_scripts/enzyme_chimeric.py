import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as pl


def make_chimeric_enzyme():
    AA = '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    aa_to_id = {y: x for x, y in enumerate(list(AA))}
    id_to_aa = {x: y for x, y in enumerate(list(AA))}
    root_dir = '/home/hyang/bio/erf/data/stability/enzyme'
    pdb_list = ['6THT_A', '6EQE_A', '5ZOA_A']
    for pdb in tqdm(pdb_list):
        # read sequences
        protein_name = []
        seq = []
        ins_num = []
        with open(f'{root_dir}/{pdb}.fil.a3m', 'rt') as msa_file:
            for line in msa_file:
                if line[0] == '>':
                    protein_name.append(line.split(' ')[0][1:])
                if line[0] != '>':
                    s = np.array([aa_to_id[x] for x in line[:-1]])
                    s2 = s[s <= 26]  # remove lower case letter
                    ins_num.append(len(s[s > 26]))
                    seq.append(s2)
        seq = np.vstack(seq)
        ins_num = np.array(ins_num)
        # make chimeric seq
        ref = seq[0]  # the first seq is the PDB seq
        chimeric = [''.join([id_to_aa[x] for x in ref])]
        hamming_dist = [0]
        seq_len = len(ref)
        del_frac = [0]
        ins_frac = ins_num / seq_len
        for s in seq[1:]:
            idx = (s == 0)  # 0 is the index of '-'
            del_frac.append(len(s[idx]) * 1.0 / seq_len)
            s[idx] = ref[idx]  # replace '-' to AA in reference seq
            hamming_dist.append(np.sum(s != ref) * 1.0 / seq_len)
            chimeric.append(''.join([id_to_aa[x] for x in s]))
        df = pd.DataFrame({'protein_name': protein_name,
                           'ins_frac': ins_frac, 'del_frac': del_frac,
                           'hamming_dist': hamming_dist, 'seq': chimeric})
        df.to_csv(f'{root_dir}/{pdb}.chimeric', index=False, float_format='%.3f')


def plot_enzyme_score():
    root_dir = '/home/hyang/bio/erf/data/stability/enzyme'
    pdb_list = ['6THT_A', '6EQE_A', '5ZOA_A']

    for pdb in pdb_list:
        df = pd.read_csv(f'{root_dir}/{pdb}_chimeric_energy.csv')
        energy = df['energy_score'].values
        idx = (energy > 0)
        energy2 = energy[idx]
        print(energy2.min(), energy2.max())

        # fig = pl.figure()
        # pl.hist(energy2, bins=np.arange(25)*10+910)
        # pl.xlabel('energy score')
        # pl.ylabel('N')
        # pl.title(f'{pdb} chimeric')
        # pl.savefig(f'{root_dir}/{pdb}_energy_hist.pdf')
        # pl.close()

        idx = (energy > 0) & (energy < 970)
        protein_name = df['protein_name'].values[idx]
        print(protein_name, energy[idx])


def check_uniprot():
    root_dir = '/home/plover/study/bio/play/erf/data/stability/enzyme'
    pdb_list = ['6THT_A', '6EQE_A', '5ZOA_A']
    petase_family = pd.read_csv(f'{root_dir}/PETase_subfamily.csv')['UniProtAcc'].values

    protein_name_all = np.array([])
    for pdb in pdb_list:
        df = pd.read_csv(f'{root_dir}/{pdb}_chimeric_energy.csv')
        protein_name = df['protein_name'].apply(lambda x: x[3:].split('|')[0])
        protein_name_all = np.append(protein_name_all, protein_name)

    p_list = []
    for p in petase_family:
        if p in protein_name_all:
            p_list.append(p)




