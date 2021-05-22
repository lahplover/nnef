import pandas as pd
import numpy as np
from tqdm import tqdm


df = pd.read_csv('cath-b-newest-all-2020.08.12.txt', sep='\s+', header=None,
                 names=['pdb', 'flag', 'cath', 'boundary'])
pdb = df['pdb'].apply(lambda x: x[:-2])
cath = df['cath']
cat = df['cath'].apply(lambda x: '.'.join(x.split('.')[0:3]))

pdb_cat_dict = {}
pdb_cath_dict = {}
cat_pdb_dict = {}
cath_pdb_dict = {}

for i in tqdm(range(pdb.shape[0])):
    pdb_cat_dict.setdefault(pdb[i], [])
    pdb_cat_dict[pdb[i]].append(cat[i])
    pdb_cath_dict.setdefault(pdb[i], [])
    pdb_cath_dict[pdb[i]].append(cath[i])

    cat_pdb_dict.setdefault(cat[i], [])
    cat_pdb_dict[cat[i]].append(pdb[i])
    cath_pdb_dict.setdefault(cath[i], [])
    cath_pdb_dict[cath[i]].append(pdb[i])


# partition train and test on the level of Class
df = pd.read_csv('hhsuite_CB_pdb_list_cullpdb.csv')
pdb_list = df['pdb'].values

idx = np.zeros(df.shape[0])
for i, p in enumerate(pdb_list):
    pc, c = p.split('_')
    pc = pc.lower() + c
    try:
        cat_list = np.unique(np.array([c.split('.')[0] for c in pdb_cat_dict[pc]]))
        if cat_list.shape[0] == 1:  # the chain has one CATH class
            if cat_list[0] == '1':  # mainly alpha
                idx[i] = 1
            elif cat_list[0] == '2':  # mainly beta
                idx[i] = 2
            elif cat_list[0] == '3':   # alpha beta
                idx[i] = 3
            elif cat_list[0] == '4':   # irregular
                idx[i] = 4
            else:
                idx[i] = 99  # other cases (new classes? such as 6.10.250)
        else:
            idx[i] = 5   # mixed classes in a single chain
    except KeyError:
        idx[i] = 0   # not covered in CATH

df_alpha = df[idx == 1]
df_beta = df[idx == 2]
df_alpha_beta = df[idx == 3]
df_irregular = df[idx == 4]
df_mix = df[idx == 5]
df_no_cath = df[idx == 0]

df_alpha.to_csv(f'hhsuite_CB_cullpdb_cath-alpha.csv', index=False)
df_beta.to_csv(f'hhsuite_CB_cullpdb_cath-beta.csv', index=False)
df_alpha_beta.to_csv(f'hhsuite_CB_cullpdb_cath-alpha-beta.csv', index=False)
df_irregular.to_csv(f'hhsuite_CB_cullpdb_cath-irregular.csv', index=False)
df_mix.to_csv(f'hhsuite_CB_cullpdb_cath-mix.csv', index=False)
df_no_cath.to_csv(f'hhsuite_CB_cullpdb_cath-nocath.csv', index=False)

df2 = df_alpha_beta.sample(frac=1.0)
df_train = df2[:-500]
df_val = df2[-500:]
df_train.to_csv(f'hhsuite_CB_cullpdb_cath-alpha-beta_train.csv', index=False)
df_val.to_csv(f'hhsuite_CB_cullpdb_cath-alpha-beta_val.csv', index=False)


def cullpdb_cath_no_missing_residue():
    df_cath = pd.read_csv('data/hhsuite_CB_cullpdb_cath-alpha-beta.csv')
    pdb_list = df_cath['pdb'].values
    no_missing_res_list = []
    seq_len_list = []
    idx = np.zeros(df_cath.shape[0])
    for i, pdb in tqdm(enumerate(pdb_list)):
        df = pd.read_csv(f'../hhsuite/hhsuite_beads/hhsuite/{pdb}_bead.csv')
        group_num_pdb = df['group_num_pdb'].values
        group_num = df['group_num'].values
        if type(group_num_pdb[0]) == str:
            continue
        num_res_pdb = group_num_pdb[-1] - group_num_pdb[0] + 1
        if group_num.shape[0] == num_res_pdb:
            no_missing_res_list.append(pdb)
            seq_len_list.append(group_num.shape[0])
            idx[i] = 1
    df_cath2 = df_cath[idx == 1.0].copy()
    df_cath2['seq_len'] = np.array(seq_len_list)
    df_cath3 = df_cath2[df_cath2['seq_len'] < 500]
    df_cath3.to_csv('data/hhsuite_CB_cullpdb_cath_funnel.csv', index=False)


def ref_energy_sample():
    # select a sample to calibrate the REF energy for protein design
    df = pd.read_csv('data/hhsuite_CB_cullpdb_cath_funnel.csv')
    sl = df['seq_len']
    df2 = df[(sl < 120) & (sl > 50)]
    df2.to_csv('data/hhsuite_CB_cullpdb_cath_funnel_ref.csv', index=False)
    with open('data/design/ref/cp.sh', 'w') as f:
        for pdb in df2['pdb']:
            f.write(f'cp /home/hyang/bio/hhsuite/hhsuite_beads/hhsuite/{pdb}_bead.csv .\n')


def save_funnel_h5():
    import h5py
    df_cath = pd.read_csv('data/hhsuite_CB_cullpdb_cath_funnel.csv')
    pdb_list = df_cath['pdb'].values
    hh_data_seq = h5py.File('data/hhsuite_pdb_seq_cullpdb.h5', 'r', libver='latest', swmr=True)
    amino_acids = pd.read_csv(f'data/amino_acids.csv')
    vocab = {x.upper(): y-1 for x, y in zip(amino_acids.AA3C, amino_acids.idx)}

    with h5py.File(f'data/hhsuite_CB_cullpdb_cath_funnel.h5', 'w') as f:
        for pdb_id in tqdm(pdb_list):
            seq_hh = hh_data_seq[pdb_id][()]
            if seq_hh.shape[0] > 0:
                seq_hh = seq_hh[0]
            else:
                print(pdb_id)
                continue

            df_beads = pd.read_csv(f'../hhsuite/hhsuite_beads/hhsuite/{pdb_id}_bead.csv')
            seq = df_beads['group_name'].values
            seq = np.array([vocab[x] for x in seq])
            group_num = df_beads['group_num'].values
            seq_hh = seq_hh[group_num]

            if np.sum((seq - seq_hh) ** 2) != 0:
                print(pdb_id)
                continue

            coords = df_beads[['xcb', 'ycb', 'zcb']].values
            dset = f.create_dataset(pdb_id, shape=coords.shape, data=coords, dtype='f4')
            dset = f.create_dataset(pdb_id+'_group_num', shape=group_num.shape, data=group_num, dtype='i')


def train_val_partition():
    df = pd.read_csv(f'data/hhsuite_CB_cullpdb_cath_funnel.csv')
    df2 = df.sample(frac=1.0)
    df_train = df2[:3200]
    df_val = df2[3200:]
    df_train.to_csv(f'data/hhsuite_CB_cullpdb_cath_funnel_train.csv', index=False)
    df_val.to_csv(f'data/hhsuite_CB_cullpdb_cath_funnel_val.csv', index=False)


def check_train_test():
    def get_cath(pdb_list):
        cat_list = []
        cath_list = []
        no_cath = []
        for p in pdb_list:
            p, c = p.split('_')
            p = p.lower() + c
            try:
                cat_list.extend(pdb_cat_dict[p])
                cath_list.extend(pdb_cath_dict[p])
            except KeyError:
                no_cath.append(p)
        cat_unique = np.unique(np.array(cat_list))
        cath_unique = np.unique(np.array(cath_list))
        no_cath = np.array(no_cath)
        return cat_unique, cath_unique, no_cath

    # train set
    train_pdb_list = pd.read_csv('hhsuite_CB_cullpdb_train.csv')['pdb'].values
    train_cat, train_cath, train_no_cath = get_cath(train_pdb_list)

    # test set
    val_pdb_list = pd.read_csv('hhsuite_CB_cullpdb_val.csv')['pdb'].values
    val_cat, val_cath, val_no_cath = get_cath(val_pdb_list)
    val_pdb_no_missing_list = pd.read_csv('hhsuite_CB_cullpdb_val_no_missing_residue.csv')['pdb'].values

    # all = set(train_cath) | set(val_cath)
    # common = set(train_cath) & set(val_cath)
    val_diff = set(val_cat) - set(train_cat)
    val_diff_pdb = []
    for c in val_diff:
        val_diff_pdb.extend(cat_pdb_dict[c])
    val_diff_pdb = np.unique(np.array(val_diff_pdb))

    val_pdb_unique = []
    for p in val_pdb_list:
        pc, c = p.split('_')
        pc = pc.lower() + c
        if pc in val_diff_pdb:
            val_pdb_unique.append(p)
    val_pdb_unique = np.array(val_pdb_unique)

    # no missing residues
    val_pdb_unique2 = set(val_pdb_unique) & set(val_pdb_no_missing_list)
    # varify no folds in train_cat
    val_pdb_unique3 = []
    for p in val_pdb_unique2:
        ind = 0
        pc, c = p.split('_')
        pc = pc.lower() + c
        for c in pdb_cat_dict[pc]:
            if c in train_cat:
                ind = 1
        if ind == 0:
            val_pdb_unique3.append(p)

    # val deep
    protein_sample = pd.read_csv(f'design/cullpdb_val_deep/sample.csv')
    pdb_selected = protein_sample['pdb'].values
    deep_cat, deep_cath, deep_no_cath = get_cath(pdb_selected)

    set(deep_cat) - set(train_cat)
    set(deep_cath) - set(train_cath)


