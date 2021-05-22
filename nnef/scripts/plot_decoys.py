import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy import stats
from tqdm import tqdm
import mdtraj as md


########################################################
def get_3drobot_native(data_flag):
    root_dir = '/home/hyang/bio/erf/data/decoys/3DRobot_set'
    pdb_list = pd.read_csv(f'{root_dir}/pdb_no_missing_residue.csv')['pdb'].values
    energy_native = []
    for pdb_id in pdb_list:
        df = pd.read_csv(f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_decoy_loss.csv')
        energy_native.append(df['loss'].values[0])
    energy_native = np.array(energy_native)
    print(energy_native, np.mean(energy_native), np.min(energy_native), np.max(energy_native), np.std(energy_native))


def plot_3drobot(data_flag):
    root_dir = '/home/hyang/bio/erf/data/decoys/3DRobot_set'
    # pdb_list = pd.read_csv('pdb_local_rot.txt')['pdb'].values
    # pdb_list = pd.read_csv('pdb_profile_diff.txt')['pdb'].values
    # pdb_list = pd.read_csv(f'{root_dir}/pdb_profile_diff_match.txt')['pdb'].values
    pdb_list = pd.read_csv(f'{root_dir}/pdb_no_missing_residue.csv')['pdb'].values

    # data_flag = 'exp005_v2'
    # data_flag = 'exp5'
    # data_flag = 'exp6'
    # data_flag = 'exp12'
    # data_flag = 'exp14'
    # data_flag = 'exp17'
    # data_flag = 'exp21'
    # data_flag = 'exp24'
    # data_flag = 'exp29'
    # data_flag = 'exp33'
    # data_flag = 'exp35'
    # data_flag = 'exp50'
    # data_flag = 'exp50_relax'
    # data_flag = 'exp49'
    # data_flag = 'exp49_relax'
    # data_flag = 'exp54'
    # data_flag = 'exp61'
    # data_flag = 'rosetta'
    # data_flag = 'rosetta_relax'
    # data_flag = 'rosetta_cen'

    # if not os.path.exists(f'{root_dir}/fig_3drobot_{data_flag}'):
    #     os.system(f'mkdir -p {root_dir}/fig_3drobot_{data_flag}')

    correct = 0
    rank = []
    for pdb_id in pdb_list:
        df = pd.read_csv(f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_decoy_loss.csv')
        decoy_name = df['NAME'].values
        assert(decoy_name[0] == 'native.pdb')

        ind = (df['loss'] != 999)
        loss = df['loss'][ind].values
        rmsd = df['RMSD'][ind].values

        if np.argmin(loss) == 0:
            correct += 1

        num = np.arange(loss.shape[0]) + 1
        rank_i = num[np.argsort(loss) == 0][0]
        rank.append(rank_i)

        if rank_i > 1:
            print(pdb_id, rmsd[np.argmin(loss)])

        fig = pl.figure()
        pl.plot(rmsd, loss, 'bo')
        pl.plot([rmsd[0]], [loss[0]], 'rs', markersize=12)
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        # pl.savefig(f'{root_dir}/fig_3drobot_{data_flag}/{pdb_id}_score.pdf')
        pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_score.pdf')
        pl.close(fig)

    rank = np.array(rank)
    print(rank)
    fig = pl.figure()
    pl.hist(rank, bins=np.arange(21)+0.5)
    pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/rank.pdf')
    pl.close(fig)


########################################################
def plot_casp11_loss():
    # pdb_list = pd.read_csv('pdb_list_new.txt')['pdb'].values
    pdb_list = pd.read_csv('pdb_no_need_copy_native.txt')['pdb'].values

    flist = pd.read_csv('list_casp11.txt')['fname'].values
    casp_dict = {x.split('#')[1][:5]: x.split('_')[0] for x in flist}

    df_tm = pd.read_csv('casp11_decoy.csv')
    tm_score_dict = {x: y for x, y in zip(df_tm['Target'], df_tm['Decoys'])}

    # data_flag = 'exp3_v2'
    # data_flag = 'exp5'
    # data_flag = 'exp7'
    # data_flag = 'exp13'
    # data_flag = 'exp15'
    # data_flag = 'exp21'
    # data_flag = 'exp24'
    # data_flag = 'exp29'
    # data_flag = 'exp33'
    # data_flag = 'exp35'
    data_flag = 'exp61'

    if not os.path.exists(f'fig_casp11_{data_flag}'):
        os.system(f'mkdir fig_casp11_{data_flag}')

    correct = 0
    rank = []
    tm_score = []
    for pdb_id in pdb_list:
        data_path = f'data_casp11_{data_flag}/{pdb_id}_decoy_loss.csv'
        if not os.path.exists(data_path):
            continue
        df = pd.read_csv(data_path)
        decoy_name = df['NAME'].values
        # ind = (df['loss'] != 999)
        # loss = df['loss'][ind].values
        tm_score.append(tm_score_dict[pdb_id])

        loss = df['loss'].values
        num = np.arange(loss.shape[0])

        i = (decoy_name == f'{pdb_id}.native.pdb')

        if num[i] == np.argmin(loss):
            # print(num.shape[0] - num[i])
            correct += 1

        rank.append(num[np.argsort(loss) == num[i]][0] + 1)

        fig = pl.figure()
        pl.plot(num, loss, 'bo')
        i = (decoy_name == f'{pdb_id}.Zhang-Server_model1.pdb')
        pl.plot([num[i]], [loss[i]], 'g^', markersize=12, label='zhang')
        i = (decoy_name == f'{pdb_id}.QUARK_model1.pdb')
        pl.plot([num[i]], [loss[i]], 'c*', markersize=12, label='quark')
        i = (decoy_name == f'{pdb_id}.native.pdb')
        pl.plot([num[i]], [loss[i]], 'rs', markersize=12, label='native')

        pdb_id = casp_dict[pdb_id]
        pl.title(f'{pdb_id}')
        pl.xlabel('num')
        pl.ylabel('energy score')
        pl.savefig(f'fig_casp11_{data_flag}/{pdb_id}_score.pdf')
        pl.close(fig)

    rank = np.array(rank)
    tm_score = np.array(tm_score)

    pl.figure()
    pl.hist(rank, bins=np.arange(21)+0.5)

    # pl.figure()
    # pl.plot(tm_score, rank, 'bo')

    a = (rank <= 5)
    b = (rank > 5)
    pl.figure()
    pl.hist(tm_score[a], bins=np.arange(9)*0.1+0.2, label='rank=1 or 2', histtype='stepfilled')
    pl.hist(tm_score[b], bins=np.arange(9)*0.1+0.2, label='rank>10', histtype='step')
    pl.xlabel('Best TM-score in decoys')
    pl.ylabel('Num')
    pl.legend(loc=2)


########################################################
def plot_casp11(data_flag):
    # plot RMSD vs. loss for CASP11
    root_dir = '/home/hyang/bio/erf/data/decoys/casp11'
    pdb_list = pd.read_csv(f'{root_dir}/casp11_rmsd/casp11_rmsd.txt')['pdb']
    flist = pd.read_csv(f'{root_dir}/list_casp11.txt')['fname'].values
    casp_dict = {x.split('#')[1][:5]: x.split('_')[0] for x in flist}

    # data_flag = 'exp3_v2'
    # data_flag = 'exp5'
    # data_flag = 'exp7'
    # data_flag = 'exp13'
    # data_flag = 'exp21'
    # data_flag = 'exp24'
    # data_flag = 'exp29'
    # data_flag = 'exp33'
    # data_flag = 'exp35'
    # data_flag = 'exp61'

    for pdb_id in pdb_list:
        data_path = f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_decoy_loss.csv'
        if not os.path.exists(data_path):
            continue
        df = pd.read_csv(data_path)
        decoy_name = df['NAME'].values
        # ind = (df['loss'] != 999)
        # loss = df['loss'][ind].values
        loss = df['loss'].values

        df2 = pd.read_csv(f'{root_dir}/casp11_rmsd/{pdb_id}_rmsd.csv')
        rmsd = df2['rmsd'].values
        assert(rmsd.shape[0] == loss.shape[0])

        fig = pl.figure()
        pl.plot(rmsd, loss, 'bo')
        i = (decoy_name == f'{pdb_id}.Zhang-Server_model1.pdb')
        pl.plot([rmsd[i]], [loss[i]], 'g^', markersize=12, label='zhang')
        i = (decoy_name == f'{pdb_id}.QUARK_model1.pdb')
        pl.plot([rmsd[i]], [loss[i]], 'c*', markersize=12, label='quark')
        i = (decoy_name == f'{pdb_id}.native.pdb')
        pl.plot([rmsd[i]], [loss[i]], 'rs', markersize=12, label='native')

        pdb_id = casp_dict[pdb_id]
        pl.title(f'{pdb_id}')
        a = max(12, rmsd.max())
        pl.xlim(-1, a)
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/rmsd_{pdb_id}_score.pdf')
        pl.close(fig)


########################################################
def prepare_casp13():
    # prepare casp13 decoys
    df = pd.read_csv('flist.txt')
    pdb_count = df['pdb'].value_counts()

    pdb_list = []
    for pdb, count in zip(pdb_count.index, pdb_count.values):
        if count > 1:
            pdb_list.append(pdb)
        else:
            pdb_list.append(pdb + '-D1')
    pdb_list = np.array(pdb_list)
    pdb_list.sort()
    df2 = pd.DataFrame({'pdb': pdb_list})
    df2.to_csv('pdb_list.txt', index=False)


def plot_casp13(data_flag, casp_id='casp13', casp_score_type='GDT_TS'):
    # plot results of casp13 / casp14 decoys
    root_dir = f'/home/hyang/bio/erf/data/decoys/{casp_id}'
    if casp_id == 'casp13':
        pdb_list = pd.read_csv(f'{root_dir}/pdb_list_domain.txt')['pdb'].values
        pdb_ids = [x.split('-')[0] for x in pdb_list]
    else:
        pdb_list = pd.read_csv(f'{root_dir}/pdb_list.txt')['pdb'].values
        pdb_ids = pdb_list
    # data_flag = 'exp61'

    # if not os.path.exists(f'fig_casp13_{data_flag}'):
    #     os.system(f'mkdir fig_casp13_{data_flag}')

    pearsonr_list = []
    pearsonp_list = []
    used_pdb_list = []
    casp_score_max = []
    casp_score_min = []
    rank_1 = 0
    for pdb_id, pdb_casp_name in zip(pdb_ids, pdb_list):
        data_path = f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_decoy_loss.csv'
        if not os.path.exists(data_path):
            continue
        df = pd.read_csv(data_path)
        decoy_name = df['pdb'].values
        # ind = (df['loss'] != 999)
        # loss = df['loss'][ind].values
        loss = df['loss'].values

        if not os.path.exists(f'{root_dir}/casp_score/{pdb_casp_name}.txt'):
            continue
        df2 = pd.read_csv(f'{root_dir}/casp_score/{pdb_casp_name}.txt', sep='\s+')

        casp_model = df2['Model']
        if (casp_id == 'casp13') & (pdb_casp_name.endswith('-D1')):
            casp_model = df2['Model'].apply(lambda x: x[:-3])

        if casp_score_type == 'GDT_TS':
            casp_score_data = df2['GDT_TS'].values
        elif casp_score_type == 'RMSD_CA':
            casp_score_data = df2['RMS_CA'].values
        else:
            raise ValueError('casp score type should be GDT_TS / RMSD_CA')
        casp_dict = {x: y for x, y in zip(casp_model, casp_score_data)}

        casp_score = []
        for x in decoy_name:
            try:
                casp_score.append(casp_dict[x])
            except KeyError:
                casp_score.append(-1)
        casp_score = np.array(casp_score)

        idx = (casp_score > 0) & (loss > 0)
        casp_score_good = casp_score[idx]
        loss_good = loss[idx]
        decoy_name_good = decoy_name[idx]
        # if np.argmax(casp_score_good) == np.argmin(loss_good):
        #     rank_1 += 1
        top5_idx = np.argpartition(loss_good, 5)[:5]
        best_gdt_idx = np.argmax(casp_score_good)
        if best_gdt_idx in top5_idx:
            print(best_gdt_idx, top5_idx)
            rank_1 += 1
        print(pdb_casp_name, decoy_name_good[best_gdt_idx], decoy_name_good[top5_idx])

        pearsonr = stats.pearsonr(casp_score_good, loss_good)
        pearsonr_list.append(pearsonr[0])
        pearsonp_list.append(pearsonr[1])
        used_pdb_list.append(pdb_id)
        casp_score_max.append(casp_score[idx].max())
        casp_score_min.append(casp_score[idx].min())
        df_i = pd.DataFrame({'pdb': decoy_name_good, casp_score_type: casp_score_good, 'energy': loss_good})
        df_i.to_csv(f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_casp_score_{casp_score_type}_energy.csv', index=False)

        fig = pl.figure()
        # pl.plot(100.0, loss[0], 'rs')
        pl.plot(casp_score[idx], loss[idx], 'bo')
        pl.title(f'{pdb_id}')
        # a = max(12, rmsd.max())
        # pl.xlim(-1, a)
        pl.xlabel(f'CASP {casp_score_type}')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_{casp_score_type}.pdf')
        pl.close(fig)

        fig = pl.figure()
        # pl.plot(100.0, loss[0], 'rs')
        pl.plot(casp_score_good, loss_good, 'bo')
        for i in range(loss_good.shape[0]):
            pl.text(casp_score_good[i], loss_good[i], decoy_name_good[i].split('S')[1][:-3], fontsize=6)
        pl.title(f'{pdb_id}')
        y_min = loss_good.min()
        y_max = loss_good.max()
        pl.ylim(y_min - (y_max - y_min) * 0.01, y_min + (y_max - y_min) * 0.15)
        # a = max(12, rmsd.max())
        pl.xlim(0, 100)
        pl.xlabel(f'CASP {casp_score_type}')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/{pdb_id}_{casp_score_type}_zoom.pdf')
        pl.close(fig)

    print(f'rank_1 = {rank_1}')
    df = pd.DataFrame({'pdb': used_pdb_list, 'pearsonr': pearsonr_list, 'pearsonp': pearsonp_list,
                       'casp_score_max': casp_score_max, 'casp_score_min': casp_score_min})
    df.to_csv(f'{root_dir}/decoy_loss_{data_flag}/pearsonr_{casp_score_type}.txt', index=False)
    fig = pl.figure()
    if casp_score_type == 'GDT_TS':
        pearsonr_bins = np.arange(11)*0.1-1
    elif casp_score_type == 'RMSD_CA':
        pearsonr_bins = np.arange(11)*0.1
    else:
        raise ValueError('casp score type should be gdt_ts / rmsd_ca')
    pl.hist(df['pearsonr'], bins=pearsonr_bins)
    pl.xlabel(r'Pearson $\rho$')
    pl.ylabel('N')
    pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/pearsonr_{casp_score_type}.pdf')
    pl.close(fig)

    # casp_score_max = df['casp_score_max'].values
    # fig = pl.figure()
    # idx = (casp_score_max >= 50)
    # pl.hist(df['pearsonr'][idx], bins=np.arange(11)*0.1-1)
    # pl.xlabel(r'Pearson $\rho$')
    # pl.ylabel('N')
    # pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/pearsonr_1.pdf')
    # pl.close(fig)
    # fig = pl.figure()
    # idx = (casp_score_max < 50)
    # pl.xlabel(r'Pearson $\rho$')
    # pl.ylabel('N')
    # pl.hist(df['pearsonr'][idx], bins=np.arange(11)*0.1-1)
    # pl.savefig(f'{root_dir}/decoy_loss_{data_flag}/pearsonr_2.pdf')
    # pl.close(fig)


########################################################
def plot_ru(decoy_set, decoy_loss_dir):
    # decoy_set = '4state_reduced'
    # decoy_set = 'lattice_ssfit'
    # decoy_set = 'lmds'
    # decoy_set = 'lmds_v2'

    root_dir = f'/home/hyang/bio/erf/data/decoys/rudecoy/multiple/{decoy_set}'
    # decoy_loss_dir = 'exp61'

    if not os.path.exists(f'{root_dir}/{decoy_loss_dir}'):
        os.system(f'mkdir -p {root_dir}/{decoy_loss_dir}')

    pdb_id_list = pd.read_csv(f'{root_dir}/list', header=None, names=['pdb'])['pdb'].values

    for pdb_id in pdb_id_list:
        df = pd.read_csv(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_decoy_loss.csv')
        pdb_list = df['pdb'].values
        loss = df['loss'].values
        rmsd = df['score'].values

        native_name = f'{pdb_id}.pdb'
        i_native = np.arange(pdb_list.shape[0])[(pdb_list == native_name)]
        i = np.argmin(loss)
        print(i_native, i, pdb_list[i])

        fig = pl.figure()
        pl.plot(rmsd, loss, 'bo')
        pl.plot([rmsd[i_native]], [loss[i_native]], 'rs', markersize=12)
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_score.pdf')
        pl.close(fig)


########################################################
def plot_md_trj(decoy_loss_dir):
    # plot the MD trajectory data
    root_dir = f'/home/hyang/bio/openmm/data'
    if not os.path.exists(f'{root_dir}/{decoy_loss_dir}'):
        os.system(f'mkdir -p {root_dir}/{decoy_loss_dir}')

    pdb_id_list = pd.read_csv(f'{root_dir}/list', header=None, names=['pdb'])['pdb'].values
    for pdb_id in pdb_id_list:
        df = pd.read_csv(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_decoy_loss.csv')
        loss = df['loss'].values
        rmsd = df['rmsd'].values
        pdb = df['pdb'].values
        # plot RMSD vs. Energy
        fig = pl.figure()
        idx = np.zeros(pdb.shape)
        for i in range(pdb.shape[0]):
            if pdb[i].startswith('T300'):
                idx[i] = 1
            elif pdb[i].startswith('T500'):
                idx[i] = 2
        pl.plot([rmsd[0]], [loss[0]], 'gs', markersize=12)
        pl.plot([rmsd[1]], [loss[1]], 'g^', markersize=12)
        pl.plot(rmsd[idx == 1], loss[idx == 1], 'g.', label='md_T300')
        pl.plot(rmsd[idx == 2], loss[idx == 2], 'c.', label='md_T500')
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_score.pdf')
        pl.close(fig)
        # plot RMSD vs. time & Energy vs. time
        fig = pl.figure()
        idx = np.zeros(pdb.shape)
        for i in range(pdb.shape[0]):
            if pdb[i].startswith('T300'):
                idx[i] = 1
            elif pdb[i].startswith('T500'):
                idx[i] = 2
        pl.subplot(211)
        pl.plot(rmsd[idx == 1], 'g', label='md_T300')
        pl.plot(rmsd[idx == 2], 'c', label='md_T500')
        pl.ylabel('RMSD')
        pl.legend()
        pl.title(f'{pdb_id}')
        pl.subplot(212)
        pl.plot(loss[idx == 1], 'g')
        pl.plot(loss[idx == 2], 'c')
        pl.ylabel('energy score')
        pl.xlabel('time-steps')
        pl.savefig(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_rmsd_energy_time.pdf')
        pl.close(fig)


def plot_md_trj2():
    # plot the MD trajectory data
    root_dir = '/home/hyang/bio/erf/data/decoys/md/cullpdb_val_deep/'

    pdb_id_list = pd.read_csv(f'{root_dir}/list', header=None, names=['pdb'])['pdb'].values
    # pdb_id_list = ['3KXT']
    for pdb_id in pdb_id_list:
        df1 = pd.read_csv(f'{root_dir}/{pdb_id}_T300_energy_rmsd.csv')
        loss1 = df1['energy'].values
        rmsd1 = df1['rmsd'].values
        df2 = pd.read_csv(f'{root_dir}/{pdb_id}_T500_energy_rmsd.csv')
        loss2 = df2['energy'].values
        rmsd2 = df2['rmsd'].values
        # plot RMSD vs. Energy
        fig = pl.figure()
        pl.plot([rmsd1[0]], [loss1[0]], 'gs', markersize=12)
        pl.plot(rmsd1, loss1, 'g.', label='T300')
        pl.plot(rmsd2, loss2, 'c.', label='T500')
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{pdb_id}_score.pdf')
        pl.close(fig)
        # plot RMSD vs. time & Energy vs. time
        fig = pl.figure()
        pl.subplot(211)
        pl.plot(rmsd1, 'g', label='md_T300')
        pl.plot(rmsd2, 'c', label='md_T500')
        pl.ylabel('RMSD')
        pl.legend()
        pl.title(f'{pdb_id}')
        pl.subplot(212)
        pl.plot(loss1, 'g')
        pl.plot(loss2, 'c')
        pl.ylabel('energy score')
        pl.xlabel('time-steps')
        pl.savefig(f'{root_dir}/{pdb_id}_rmsd_energy_time.pdf')
        pl.close(fig)


def plot_md_trj3():
    # plot the MD trajectory data
    root_dir = '/home/hyang/bio/erf/data/decoys/md/BPTI'

    df = pd.read_csv(f'{root_dir}/BPTI_energy_rmsd.csv')
    loss1 = df['energy'].values
    rmsd1 = df['rmsd'].values
    # plot RMSD vs. Energy
    fig = pl.figure()
    pl.plot(rmsd1, loss1, 'g.', markersize=0.01)
    pl.title('BPTI')
    pl.xlabel('RMSD')
    pl.ylabel('energy score')
    pl.savefig(f'{root_dir}/BPTI_score.jpg')
    pl.close(fig)
    # plot RMSD vs. time & Energy vs. time
    fig = pl.figure()
    pl.subplot(211)
    pl.plot(rmsd1, 'b.', markersize=0.01)
    pl.ylabel('RMSD')
    pl.title('BPTI')
    pl.subplot(212)
    pl.plot(loss1, 'g.', markersize=0.01)
    pl.ylabel('energy score')
    pl.xlabel('time-steps')
    pl.savefig(f'{root_dir}/BPTI_rmsd_energy_time.jpg')
    pl.close(fig)


def plot_bd_trj():
    # plot the mixed Langevin dynamics trajectory data
    root_dir = '/home/hyang/bio/erf/data/fold/exp205dynamics_val_deep501/'
    pdb_selected = pd.read_csv(f'/home/hyang/bio/erf/data/fold/cullpdb_val_deep/sample.csv')['pdb'].values
    pdb_selected = np.append(np.array(['1BPI_A']), pdb_selected)

    for pdb_id in pdb_selected:
        df1 = pd.read_csv(f'{root_dir}/{pdb_id}_energy.csv')
        loss1 = df1['sample_energy'].values
        rmsd1 = df1['sample_rmsd'].values
        # plot RMSD vs. Energy
        fig = pl.figure()
        pl.plot([rmsd1[0]], [loss1[0]], 'gs', markersize=12)
        pl.plot(rmsd1, loss1, 'go')
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{pdb_id}_score.pdf')
        pl.close(fig)

        # plot RMSD vs. time & Energy vs. time
        fig = pl.figure()
        pl.subplot(211)
        pl.plot(rmsd1, 'go')
        pl.ylabel('RMSD')
        pl.title(f'{pdb_id}')
        pl.subplot(212)
        pl.plot(loss1, 'bs')
        pl.ylabel('energy score')
        pl.xlabel('time-steps')
        pl.savefig(f'{root_dir}/{pdb_id}_rmsd_energy_time.pdf')
        pl.close(fig)


def plot_openmm2():
    root_dir = f'/home/hyang/bio/openmm/data'
    decoy_loss_dir = 'exp63_65'
    if not os.path.exists(f'{root_dir}/{decoy_loss_dir}'):
        os.system(f'mkdir -p {root_dir}/{decoy_loss_dir}')

    pdb_id_list = pd.read_csv(f'{root_dir}/list', header=None, names=['pdb'])['pdb'].values
    for pdb_id in pdb_id_list:
        fig = pl.figure()

        df = pd.read_csv(f'{root_dir}/exp61/{pdb_id}_decoy_loss.csv')
        loss = df['loss'].values * 15.0
        rmsd = df['rmsd'].values
        pl.plot(rmsd, loss, 'g.')
        pl.plot([rmsd[0]], [loss[0]], 'g^', markersize=12)

        df = pd.read_csv(f'{root_dir}/exp63/{pdb_id}_decoy_loss.csv')
        loss = df['loss'].values
        rmsd = df['rmsd'].values
        pl.plot(rmsd, loss, 'bo')
        pl.plot([rmsd[0]], [loss[0]], 'bs', markersize=12)

        df = pd.read_csv(f'{root_dir}/exp65/{pdb_id}_decoy_loss.csv')
        loss = df['loss'].values
        rmsd = df['rmsd'].values
        pl.plot(rmsd, loss, 'c.')
        pl.plot([rmsd[0]], [loss[0]], 'cs', markersize=12)

        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{decoy_loss_dir}/{pdb_id}_score.pdf')
        pl.close(fig)


########################################################
def plot_make_decoy_relax():
    # plot the relaxed decoys
    root_dir = f'/Users/Plover/study/bio/play/erf/data/fold/exp61anneal_val_deep_loop_relax'

    # if not os.path.exists(f'{root_dir}/{decoy_loss_dir}'):
    #     os.system(f'mkdir -p {root_dir}/{decoy_loss_dir}')

    pdb_id_list = pd.read_csv(f'{root_dir}/sample.csv')['pdb'].values

    for pdb_id in pdb_id_list:
        df = pd.read_csv(f'{root_dir}/{pdb_id}_energy.csv')
        # pdb_list = df['pdb'].values
        sample_energy = df['sample_energy'].values
        sample_energy_relaxed = df['sample_energy_relaxed'].values

        sample_rmsd = df['sample_rmsd'].values
        sample_relaxed_rmsd = df['sample_relaxed_rmsd'].values

        fig = pl.figure()
        pl.plot(sample_rmsd, sample_energy, 'g.', label='sample')
        pl.plot(sample_relaxed_rmsd, sample_energy_relaxed, 'b.', label='sample_relaxed')

        pl.plot([sample_rmsd[0]], [sample_energy[0]], 'ro', markersize=10)
        pl.plot([sample_relaxed_rmsd[0]], [sample_energy_relaxed[0]], 'rs', markersize=10)

        pl.legend()
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{pdb_id}_score.pdf')
        pl.close(fig)


def plot_make_decoy(data_flag):
    # plot the decoys without relax
    root_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}anneal_val_deep_loop'

    pdb_id_list = pd.read_csv(f'{root_dir}/../cullpdb_val_deep/sample.csv')['pdb'].values

    for pdb_id in pdb_id_list:
        if not os.path.exists(f'{root_dir}/{pdb_id}_energy.csv'):
            continue
        df = pd.read_csv(f'{root_dir}/{pdb_id}_energy.csv')
        sample_energy = df['sample_energy'].values
        sample_rmsd = df['sample_rmsd'].values

        fig = pl.figure()
        pl.plot(sample_rmsd, sample_energy, 'g.', label='sample')
        pl.plot([sample_rmsd[0]], [sample_energy[0]], 'ro', markersize=10)

        pl.legend()
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{pdb_id}_score.pdf')
        pl.close(fig)


def plot_openmm_bd_loop_decoy(data_flag, plot_frag=False, plot_loop=False, flag='', pdb_all=False):
    # plot decoys from openmm / loop / Brownian dynamics together
    md_root_dir = f'/home/hyang/bio/openmm/data/{data_flag}'
    loop_root_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}anneal_val_deep_loop'
    bd_root_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}dynamics_val_deep{flag}'
    # plot_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}_md_bd_loop{flag}_plot'
    plot_dir = bd_root_dir
    frag_root_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}frag_deep/'
    if not os.path.exists(f'{plot_dir}'):
        os.system(f'mkdir -p {plot_dir}')

    pdb_id_list = ['3KXT']
    if pdb_all:
        pdb_id_list = pd.read_csv(f'{md_root_dir}/../list', header=None, names=['pdb'])['pdb'].values
    for pdb_id in pdb_id_list:
        fig = pl.figure()
        df = pd.read_csv(f'{md_root_dir}/{pdb_id}_decoy_loss.csv')
        pdb = df['pdb'].values
        idx = np.zeros(pdb.shape)
        for i in range(pdb.shape[0]):
            if pdb[i].startswith('T300'):
                idx[i] = 1
            elif pdb[i].startswith('T500'):
                idx[i] = 2
        loss = df['loss'].values
        rmsd = df['rmsd'].values
        pl.plot([rmsd[0]], [loss[0]], 'gs', markersize=12)
        pl.plot([rmsd[1]], [loss[1]], 'g^', markersize=9)
        pl.plot(rmsd[idx == 1], loss[idx == 1], 'g.', label='md_T300')
        pl.plot(rmsd[idx == 2], loss[idx == 2], 'c.', label='md_T500')

        if plot_loop:
            df = pd.read_csv(f'{loop_root_dir}/{pdb_id}_A_energy.csv')
            sample_energy = df['sample_energy'].values
            sample_rmsd = df['sample_rmsd'].values
            pl.plot(sample_rmsd, sample_energy, 'b.', label='loop')
            pl.plot([sample_rmsd[0]], [sample_energy[0]], 'bo', markersize=10)

        df = pd.read_csv(f'{bd_root_dir}/{pdb_id}_A_energy.csv')
        sample_energy = df['sample_energy'].values
        sample_rmsd = df['sample_rmsd'].values
        pl.plot(sample_rmsd, sample_energy, 'r.', label='Brownian')
        pl.plot([sample_rmsd[0]], [sample_energy[0]], 'ro', markersize=10)

        if plot_frag:
            df = pd.read_csv(f'{frag_root_dir}/{pdb_id}_A_energy_all.csv')
            sample_energy = df['sample_energy'].values
            sample_rmsd = df['sample_rmsd'].values
            pl.plot(sample_rmsd, sample_energy, 'm.', label='Fragment')
            pl.plot([sample_rmsd[0]], [sample_energy[0]], 'ro', markersize=10)

        pl.legend()
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{plot_dir}/{pdb_id}_score.pdf')
        pl.close(fig)


def plot_openmm_loop_decoy2():
    # plot two experiments together
    md1_root_dir = f'/Users/Plover/study/bio/play/erf/data/fold/openmm/exp65'
    loop1_root_dir = f'/Users/Plover/study/bio/play/erf/data/fold/exp65anneal_val_deep_loop'

    md2_root_dir = f'/Users/Plover/study/bio/play/erf/data/fold/openmm/exp63'
    loop2_root_dir = f'/Users/Plover/study/bio/play/erf/data/fold/exp63anneal_val_deep_loop'

    plot_dir = f'/Users/Plover/study/bio/play/erf/data/fold/exp63_65_md_loop_plot'

    pdb_id_list = pd.read_csv(f'{md1_root_dir}/list', header=None, names=['pdb'])['pdb'].values
    for pdb_id in pdb_id_list:
        fig = pl.figure()

        df = pd.read_csv(f'{md1_root_dir}/{pdb_id}_decoy_loss.csv')
        pdb = df['pdb'].values
        idx = np.zeros(pdb.shape)
        for i in range(pdb.shape[0]):
            if pdb[i].startswith('T300'):
                idx[i] = 1
            elif pdb[i].startswith('T500'):
                idx[i] = 2
        loss = df['loss'].values
        rmsd = df['rmsd'].values
        pl.plot([rmsd[0]], [loss[0]], 'gs', markersize=12)
        pl.plot([rmsd[1]], [loss[1]], 'g^', markersize=12)
        pl.plot(rmsd[idx == 1], loss[idx == 1], 'g.', label='md1_T300')
        pl.plot(rmsd[idx == 2], loss[idx == 2], 'c.', label='md1_T500')

        df = pd.read_csv(f'{md2_root_dir}/{pdb_id}_decoy_loss.csv')
        pdb = df['pdb'].values
        idx = np.zeros(pdb.shape)
        for i in range(pdb.shape[0]):
            if pdb[i].startswith('T300'):
                idx[i] = 1
            elif pdb[i].startswith('T500'):
                idx[i] = 2
        loss = df['loss'].values
        rmsd = df['rmsd'].values
        pl.plot([rmsd[0]], [loss[0]], 'rs', markersize=12)
        pl.plot([rmsd[1]], [loss[1]], 'r^', markersize=12)
        pl.plot(rmsd[idx == 1], loss[idx == 1], 'r.', label='md2_T300')
        pl.plot(rmsd[idx == 2], loss[idx == 2], 'm.', label='md2_T500')

        df = pd.read_csv(f'{loop1_root_dir}/{pdb_id}_A_energy.csv')
        sample_energy = df['sample_energy'].values
        sample_rmsd = df['sample_rmsd'].values

        pl.plot(sample_rmsd, sample_energy, 'b.', label='loop1')
        pl.plot([sample_rmsd[0]], [sample_energy[0]], 'bo', markersize=10)

        df = pd.read_csv(f'{loop2_root_dir}/{pdb_id}_A_energy.csv')
        sample_energy = df['sample_energy'].values
        sample_rmsd = df['sample_rmsd'].values

        pl.plot(sample_rmsd, sample_energy, 'k.', label='loop2')
        pl.plot([sample_rmsd[0]], [sample_energy[0]], 'ko', markersize=10)

        pl.legend()
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{plot_dir}/{pdb_id}_score.pdf')
        pl.close(fig)


def plot_fold_decoy():
    # plot folding decoys from annealing
    root_dir = f'/home/hyang/bio/erf/data/fold/exp63anneal_val_deep_fd3'

    pdb_id = '3SOL_A'
    fig = pl.figure()

    for i in range(2):
        if not os.path.exists(f'{root_dir}/{pdb_id}_energy_{i}.csv'):
            continue
        df = pd.read_csv(f'{root_dir}/{pdb_id}_energy_{i}.csv')
        sample_energy = df['sample_energy'].values
        sample_rmsd = df['sample_rmsd'].values

        pl.plot(sample_rmsd, sample_energy, 'g.', markersize=3)
        pl.plot([sample_rmsd[0]], [sample_energy[0]], 'ro', markersize=10)

    pl.title(f'{pdb_id}')
    pl.xlabel('RMSD')
    pl.ylabel('energy score')
    pl.savefig(f'{root_dir}/{pdb_id}_score.png')
    pl.close(fig)


def plot_decoy_seq(data_flag):
    # plot decoy seq
    root_dir = f'/home/hyang/bio/erf/data/decoys/decoys_seq/{data_flag}'
    # decoy_flag_list = ['random', 'shuffle', 'type2', 'type9', 'type2LD']
    decoy_flag_list = ['random', 'shuffle', 'type9']

    df = pd.read_csv(f'{root_dir}/hhsuite_CB_cullpdb_val_no_missing_residue_sample_loss.csv')
    pdb_list = df['pdb'].values
    native_energy = {x: y for x, y in zip(df['pdb'], df['loss_native'])}

    pl.figure()
    for decoy_flag in decoy_flag_list:
        rank = []
        delta_loss = np.array([])
        for pdb in pdb_list:
            loss = pd.read_csv(f'{root_dir}/{pdb}_{decoy_flag}_loss.csv')['loss'].values
            loss_native = native_energy[pdb]
            rank.append(loss[loss < loss_native].shape[0] + 1)
            if loss_native > loss.min():
                print(decoy_flag, pdb, loss_native, loss.min())
            delta_loss = np.append(delta_loss, loss - loss_native)
        rank = np.array(rank)
        # plot the energy gap
        pl.hist(delta_loss, linewidth=2, label=decoy_flag, histtype='step')
    pl.xlabel('E(decoy) - E(native)', fontsize=14)
    pl.ylabel('Num', fontsize=14)
    pl.legend()
    pl.savefig(f'{root_dir}/delta_loss_{data_flag}.pdf')
    pl.close()

    # plot the tpe2LD result separately
    pl.figure()
    pl.hist(delta_loss, label='type2LD', histtype='step')
    pl.xlabel('E(decoy) - E(native)')
    pl.ylabel('Num')
    pl.legend()
    pl.savefig(f'{root_dir}/delta_loss_type2LD.pdf')
    pl.close()


def plot_polyAA(data_flag):
    # plot polyAA decoys
    root_dir = f'/home/hyang/bio/erf/data/decoys/decoys_seq/{data_flag}'
    amino_acids = pd.read_csv(f'{root_dir}/../../../amino_acids.csv')

    decoy_flag = 'polyAA'
    df = pd.read_csv(f'{root_dir}/hhsuite_CB_cullpdb_val_no_missing_residue_sample_loss.csv')
    pdb_list = df['pdb'].values

    loss_all = np.zeros((len(pdb_list), 20))
    for i, pdb in enumerate(pdb_list):
        loss = pd.read_csv(f'{root_dir}/{pdb}_{decoy_flag}_loss.csv')['loss'].values
        loss_all[i] = loss

    for i in range(20):
        pl.figure()
        pl.plot(df['loss_native'], loss_all[:, i], 'bo')
        pl.title('poly-' + amino_acids.AA3C.values[i])
        pl.xlabel('E(native)')
        pl.ylabel('E(decoy)')
        pl.plot(df['loss_native'], df['loss_native'], ls='--', color='g')
        pl.savefig(f'{root_dir}/{data_flag}_{amino_acids.AA3C.values[i]}.pdf')
        pl.close()

    # plot all 20 polyAA figures in one big figure
    fig = pl.figure(figsize=(10, 8))
    # ax0 = fig.add_subplot(111)  # The big subplot
    # ax0.set(xticklabels=[], yticklabels=[])  # remove the tick labels
    # ax0.tick_params(left=False, bottom=False)  # remove the ticks
    for i in range(20):
        ax = fig.add_subplot(5, 4, i+1)
        ax.plot(df['loss_native'], loss_all[:, i], 'bo', markersize=6)
        pl.plot(df['loss_native'], df['loss_native'], ls='-', color='g')
        ax.set(xticklabels=[], yticklabels=[])  # remove the tick labels
        pl.title('poly-' + amino_acids.AA3C.values[i])
    # ax0.set_xlabel('E(native)', fontsize=14)
    # ax0.set_ylabel('E(decoy)', fontsize=14)
    fig.text(0.5, 0.05, 'E(native)', fontsize=14, ha='center')
    fig.text(0.05, 0.5, 'E(decoy)', fontsize=14, va='center', rotation='vertical')
    pl.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.95, wspace=0.3, hspace=0.3)
    pl.savefig(f'{root_dir}/{data_flag}_poly_AA_all.pdf')
    pl.close()


def plot_bd_md_rmsf(data_flag, flag='', pdb_all=False):
    bd_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}dynamics_val_deep{flag}'
    md_dir = '/home/hyang/bio/openmm/data/'
    pdb_list = ['3KXT']
    if pdb_all:
        pdb_list = pd.read_csv(f'{md_dir}/list', header=None, names=['pdb'])['pdb'].values
    for pdb in tqdm(pdb_list):
        trj_md = md.load(f'{md_dir}/{pdb}/production_T300.dcd',
                         top=f'{md_dir}/{pdb}/production_T300.pdb')
        trj_md2 = md.load(f'{md_dir}/{pdb}/production2_T300.dcd',
                          top=f'{md_dir}/{pdb}/production_T300.pdb')

        trj_bd = md.load(f'{bd_dir}/{pdb}_A_sample.pdb')
        a = np.arange(trj_bd.xyz.shape[0])
        idx = (a == 0) | (a > len(a)/2)  # frame 0 is used for coordinates alignments
        trj_bd_eq = trj_bd[idx]

        rmsd_bd = md.rmsd(trj_bd, trj_bd, frame=0)
        rmsf_bd = md.rmsf(trj_bd, trj_bd, frame=0)
        rmsf_bd_eq = md.rmsf(trj_bd_eq, trj_bd_eq, frame=0)

        def get_rmsd_rmsf_md(trj_md):
            trj_md_pro = trj_md.remove_solvent()
            top = trj_md_pro.topology
            # use CA to calculate RMSD
            ca_idx = top.select("name == 'CA'")
            xyz = trj_md_pro.xyz
            xyz_ca = xyz[:, ca_idx]
            # use CB to calculate RMSD
            ca_gly = top.select('(name == CA) and (resname == GLY)')
            cb = top.select('name == CB')
            beads = np.append(ca_gly, cb)
            beads = np.sort(beads)
            xyz_cb = xyz[:, beads]
            b = np.arange(xyz_cb.shape[0])
            idx = (b == 0) | (b > len(b)/2)  # frame 0 is used for coordinates alignments
            xyz_cb_eq = xyz_cb[idx]

            t_ca = md.Trajectory(xyz=xyz_ca, topology=None)
            rmsd_md_ca = md.rmsd(t_ca, t_ca, frame=0)
            rmsf_md_ca = md.rmsf(t_ca, t_ca, frame=0)
            t_cb = md.Trajectory(xyz=xyz_cb, topology=None)
            rmsd_md_cb = md.rmsd(t_cb, t_cb, frame=0)
            rmsf_md_cb = md.rmsf(t_cb, t_cb, frame=0)
            t_cb_eq = md.Trajectory(xyz=xyz_cb_eq, topology=None)
            # rmsd2_cb_eq = md.rmsd(t_cb_eq, t_cb_eq, frame=0)
            rmsf_md_cb_eq = md.rmsf(t_cb_eq, t_cb_eq, frame=0)
            return rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq

        rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq = get_rmsd_rmsf_md(trj_md)
        rmsd_md2_ca, rmsd_md2_cb, rmsf_md2_ca, rmsf_md2_cb, rmsf_md2_cb_eq = get_rmsd_rmsf_md(trj_md2)

        fig = pl.figure()
        pl.plot(rmsd_bd*10, label='erf')
        pl.plot(rmsd_md_ca*10, label='MD_CA')
        pl.plot(rmsd_md_cb*10, label='MD_CB')
        pl.plot(rmsd_md2_ca*10, label='MD2_CA')
        pl.plot(rmsd_md2_cb*10, label='MD2_CB')
        pl.legend()
        pl.title('RMSD')
        pl.xlabel('steps')
        pl.ylabel(r'RMSD [$\AA$]')
        pl.savefig(f'{bd_dir}/{pdb}_A_rmsd.pdf')
        pl.close(fig)

        fig = pl.figure()
        pl.plot(rmsf_bd*10, label='erf')
        pl.plot(rmsf_bd_eq*10, label='erf_eq')
        pl.plot(rmsf_md_ca*10, label='MD_CA')
        pl.plot(rmsf_md_cb*10, label='MD_CB')
        pl.plot(rmsf_md_cb_eq*10, label='MD_CB_eq')
        pl.plot(rmsf_md2_ca*10, label='MD2_CA')
        pl.plot(rmsf_md2_cb*10, label='MD2_CB')
        pl.plot(rmsf_md2_cb_eq*10, label='MD2_CB_eq')
        pl.legend()
        pl.title('RMSF')
        pl.xlabel('Resdidue Number')
        pl.ylabel(r'RMSF [$\AA$]')
        pl.savefig(f'{bd_dir}/{pdb}_A_rmsf.pdf')
        pl.close(fig)

        fig = pl.figure()
        pl.plot(rmsf_bd*10, label='erf')
        pl.plot(rmsf_bd_eq*10, label='erf_eq')
        pl.plot(rmsf_md2_cb*10, label='MD2_CB')
        pl.plot(rmsf_md2_cb_eq*10, label='MD2_CB_eq')
        pl.legend()
        pl.title('RMSF')
        pl.xlabel('Resdidue Number')
        pl.ylabel(r'RMSF [$\AA$]')
        pl.savefig(f'{bd_dir}/{pdb}_A_rmsf_simple.pdf')
        pl.close(fig)


def plot_bd_md_rmsf2(data_flag, flag_list, pdb_all=False, debug=False, b_factor=False):
    md_dir = '/home/hyang/bio/openmm/data/'
    pdb_list = ['3KXT']
    pdb_list = ['1BPI']
    if pdb_all:
        pdb_list = pd.read_csv(f'{md_dir}/list', header=None, names=['pdb'])['pdb'].values

    def get_rmsd_rmsf_md(trj_md):
        trj_md_pro = trj_md.remove_solvent()
        top = trj_md_pro.topology
        # use CA to calculate RMSD
        ca_idx = top.select("name == 'CA'")
        xyz = trj_md_pro.xyz
        xyz_ca = xyz[:, ca_idx]
        # use CB to calculate RMSD
        ca_gly = top.select('(name == CA) and (resname == GLY)')
        cb = top.select('name == CB')
        beads = np.append(ca_gly, cb)
        beads = np.sort(beads)
        xyz_cb = xyz[:, beads]
        b = np.arange(xyz_cb.shape[0])
        idx = (b == 0) | (b > len(b) / 2)  # frame 0 is used for coordinates alignments
        xyz_cb_eq = xyz_cb[idx]

        t_ca = md.Trajectory(xyz=xyz_ca, topology=None)
        rmsd_md_ca = md.rmsd(t_ca, t_ca, frame=0)
        rmsf_md_ca = md.rmsf(t_ca, t_ca, frame=0)
        t_cb = md.Trajectory(xyz=xyz_cb, topology=None)
        rmsd_md_cb = md.rmsd(t_cb, t_cb, frame=0)
        rmsf_md_cb = md.rmsf(t_cb, t_cb, frame=0)
        t_cb_eq = md.Trajectory(xyz=xyz_cb_eq, topology=None)
        # rmsd2_cb_eq = md.rmsd(t_cb_eq, t_cb_eq, frame=0)
        rmsf_md_cb_eq = md.rmsf(t_cb_eq, t_cb_eq, frame=0)
        return rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq

    for pdb in tqdm(pdb_list):
        if debug:
            trj_md = md.load(f'{md_dir}/{pdb}/production_T300.dcd',
                             top=f'{md_dir}/{pdb}/production_T300.pdb')
            rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq = get_rmsd_rmsf_md(trj_md)

        trj_md2 = md.load(f'{md_dir}/{pdb}/production2_T300.dcd',
                          top=f'{md_dir}/{pdb}/production_T300.pdb')
        rmsd_md2_ca, rmsd_md2_cb, rmsf_md2_ca, rmsf_md2_cb, rmsf_md2_cb_eq = get_rmsd_rmsf_md(trj_md2)

        rmsf_delta = []
        rmsf_eq_delta = []
        rmsf_delta_ref = []
        rmsf_eq_delta_ref = []
        for flag in tqdm(flag_list):
            bd_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}dynamics_val_deep{flag}'

            trj_bd = md.load(f'{bd_dir}/{pdb}_A_sample.pdb')
            a = np.arange(trj_bd.xyz.shape[0])
            idx = (a == 0) | (a > len(a) / 2)  # frame 0 is used for coordinates alignments
            trj_bd_eq = trj_bd[idx]

            rmsd_bd = md.rmsd(trj_bd, trj_bd, frame=0)
            rmsf_bd = md.rmsf(trj_bd, trj_bd, frame=0)
            rmsf_bd_eq = md.rmsf(trj_bd_eq, trj_bd_eq, frame=0)
            # save RMSF
            df_rmsf = pd.DataFrame({'RMSF_bd': rmsf_bd, 'RMSF_bd_eq': rmsf_bd_eq,
                                    'RMSF_CB': rmsf_md2_cb, 'RMSF_CB_eq': rmsf_md2_cb_eq})
            df_rmsf.to_csv(f'{bd_dir}/{pdb}_{flag}_rmsf.csv', index=False)

            # compute RMSF differences between Browninan dynamics and MD
            # rmsf_delta.append(np.sqrt(np.mean((rmsf_bd - rmsf_md2_cb)**2)))   # root mean square of delta
            # rmsf_eq_delta.append(np.sqrt(np.mean((rmsf_bd_eq - rmsf_md2_cb_eq)**2)))
            rmsf_delta.append(np.mean(np.abs(rmsf_bd - rmsf_md2_cb)))
            rmsf_eq_delta.append(np.mean(np.abs(rmsf_bd_eq - rmsf_md2_cb_eq)))
            rmsf_delta_ref.append(np.mean(np.abs(rmsf_md2_cb - np.mean(rmsf_md2_cb))))
            rmsf_eq_delta_ref.append(np.mean(np.abs(rmsf_md2_cb_eq - np.mean(rmsf_md2_cb_eq))))

            fig = pl.figure()
            pl.plot(rmsd_bd*10, label='erf')
            if debug:
                pl.plot(rmsd_md_ca*10, label='MD_CA')
                pl.plot(rmsd_md_cb*10, label='MD_CB')
                pl.plot(rmsd_md2_ca*10, label='MD2_CA')
            pl.plot(rmsd_md2_cb*10, label='MD2_CB')
            pl.legend()
            pl.title('RMSD')
            pl.xlabel('steps')
            pl.ylabel(r'RMSD [$\AA$]')
            pl.savefig(f'{bd_dir}/{pdb}_A_rmsd.pdf')
            pl.close(fig)

            if b_factor:
                df_b_factor = pd.read_csv(f'/home/hyang/bio/erf/data/fold/cullpdb_val_deep/{pdb}_A_bead.csv')
                b_ca = df_b_factor['bca'].values
                b_cb = df_b_factor['bcb'].values
                # b_factor = (8 / 3 * pi^2) * (RMSF)^2
                rmsf_b_ca = np.sqrt(b_ca / (8/3*np.pi**2))
                rmsf_b_cb = np.sqrt(b_cb / (8/3*np.pi**2))

            fig = pl.figure()
            pl.plot(rmsf_bd*10, label='erf')
            pl.plot(rmsf_bd_eq*10, label='erf_eq')
            if debug:
                pl.plot(rmsf_md_ca*10, label='MD_CA')
                pl.plot(rmsf_md_cb*10, label='MD_CB')
                pl.plot(rmsf_md_cb_eq*10, label='MD_CB_eq')
                pl.plot(rmsf_md2_ca*10, label='MD2_CA')
            pl.plot(rmsf_md2_cb*10, label='MD2_CB')
            pl.plot(rmsf_md2_cb_eq*10, label='MD2_CB_eq')
            if b_factor:
                pl.plot(rmsf_b_ca, label='b_factor_CA')
                pl.plot(rmsf_b_cb, label='b_factor_CB')

            pl.legend()
            pl.title('RMSF')
            pl.xlabel('Resdidue Number')
            pl.ylabel(r'RMSF [$\AA$]')
            pl.savefig(f'{bd_dir}/{pdb}_A_rmsf.pdf')
            pl.close(fig)

            fig = pl.figure()
            pl.plot(rmsf_bd*10, label='erf')
            pl.plot(rmsf_md2_cb*10, label='MD2_CB')
            if b_factor:
                pl.plot(rmsf_b_cb, label='b_factor_CB')
            pl.legend()
            pl.title('RMSF')
            pl.xlabel('Resdidue Number')
            pl.ylabel(r'RMSF [$\AA$]')
            pl.savefig(f'{bd_dir}/{pdb}_A_rmsf_simple.pdf')
            pl.close(fig)

            # plot RMSD vs. time & Energy vs. time
            fig = pl.figure()
            df_bd_energy = pd.read_csv(f'{bd_dir}/{pdb}_A_energy.csv')
            sample_energy = df_bd_energy['sample_energy'].values
            sample_rmsd = df_bd_energy['sample_rmsd'].values
            pl.subplot(211)
            pl.plot(sample_rmsd*10, 'b')
            pl.ylabel('RMSD')
            pl.subplot(212)
            pl.plot(sample_energy, 'g')
            pl.ylabel('energy score')
            pl.xlabel('time-steps')
            pl.savefig(f'{bd_dir}/{pdb}_bd_rmsd_energy_time.pdf')
            pl.close(fig)

        df = pd.DataFrame({'flag': flag_list, 'RMSF_delta': rmsf_delta,
                           'RMSF_eq_delta': rmsf_eq_delta,
                           'RMSF_delta_ref': rmsf_delta_ref,
                           'RMSF_eq_delta_ref': rmsf_eq_delta_ref})
        df.to_csv(f'/home/hyang/bio/erf/data/fold/{data_flag}_bd_grid/{pdb}_rmsf_grid.csv')


def plot_bd_md_rmsf_BPTI(data_flag, flag_list):
    md_dir = '/home/hyang/bio/openmm/data/'
    rmsf_deshaw = pd.read_csv('/home/hyang/bio/erf/data/decoys/md/BPTI/BPTI_rmsf.csv')

    def get_rmsd_rmsf_md(trj_md):
        trj_md_pro = trj_md.remove_solvent()
        top = trj_md_pro.topology
        # use CA to calculate RMSD
        ca_idx = top.select("name == 'CA'")
        xyz = trj_md_pro.xyz
        xyz_ca = xyz[:, ca_idx]
        # use CB to calculate RMSD
        ca_gly = top.select('(name == CA) and (resname == GLY)')
        cb = top.select('name == CB')
        beads = np.append(ca_gly, cb)
        beads = np.sort(beads)
        xyz_cb = xyz[:, beads]
        b = np.arange(xyz_cb.shape[0])
        idx = (b == 0) | (b > len(b) / 2)  # frame 0 is used for coordinates alignments
        xyz_cb_eq = xyz_cb[idx]

        t_ca = md.Trajectory(xyz=xyz_ca, topology=None)
        rmsd_md_ca = md.rmsd(t_ca, t_ca, frame=0)
        rmsf_md_ca = md.rmsf(t_ca, t_ca, frame=0)
        t_cb = md.Trajectory(xyz=xyz_cb, topology=None)
        rmsd_md_cb = md.rmsd(t_cb, t_cb, frame=0)
        rmsf_md_cb = md.rmsf(t_cb, t_cb, frame=0)
        t_cb_eq = md.Trajectory(xyz=xyz_cb_eq, topology=None)
        # rmsd2_cb_eq = md.rmsd(t_cb_eq, t_cb_eq, frame=0)
        rmsf_md_cb_eq = md.rmsf(t_cb_eq, t_cb_eq, frame=0)
        return rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq

    pdb = '1BPI'
    trj_md2 = md.load(f'{md_dir}/{pdb}/production2_T300.dcd',
                      top=f'{md_dir}/{pdb}/production2_T300.pdb')
    rmsd_md2_ca, rmsd_md2_cb, rmsf_md2_ca, rmsf_md2_cb, rmsf_md2_cb_eq = get_rmsd_rmsf_md(trj_md2)

    rmsf_delta = []
    rmsf_eq_delta = []
    rmsf_delta_ref = []
    rmsf_eq_delta_ref = []
    for flag in tqdm(flag_list):
        bd_dir = f'/home/hyang/bio/erf/data/fold/{data_flag}dynamics_val_deep{flag}'
        trj_bd = md.load(f'{bd_dir}/{pdb}_A_sample.pdb')
        a = np.arange(trj_bd.xyz.shape[0])
        idx = (a == 0) | (a > len(a) / 2)  # frame 0 is used for coordinates alignments
        trj_bd_eq = trj_bd[idx]

        rmsd_bd = md.rmsd(trj_bd, trj_bd, frame=0)
        rmsf_bd = md.rmsf(trj_bd, trj_bd, frame=0)
        rmsf_bd_eq = md.rmsf(trj_bd_eq, trj_bd_eq, frame=0)
        # save RMSF
        df_rmsf = pd.DataFrame({'RMSF_bd': rmsf_bd, 'RMSF_bd_eq': rmsf_bd_eq,
                                'RMSF_CB': rmsf_md2_cb, 'RMSF_CB_eq': rmsf_md2_cb_eq})
        df_rmsf.to_csv(f'{bd_dir}/{pdb}_{flag}_rmsf.csv', index=False)
        # compute RMSF differences between Browninan dynamics and MD
        # rmsf_delta.append(np.sqrt(np.mean((rmsf_bd - rmsf_md2_cb)**2)))   # root mean square of delta
        # rmsf_eq_delta.append(np.sqrt(np.mean((rmsf_bd_eq - rmsf_md2_cb_eq)**2)))
        rmsf_delta.append(np.mean(np.abs(rmsf_bd - rmsf_md2_cb)))
        rmsf_eq_delta.append(np.mean(np.abs(rmsf_bd_eq - rmsf_md2_cb_eq)))
        rmsf_delta_ref.append(np.mean(np.abs(rmsf_md2_cb - np.mean(rmsf_md2_cb))))
        rmsf_eq_delta_ref.append(np.mean(np.abs(rmsf_md2_cb_eq - np.mean(rmsf_md2_cb_eq))))

        fig = pl.figure()
        pl.plot(rmsd_bd*10, label='erf')
        pl.plot(rmsd_md2_cb*10, label='MD2_CB')
        pl.legend()
        pl.title('RMSD')
        pl.xlabel('steps')
        pl.ylabel(r'RMSD [$\AA$]')
        pl.savefig(f'{bd_dir}/{pdb}_A_rmsd.pdf')
        pl.close(fig)

        fig = pl.figure()
        pl.plot(rmsf_bd*10, label='erf')
        pl.plot(rmsf_bd_eq*10, label='erf_eq')
        pl.plot(rmsf_md2_cb*10, label='MD2_CB')
        pl.plot(rmsf_md2_cb_eq*10, label='MD2_CB_eq')
        pl.plot(rmsf_deshaw, label='DEShaw')
        pl.legend()
        pl.title('RMSF')
        pl.xlabel('Resdidue Number')
        pl.ylabel(r'RMSF [$\AA$]')
        pl.savefig(f'{bd_dir}/{pdb}_A_rmsf.pdf')
        pl.close(fig)

        fig = pl.figure()
        pl.plot(rmsf_bd*10, label='erf')
        pl.plot(rmsf_md2_cb*10, label='MD2_CB')
        pl.plot(rmsf_deshaw, label='DEShaw')
        pl.legend()
        pl.title('RMSF')
        pl.xlabel('Resdidue Number')
        pl.ylabel(r'RMSF [$\AA$]')
        pl.savefig(f'{bd_dir}/{pdb}_A_rmsf_simple.pdf')
        pl.close(fig)

    df = pd.DataFrame({'flag': flag_list, 'RMSF_delta': rmsf_delta,
                       'RMSF_eq_delta': rmsf_eq_delta,
                       'RMSF_delta_ref': rmsf_delta_ref,
                       'RMSF_eq_delta_ref': rmsf_eq_delta_ref})
    df.to_csv(f'/home/hyang/bio/erf/data/fold/{data_flag}_bd_grid/{pdb}_rmsf_grid.csv')


def plot_3SNY_long_trj():
    def get_rmsd_rmsf_md(trj_md):
        trj_md_pro = trj_md.remove_solvent()
        top = trj_md_pro.topology
        # use CA to calculate RMSD
        ca_idx = top.select("name == 'CA'")
        xyz = trj_md_pro.xyz
        xyz_ca = xyz[:, ca_idx]
        # use CB to calculate RMSD
        ca_gly = top.select('(name == CA) and (resname == GLY)')
        cb = top.select('name == CB')
        beads = np.append(ca_gly, cb)
        beads = np.sort(beads)
        xyz_cb = xyz[:, beads]
        b = np.arange(xyz_cb.shape[0])
        idx = (b == 0) | (b > len(b) / 2)  # frame 0 is used for coordinates alignments
        xyz_cb_eq = xyz_cb[idx]

        t_ca = md.Trajectory(xyz=xyz_ca, topology=None)
        rmsd_md_ca = md.rmsd(t_ca, t_ca, frame=0)
        rmsf_md_ca = md.rmsf(t_ca, t_ca, frame=0)
        t_cb = md.Trajectory(xyz=xyz_cb, topology=None)
        rmsd_md_cb = md.rmsd(t_cb, t_cb, frame=0)
        rmsf_md_cb = md.rmsf(t_cb, t_cb, frame=0)
        t_cb_eq = md.Trajectory(xyz=xyz_cb_eq, topology=None)
        # rmsd2_cb_eq = md.rmsd(t_cb_eq, t_cb_eq, frame=0)
        rmsf_md_cb_eq = md.rmsf(t_cb_eq, t_cb_eq, frame=0)
        return rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq

    trj_md2 = md.load('production3_T300.dcd', top='production2_T300.pdb')
    rmsd_md2_ca, rmsd_md2_cb, rmsf_md2_ca, rmsf_md2_cb, rmsf_md2_cb_eq = get_rmsd_rmsf_md(trj_md2)
    trj_md1 = md.load('production2_T300.dcd', top='production2_T300.pdb')
    rmsd_md_ca, rmsd_md_cb, rmsf_md_ca, rmsf_md_cb, rmsf_md_cb_eq = get_rmsd_rmsf_md(trj_md1)

    pdb = '3SNY'

    fig = pl.figure()
    pl.plot(rmsd_md2_cb * 10, label='MD3_CB')
    pl.plot(rmsd_md_cb * 10, label='MD2_CB')
    pl.legend()
    pl.title('RMSD')
    pl.xlabel('steps')
    pl.ylabel(r'RMSD [$\AA$]')
    pl.savefig(f'{pdb}_A_rmsd_md3.pdf')
    pl.close(fig)

    fig = pl.figure()
    pl.plot(rmsf_md2_cb * 10, label='MD3_CB')
    pl.plot(rmsf_md2_cb_eq * 10, label='MD3_CB_eq')
    pl.plot(rmsf_md_cb * 10, label='MD2_CB')
    pl.plot(rmsf_md_cb_eq * 10, label='MD2_CB_eq')
    pl.legend()
    pl.title('RMSF')
    pl.xlabel('Resdidue Number')
    pl.ylabel(r'RMSF [$\AA$]')
    pl.savefig(f'{pdb}_A_rmsf_md3.pdf')
    pl.close(fig)


def plot_stability_energy(data_flag):
    data_file = f'/home/hyang/bio/erf/data/stability/{data_flag}/energy.csv'
    # data_file = 'rosetta/rosetta_energy.csv'

    df_sta = pd.read_csv('/home/hyang/bio/erf/data/stability/stability_scores/rd1234_stability_score.csv')
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

    pl.figure()
    idx = (pdb_t == 'EEHEE')
    pl.plot(energy[idx], sta_score2[idx], 'b.', label='EEHEE')
    idx = (pdb_t == 'EHEE')
    pl.plot(energy[idx], sta_score2[idx], 'g.', label='EHEE')
    idx = (pdb_t == 'HEEH')
    pl.plot(energy[idx], sta_score2[idx], 'm.', label='HEEH')
    idx = (pdb_t == 'HHH')
    pl.plot(energy[idx], sta_score2[idx], 'r.', label='HHH')
    pl.xlabel('energy score')
    # pl.ylabel('protease stability score')
    pl.ylabel('stability score')
    pl.title(t)
    pl.savefig(data_file[:-3] + 'all.pdf')
    print('all', stats.pearsonr(energy, sta_score2))

    energy_bin = np.arange(6) * 50.0 + 725
    x_bin_all = []
    y_bin_all = []
    pl.figure()
    for t in ['EEHEE', 'EHEE', 'HEEH', 'HHH']:
        idx = (pdb_t == t)
        x_bin = []
        y_bin = []
        y_err_bin = []
        e = energy[idx]
        s = sta_score2[idx]
        for j in range(6):
            idx_e = (e > energy_bin[j] - 25) & (e < energy_bin[j] + 25)
            if len(e[idx_e]) > 10:
                x_bin.append(energy_bin[j])
                y_bin.append(np.mean(s[idx_e]))
                y_err_bin.append(np.std(s[idx_e]))
        pl.errorbar(np.array(x_bin), np.array(y_bin), xerr=25, yerr=np.array(y_err_bin), fmt='o', capsize=4, label=t)
        x_bin_all += x_bin
        y_bin_all += y_bin
    pl.xlabel('energy score')
    pl.ylabel('stability score')
    pl.legend()
    pl.savefig(data_file[:-3] + 'all_bin.pdf')
    print('all_bin', stats.pearsonr(np.array(x_bin_all), np.array(y_bin_all)))


def plot_zdock_decoys(data_flag):
    root_dir = '/home/hyang/bio/erf/data/decoys/zdock/decoys/'
    pdb_list = pd.read_csv(f'{root_dir}/../pdb_list.txt')['pdb'].values
    # pdb_list = np.array(['1HE8'])

    for pdb_id in pdb_list:
        df = pd.read_csv(f'{root_dir}/{data_flag}/{pdb_id}_loss.csv')
        decoy_name = df['pdb'].values
        loss = df['loss'].values
        df_rmsd = pd.read_csv(f'{root_dir}/{pdb_id}/{pdb_id}.zd3.0.2.cg.out.rmsds',
                              header=None, sep='\s+',)
                              # names=['pdb', 'RMSD'])
        # decoy_name2 = df_rmsd['pdb'].values
        decoy_name2 = df_rmsd[0].values
        assert(np.sum(decoy_name == decoy_name2) == len(decoy_name))
        # rmsd = df_rmsd['RMSD'].values
        rmsd = df_rmsd[1].values

        fig = pl.figure()
        pl.plot(rmsd, loss, 'bo')
        pl.xlim(0, rmsd.max()+2)
        pl.title(f'{pdb_id}')
        pl.xlabel('RMSD')
        pl.ylabel('energy score')
        pl.savefig(f'{root_dir}/{data_flag}/{pdb_id}_score.pdf')
        pl.close(fig)


def scp_plots(data_flag):
    # write a zsh script to copy plots from cbio to local
    # plot_dir = f'/Users/Plover/study/bio/play/erf/data/decoy/plot/{data_flag}'
    plot_dir = f'/home/plover/study/bio/play/erf/data/decoy/plot/{data_flag}'
    if not os.path.exists(f'{plot_dir}'):
        os.system(f'mkdir -p {plot_dir}')

    with open(f'{plot_dir}/{data_flag}_scp.sh', 'wt') as mf:
        mf.write(f"""
mkdir 3drobot_{data_flag}
scp $cbio:~/bio/erf/data/decoys/3DRobot_set/decoy_loss_{data_flag}/\*.pdf 3drobot_{data_flag}/
mkdir casp11_{data_flag}
scp $cbio:~/bio/erf/data/decoys/casp11/decoy_loss_{data_flag}/\*.pdf casp11_{data_flag}/
mkdir casp13_{data_flag}
scp $cbio:~/bio/erf/data/decoys/casp13/decoy_loss_{data_flag}/\*.pdf casp13_{data_flag}/
mkdir casp14_{data_flag}
scp $cbio:~/bio/erf/data/decoys/casp14/decoy_loss_{data_flag}/\*.pdf casp14_{data_flag}/
mkdir ru_4state_{data_flag}
scp $cbio:~/bio/erf/data/decoys/rudecoy/multiple/4state_reduced/{data_flag}/\*.pdf ru_4state_{data_flag}/
mkdir ru_lattice_{data_flag}
scp $cbio:~/bio/erf/data/decoys/rudecoy/multiple/lattice_ssfit/{data_flag}/\*.pdf ru_lattice_{data_flag}/
mkdir ru_lmds_{data_flag}
scp $cbio:~/bio/erf/data/decoys/rudecoy/multiple/lmds/{data_flag}/\*.pdf ru_lmds_{data_flag}/
mkdir ru_lmds2_{data_flag}
scp $cbio:~/bio/erf/data/decoys/rudecoy/multiple/lmds_v2/{data_flag}/\*.pdf ru_lmds2_{data_flag}/
mkdir md_{data_flag}
scp $cbio:/home/hyang/bio/openmm/data/{data_flag}/\*.pdf md_{data_flag}/
mkdir loop_{data_flag}
scp $cbio:/home/hyang/bio/erf/data/fold/{data_flag}anneal_val_deep_loop/\*.pdf loop_{data_flag}/
mkdir md_bd_loop_{data_flag}
scp $cbio:/home/hyang/bio/erf/data/fold/{data_flag}_md_bd_loop_plot/\*.pdf md_bd_loop_{data_flag}/
mkdir decoy_seq_{data_flag}
scp $cbio:/home/hyang/bio/erf/data/decoys/decoys_seq/{data_flag}/\*.pdf decoy_seq_{data_flag}/
mkdir bd_md_rmsf_{data_flag}
scp $cbio:/home/hyang/bio/erf/data/fold/{data_flag}dynamics_val_deep/\*.pdf bd_md_rmsf_{data_flag}/
mkdir stability_{data_flag}
scp $cbio:/home/hyang/bio/erf/data/stability/{data_flag}/\*.pdf stability_{data_flag}/
""")


if __name__ == '__main__':
    data_flag = 'exp216'
    plot_3drobot(data_flag)
    plot_casp11(data_flag)
    plot_casp13(data_flag)
    plot_casp13(data_flag, casp_id='casp14')
    plot_ru('4state_reduced', data_flag)
    plot_ru('lattice_ssfit', data_flag)
    plot_ru('lmds', data_flag)
    plot_ru('lmds_v2', data_flag)
    plot_md_trj(data_flag)
    # plot_make_decoy(data_flag)
    plot_openmm_bd_loop_decoy(data_flag, plot_frag=False)
    plot_decoy_seq(data_flag)
    plot_polyAA(data_flag)
    plot_bd_md_rmsf(data_flag)
    plot_stability_energy(data_flag)

    for flag in range(11, 36):
        print(flag)
        # plot_openmm_bd_loop_decoy(data_flag, flag=str(flag))
        plot_bd_md_rmsf(data_flag, flag=str(flag))

    for flag in range(101, 173):
        print(flag)
        # plot_openmm_bd_loop_decoy(data_flag, flag=str(flag))
        plot_bd_md_rmsf(data_flag, flag=str(flag))

    plot_openmm_bd_loop_decoy(data_flag, flag='7', pdb_all=True)
    plot_bd_md_rmsf(data_flag, flag='7', pdb_all=True)

    for flag in range(201, 207):
        print(flag)
        plot_openmm_bd_loop_decoy(data_flag, flag=str(flag), pdb_all=True)
        plot_bd_md_rmsf(data_flag, flag=str(flag), pdb_all=True)

    plot_openmm_bd_loop_decoy(data_flag, flag='8', pdb_all=True)
    plot_bd_md_rmsf(data_flag, flag='8', pdb_all=True)

    plot_openmm_bd_loop_decoy(data_flag, flag='9', pdb_all=False)
    plot_bd_md_rmsf(data_flag, flag='9', pdb_all=False)

    flag_list = np.append(np.arange(40) + 301,
                          np.array([4, 5, 7, 8, 201, 202, 203, 204, 205, 206]))
    for flag in flag_list:
        print(flag)
        plot_openmm_bd_loop_decoy(data_flag, flag=str(flag), pdb_all=True)
        # plot_bd_md_rmsf(data_flag, flag=str(flag), pdb_all=True)

    plot_bd_md_rmsf2(data_flag, flag_list, pdb_all=True)

    plot_bd_md_rmsf2(data_flag, [400], pdb_all=True, b_factor=True)






