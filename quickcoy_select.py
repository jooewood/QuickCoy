#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:33:40 2022

@author: zdx
"""
import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

from utils import remove_duplicate_mols, add_decoy_properties, GetFileName,\
    build_new_folder, ProcessBigDataFrame, parallel_molSimilarity,\
    lads_score_v2, doe_score, fileExist, ProcessBigDataFrame
from tqdm import tqdm

def DecoyScore(df, decoy_col='decoy', active_col='active',
               active_id_col='active_id', active_df=None, compute=True,
                  pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
                       'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
                       'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
                       'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
                       'HBD', 'rings', 'stereo_centers', 'MW', # 4
                       'aromatic_rings', 'NRB', 'pos_charge', # 3
                       'neg_charge', 'formal_net_charge', 'TPSA', # 3
                       'SA', 'QED' # 2
                       ], doe=True):
    columns = df.columns
    
    if active_df is None:
        active_df = df[[active_id_col, active_col]].drop_duplicates(subset=[
            active_id_col])
        
    if compute:
        active_df = add_decoy_properties(active_df, smiles_col=active_col, 
                                         pro=pro)
        df = add_decoy_properties(df, smiles_col=decoy_col, pro=pro)
        
    actives_feat = active_df[pro].values
    decoys_feat = df[pro].values
    
    if doe:
        print('Computer DOE score.')
        ave_doe = doe_score(actives_feat, decoys_feat)
    
    actives = list(active_df[active_col].values)

    labels = []
    for i, active in tqdm(enumerate(actives), total=len(actives)):
        # active = actives[0]
        label = '_'.join(['tanimoto', str(i)])
        labels.append(label)
        df = parallel_molSimilarity(df, active, smiles_col=decoy_col, 
                                    label=label)
        
    df_s = df[columns]
    df_s['dg_score'] = df[labels].max(axis=1).values
    df_s['lads_score'] = lads_score_v2(actives, df[decoy_col].values)

    if doe:
        active_dg_score = df[labels].max(axis=0).values
        ave_dg = np.mean(active_dg_score)
        max_dg = max(active_dg_score)
        ave_lads = np.mean(df_s['lads_score'].values)
        max_lads = max(df_s['lads_score'].values)
        ave_sa = np.mean(df['SA'].values)
        ave_qed = np.mean(df['QED'].values)
        return df_s, pd.DataFrame({ 
                    'ave_doe': round(ave_doe, 4),
                    'ave_dg':  round(ave_dg, 4), 
                    'max_dg':  round(max_dg, 4), 
                    'ave_lads':round(ave_lads, 4),
                    'max_lads':round(max_lads, 4),
                    'ave_sa':  round(ave_sa, 2),
                    'ave_qed': round(ave_qed,3)
                }, index=[0])
    else:
        return df_s


def batchDecoyScore(files, outfile=None, 
                    pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
                         'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
                         'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
                         'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
                         'HBD', 'rings', 'stereo_centers', 'MW', # 4
                         'aromatic_rings', 'NRB', 'pos_charge', # 3
                         'neg_charge', 'formal_net_charge', 'TPSA', # 3
                         'SA', 'QED' # 2
                         ], compute=True):
    
    targets = []
    evls = []
    for file in tqdm(files):
        # file = '/home/zdx/data/decoy_generation/DeepCoy_decoys/DeepCoy-DUDE-SMILES/dude-target-mk01-decoys-final.csv'
        filename = GetFileName(file)
        target = filename.split('-')[2]
        targets.append(target)
        df = pd.read_csv(file)
        try:
            _, sub_evl = DecoyScore(df, pro=pro, compute=compute)
        except:
            sub_evl = pd.DataFrame({ 
                        'ave_doe': np.nan, 
                        'ave_dg':  np.nan, 
                        'max_dg':  np.nan, 
                        'ave_lads':np.nan,
                        'max_lads':np.nan,
                        'ave_sa':  np.nan,
                        'ave_qed': np.nan
                    }, index=[0])
        evls.append(sub_evl)
    evl = pd.concat(evls)
    evl = pd.concat([evl, evl.mean(level=0)])
    targets.append('MEAN')
    evl.insert(0, 'target', targets)
    if outfile is not None:
        evl.to_csv(outfile, index=False)
    return evl

# files = glob.glob('/home/zdx/data/decoy_generation/DeepCoy_decoys/DeepCoy-DUDE-SMILES/*final.csv')
# batchDecoyScore(files, '/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_eval.csv')

# files = glob.glob('/home/zdx/data/decoy_generation/DeepCoy_decoys/DeepCoy-DEKOIS-SMILES/*final.csv')
# batchDecoyScore(files, '/home/zdx/data/decoy_generation/DeepCoy_decoys/DEKOIS_eval.csv')

# files = glob.glob('/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/*discrete-matched.csv')
# batchDecoyScore(files, '/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E_eval.csv', compute=False)

# files = glob.glob('/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/*discrete-matched.csv')
# batchDecoyScore(files, '/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E_eval.csv')


def propertyMatch(active, df, decoy_col='decoy', active_col='active',
                  active_id_col='active_id', select=True,
                  pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
                       'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
                       'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
                       'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
                       'HBD', 'rings', 'stereo_centers', 'MW', # 4
                       'aromatic_rings', 'NRB', 'pos_charge', # 3
                       'neg_charge', 'formal_net_charge', 'TPSA', # 3
                       'SA', 'QED' # 2
                       ], chunksize=1000, condition_file='95_difference.txt',
                  candidate_n=50
                  ):
    # discrete_var = ['atom_n', 'heavy_n', 'carbon_n',# 4
    #  'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
    #  'sulfur_n', 'chlorine_n', # 3
    #  'bromine_n', 'HBA', 'HBD', 'rings',
    #  'stereo_centers', 'aromatic_rings', 'NRB']
    # continus_var = ['logP', 'MW', 'TPSA', 'SA', 'QED']
    # df = sub_df
    # org_df = df.copy()
    if select:
        condition_file = os.path.join('./decoy_condition/', condition_file)
        with open(condition_file, 'r') as f:
            condition_text = f.readline()
        
        # remove active decoy duplicate
        active_inchi = active['InChI'].values[0]
        active_label_series = active[pro].squeeze(axis=0)
        df = df[df['InChI']!=active_inchi]
        df_diff = df[pro].sub(active_label_series, axis=1)
        df_matched = df.loc[df_diff.query(condition_text).index]
        df_matched.sort_values('dg_score', inplace=True)
        df_matched = df_matched[:candidate_n]
    else:
        df_matched = df
    return df_matched

def selectDecoy(file, out_dir, save=False, select=True, 
    active_col='active', decoy_col='decoy', active_id_col='active_id',
                pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
                     'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
                     'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
                     'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
                     'HBD', 'rings', 'stereo_centers', 'MW', # 4
                     'aromatic_rings', 'NRB', 'pos_charge', # 3
                     'neg_charge', 'formal_net_charge', 'TPSA', # 3
                     'SA', 'QED' # 2
                     ], condition_file='95_difference.txt', chunksize=1000,
                candidate_n=50):
    """
    save=False
    out_dir=None
    select=True
    active_col='active'
    decoy_col='decoy'
    active_id_col='active_id'
    pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
         'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
         'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
         'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
         'HBD', 'rings', 'stereo_centers', 'MW', # 4
         'aromatic_rings', 'NRB', 'pos_charge', # 3
         'neg_charge', 'formal_net_charge', 'TPSA', # 3
         'SA', 'QED']
    chunksize = 1000  
    candidate_n=50
    condition_file='95_difference.txt'
    
    file = '/home/zdx/project/decoy_generation/result/REAL_202/init_decoy/DUD-E/dude-target-abl1-decoys-final.csv'
    
    """ 
    in_dir = os.path.dirname(file)
    filename = GetFileName(file)
    target = filename.split('-')[2]
    decoy_add_feature_file = os.path.join(in_dir, f'{target}_decoy.csv')
    active_add_feature_file = os.path.join(in_dir, f'{target}_active.csv')
    
    if not os.path.exists(decoy_add_feature_file) or not\
        os.path.exists(active_add_feature_file):
        # Add properties
        df = pd.read_csv(file, chunksize=chunksize)
        # df = list(df)
        active_sub_dfs = []
        dfs = []
        print('Adding properties...')
        for i, sub_df in enumerate(df):
            # sub_df = df[0]
            print(i)
            active_sub_dfs.append(sub_df[[active_id_col, active_col]])
            sub_df = remove_duplicate_mols(sub_df, smiles_col=decoy_col, 
                                         remain_InChI=True)
            sub_df = add_decoy_properties(sub_df, smiles_col=decoy_col, 
                                          pro=pro, id_col=None)
            dfs.append(sub_df)
        active_df = pd.concat(active_sub_dfs)
        active_df.drop_duplicates(active_id_col, inplace=True)
        active_df = add_decoy_properties(active_df, smiles_col=active_col, 
                                      pro=['InChI']+pro, id_col=None)
        
        active_df.to_csv(active_add_feature_file, index=False)
        print('Save active with properties.')
        df = pd.concat(dfs)
        df.drop_duplicates('InChI', inplace=True)
        
        # Add DG score and LADS score
        print('Adding DGscore and LADS score.')
        df = DecoyScore(df, active_df=active_df, compute=False, doe=False)
        df.to_csv(decoy_add_feature_file, index=False)
    else:
        active_df = pd.read_csv(active_add_feature_file)
        df = pd.read_csv(decoy_add_feature_file)

    # matched decoys
    dfs_matched = []
    print("Selecting decoys...")
    for active_id in tqdm(active_df[active_id_col]):
        # active_id = 'abl1_active_31'
        print(active_id)
        df_sub = df[df[active_id_col]==active_id]
        active = active_df[active_df[active_id_col]==active_id]
        matched = propertyMatch(active, df_sub, pro=pro, 
                                condition_file=condition_file, select=select,
                                candidate_n=candidate_n)
        dfs_matched.append(matched)
    df_matched = pd.concat(dfs_matched)
    
    if save:
        selected_decoy_path = os.path.join(out_dir, f'{target}.csv')
        df_matched.to_csv(selected_decoy_path,index=False)


def batchSelectDecoy(files, out_dir=None, save=False, select=True, 
                     active_col='active', decoy_col='decoy', 
                     active_id_col='active_id',     
                pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
                     'nitrogen_n', 'oxygen_n', 'fluorine_n', # 7
                     'phosphorus_n', 'sulfur_n', 'chlorine_n', # 10
                     'bromine_n', 'iodine_n', 'logP', 'HBA', # 14
                     'HBD', 'rings', 'stereo_centers', 'MW', # 18
                     'aromatic_rings', 'NRB', 'pos_charge', # 21
                     'neg_charge', 'formal_net_charge', 'TPSA', # 24
                     'SA', 'QED' 
                     ], # 26
                candidate_n=50
    ):
    
    if out_dir is None:
        out_dir = os.path.dirname(files[0])
    if save:
        build_new_folder(out_dir)
        
    failed = []
    for file in tqdm(files):
        try:
            selectDecoy(file=file, save=save, select=select, 
                        active_col=active_col,
                        decoy_col=decoy_col, out_dir=out_dir,
                        active_id_col=active_id_col, pro=pro,
                        candidate_n=candidate_n)
        except:
            failed.append(file)
    print(failed)

input_dir = '/home/zdx/project/decoy_generation/result/REAL_202/init_decoy/DUD-E'
out_dir = '/home/zdx/data/VS_dataset/DUD-E/quickcoy_decoys'
files = glob.glob(os.path.join(input_dir, '*final.csv'))
batchSelectDecoy(files, out_dir=out_dir, save=True)

# input_dir = '/home/zdx/data/decoy_generation/DeepCoy_decoys/DeepCoy-DUDE-SMILES'
# files = glob.glob(os.path.join(input_dir, '*final.csv'))
# batchSelectDecoy(files, out_dir='/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_diff', 
#                  save=True, select=False)

# input_dir = '/home/zdx/data/decoy_generation/DeepCoy_decoys/DeepCoy-DEKOIS-SMILES'
# files = glob.glob(os.path.join(input_dir, '*final.csv'))
# batchSelectDecoy(files, out_dir='/home/zdx/data/decoy_generation/DeepCoy_decoys/DEKOIS_diff', 
#                  save=True, select=False)

# input_dir = '/home/zdx/project/decoy_generation/result/REAL_202/init_decoy/DUD-E'
# files = glob.glob(os.path.join(input_dir, '*final.csv'))
# batchSelectDecoy(files, out_dir='/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E', 
#                   save=True)

# input_dir = '/home/zdx/project/decoy_generation/result/REAL_202/init_decoy/DEKOIS'
# files = glob.glob(os.path.join(input_dir, '*final.csv'))
# batchSelectDecoy(files, out_dir='/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DEKOIS', 
#                   save=True)