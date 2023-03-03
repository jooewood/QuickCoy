#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:14:18 2022

@author: zdx
"""
from plot import DataFrameDistribution, df_feature_comparison
import pandas as pd
import numpy as np
from tqdm import tqdm

def count_active_decoy_for_each_target(df, target_col_id='target', output_file=None):
    
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    
    if target_col_id not in df.columns:
        targets = []
        for active_id in df.active_id:
            target = active_id.split('_')[0]
            targets.append(target)
        df.insert(0, target_col_id, targets)
    
    targets = list(set(targets))
    active_ns = []
    decoy_ns = []
    for target in targets:
        target_df = df[df['target']==target]
        active_n = len(set(target_df.active))
        decoy_n = len(set(target_df.decoy))
        active_ns.append(active_n)
        decoy_ns.append(decoy_n)
    
    df = pd.DataFrame({
        'target': targets,
        'active_count': active_ns,
        'decoy_count': decoy_ns
        })
    if output_file is not None:
        df.to_csv(output_file, index=False)
    return df

def count_decoy_for_each_active(df, active_col='active', 
                                active_id_col='active_id',
                                target_col_id='target'):
    """
    df = '/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/0-discrete-matched-all.csv'
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    
    if target_col_id not in df.columns:
        targets = []
        for active_id in df.active_id:
            target = active_id.split('_')[0]
            targets.append(target)
        df.insert(0, target_col_id, targets)
        
    active_df = df[[target_col_id, active_id_col, active_col]].drop_duplicates(
        subset=[active_id_col])
    
    decoy_ns = []
    for id_ in tqdm(active_df.active_id):
        # id_ = 'dyr_active_0'
        sub_df = df[df[active_id_col]==id_]
        decoy_ns.append(len(sub_df))
    active_df['decoy_n'] = decoy_ns
    return active_df
    
pro = ['atom_n', 'heavy_n', 'boron_n',
       'carbon_n', 'nitrogen_n', 'oxygen_n', 'fluorine_n', 'phosphorus_n',
       'sulfur_n', 'chlorine_n', 'bromine_n', 'iodine_n', 'logP', 'HBA', 'HBD',
       'rings', 'stereo_centers', 'MW', 'aromatic_rings', 'NRB', 'pos_charge',
       'neg_charge', 'formal_net_charge', 'TPSA', 'SA', 'QED']

# QuickCoy selected decoy count decoy number for each active
res = count_decoy_for_each_active('/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/0-discrete-matched-all.csv')
res.to_csv('/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/QuickCoy_each_active_summary.csv', index=False)

hxk4 = res[res['target']=='hxk4']


# QuickCoy selected decoy count active and decoy number for each target
quick = count_active_decoy_for_each_target(
    '/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/0-discrete-matched-all.csv')
    #output_file='/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E_active_decoy_number.csv')
# DeepCoy active decoy number count
Deep = count_active_decoy_for_each_target(
    '/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_diff/1-diff-all.csv')
    #output_file='/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_active_decoy_number.csv')

DeepMergeQuick = Deep.merge(quick, on='target', how='left',  suffixes=('_DeepCoy', '_QuickCoy'))

DeepMergeQuick['active_match'] = DeepMergeQuick['active_count_DeepCoy'] - DeepMergeQuick['active_count_QuickCoy'] 




"""
DeepCoy DUD-E property difference
DUD-E
"""
df = pd.read_csv('/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_diff/1-diff-all.csv')
df_pro_abs = df[pro].applymap(abs)
df = pd.concat([df[['active_id', 'active', 'decoy']], df_pro_abs], axis=1)

DataFrameDistribution(df, columns=pro, plot_type='vio', figsize=(40,8))
DataFrameDistribution(df, columns=pro, plot_type='box', figsize=(40,8))
df_stat = df.describe(percentiles=[.05, .95])


df_pro_diff_max = df.groupby(by='active_id')[pro].agg(max)
df_pro_diff_max_stat = df_pro_diff_max.describe(percentiles=[.05, .95])
DataFrameDistribution(df_pro_diff_max, columns=pro, plot_type='box', figsize=(40,8))
DataFrameDistribution(df_pro_diff_max, columns=pro, plot_type='vio', figsize=(40,8))

# QuickCoy DUD-E
df = pd.read_csv('/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/1-diff-all.csv')
df_pro_abs = df[pro].applymap(abs)
df = pd.concat([df[['active_id', 'active', 'decoy']], df_pro_abs], axis=1)

DataFrameDistribution(df, columns=pro, plot_type='vio', figsize=(40,8))
DataFrameDistribution(df, columns=pro, plot_type='box', figsize=(40,8))
df_stat = df.describe(percentiles=[.05, .95])

df_pro_diff_max = df.groupby(by='active_id')[pro].agg(max)
df_pro_diff_max_stat = df_pro_diff_max.describe(percentiles=[.05, .95])
DataFrameDistribution(df_pro_diff_max, columns=pro, plot_type='box', figsize=(40,8))
DataFrameDistribution(df_pro_diff_max, columns=pro, plot_type='vio', figsize=(40,8))


# DUD-E --- DeepCoy vs QuickCoy
deepcoy_dude = pd.read_csv('/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_diff/1-diff-all.csv')
quickcoy_dude = pd.read_csv('/home/zdx/project/decoy_generation/result/REAL_202/selected_decoy/DUD-E/1-diff-all.csv')
df_feature_comparison(deepcoy_dude, quickcoy_dude, columns=pro,
                      output_file='/home/zdx/project/decoy_generation/picture/DUD-E_DeepCoy_vs_QuickCoy.png',
                      left_name="DeepCoy", right_name="QuickCoy", figsize=(40,8))

# DeepCoy DUD-E property statistic

df = pd.read_csv('/home/zdx/data/decoy_generation/DeepCoy_decoys/DUD-E_diff/0-discrete-matched-all.csv')

df_pro_std = df.groupby(by='active_id')[pro].agg(np.std, ddof=0)
DataFrameDistribution(df_pro_std, columns=pro, plot_type='box', figsize=(40,8))
DataFrameDistribution(df_pro_std, columns=pro, plot_type='vio', figsize=(40,8))

"""
Initial generation
"""
df = pd.read_csv('/home/zdx/project/decoy_generation/result/REAL_202/init_decoy/DUD-E_active_summary.csv')

stat = DataFrameDistribution(df, columns=['decoy_count', 'validity', 'time_cost',
       'init_gen_n'], plot_type='vio', figsize=(6,6))

df = pd.read_csv('/home/zdx/project/decoy_generation/result/REAL_202/init_decoy/DEKOIS_active_summary.csv')

stat = DataFrameDistribution(df, columns=['decoy_count', 'validity', 'time_cost',
       'init_gen_n'], plot_type='vio', figsize=(10,8))