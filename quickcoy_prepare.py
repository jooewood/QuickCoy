#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:41:53 2022

@author: zdx
"""
import os
import pandas as pd
from utils import get_can_smiles, GetFileName, add_can_smiles, saveDictJson,\
    readJson, build_new_folder, add_decoy_properties, adjust_order
import glob
from tqdm import tqdm

data_root = '/home/zdx/data/VS_dataset'

## Remove duplicate

# # DeepCoy DUD-E add True ID
# dataset = 'DUD-E'
# files = glob.glob('/home/zdx/data/DeepCoy_decoys/DeepCoy-DUDE-SMILES/*.txt')
# missed = {}
# for i, file in enumerate(tqdm(files)):
#     # file = files[0]
    
#     df = pd.read_csv(file, sep=' ', header=None)
#     df.columns = ['active', 'decoy']
#     target = GetFileName(file).split('-')[2]
    
#     raw_active_file = os.path.join(data_root, dataset, 'original_actives', f'{target}.csv')
    
#     df = add_can_smiles(df, smiles_col='active', overwrite=True)
#     df = add_can_smiles(df, smiles_col='decoy', overwrite=True)
    
#     df_active = pd.read_csv(raw_active_file)
    
#     dfs = []
#     missed[target] = []
#     for i, active in enumerate(tqdm(df_active.SMILES)):
#         # i=0; active = df_active.SMILES[i]
#         try:
#             df_sub = df.query(f'active=="{active}"')
#             active_id = df_active.ID.values[i]
#             df_sub.insert(0, 'active_id', active_id)
#             decoys = list(set(list(df_sub['decoy'])))
            
#             decoy_ids = []
#             for i, decoy in enumerate(decoys):
#                 # i = 0
#                 decoy = decoys[i]
#                 decoy_id = f'{target}-{active_id}-decoy-{str(i)}'
#                 df_sub.loc[df_sub['decoy']==decoy, 'decoy_id'] = decoy_id
#             dfs.append(df_sub)
#         except:
#             missed[target].append(active_id)
    
#     try:
#         if len(missed[target]) == 0:
#             try:
#                 del missed[target]
#             except:
#                 pass
#     except:
#         pass
        
#     df = pd.concat(dfs)
#     raw_process_out = os.path.join(data_root, dataset, '0_raw', 'DeepCoy', f'{target}.csv')
#     raw_process_out_dir = os.path.dirname(raw_process_out)
#     build_new_folder(raw_process_out_dir)
#     df.to_csv(raw_process_out, index=False)

# missed_path = os.path.join(data_root, dataset, 'DeepCoy_missed_active.json')
# saveDictJson(missed, missed_path)

# # Divide processed DeepCoy file into deepcoy_actives and deepcoy_decoys
# files = glob.glob(os.path.join(data_root, dataset, '0_raw', 'DeepCoy', '*.csv'))

# for file in tqdm(files):
#     # file = files[0]
#     target = GetFileName(file)
#     df = pd.read_csv(file)
#     for x in ['active', 'decoy']:
#         if x == 'active':
#             df_tmp = df[[f'{x}_id', x]].drop_duplicates(subset=f'{x}_id')
#             df_tmp.columns = ['ID', 'SMILES']
#         elif x == 'decoy':
#             df_tmp = df[[f'{x}_id', x, 'active_id']]
#             df_tmp.columns = ['ID', 'SMILES', 'active_id']
            
#         df_tmp = add_decoy_properties(df_tmp, drop_dup=True)

#         if x == 'active':
#             df_tmp['label'] = 1
#         elif x == 'decoy':
#             df_tmp['label'] = 0
            
#         df_tmp_path = os.path.join(data_root, dataset, f'deepcoy_{x}s', f'{target}.csv')
#         df_tmp.to_csv(df_tmp_path, index=False)

## DUD-E data drop duplicate
# dataset = 'DUD-E'
# for x in ['actives', 'decoys']:
#     # x = 'actives'
#     files = glob.glob(os.path.join(data_root, dataset, f'original_{x}', '*.csv'))
#     for file in files:
#         df = pd.read_csv(file)
#         df.drop_duplicates(subset='ID', inplace=True)

## QuickCoy 1st result add true ID
# files = glob.glob('/home/zdx/project/decoy_generation/result/REAL_212/init_decoy/DUD-E/*.csv')
# files = [file for file in files if 'summary' not in file]
# missed = {}
# for i, file in enumerate(tqdm(files)):
#     # file = files[0]
    
#     df = pd.read_csv(file)
#     df = df[['active', 'decoy']]
#     target = GetFileName(file).split('-')[2]
    
#     raw_active_file = os.path.join(data_root, dataset, 'original_actives', f'{target}.csv')
    
#     df = add_can_smiles(df, smiles_col='active', overwrite=True)
#     df = add_can_smiles(df, smiles_col='decoy', overwrite=True)
    
#     df_active = pd.read_csv(raw_active_file)
    
#     dfs = []
#     missed[target] = []
#     for i, active in enumerate(tqdm(df_active.SMILES)):
#         # i=0; active = df_active.SMILES[i]
#         try:
#             df_sub = df.query(f'active=="{active}"')
#             active_id = df_active.ID.values[i]
#             df_sub.insert(0, 'active_id', active_id)
#             decoys = list(set(list(df_sub['decoy'])))
            
#             decoy_ids = []
#             for i, decoy in enumerate(decoys):
#                 # i = 0
#                 decoy = decoys[i]
#                 decoy_id = f'{target}-{active_id}-decoy-{str(i)}'
#                 df_sub.loc[df_sub['decoy']==decoy, 'decoy_id'] = decoy_id
#             dfs.append(df_sub)
#         except:
#             missed[target].append(active_id)
    
#     try:
#         if len(missed[target]) == 0:
#             try:
#                 del missed[target]
#             except:
#                 pass
#     except:
#         pass
        
#     df = pd.concat(dfs)
#     raw_process_out = os.path.join('/home/zdx/project/decoy_generation/result/REAL_212/init_decoy/DUD-E/', f'{target}.csv')
#     # raw_process_out_dir = os.path.dirname(raw_process_out)
#     # build_new_folder(raw_process_out_dir)
#     df.to_csv(raw_process_out, index=False)
# missed_path = os.path.join('/home/zdx/project/decoy_generation/result/REAL_212/init_decoy/DUD-E_missed_active.json')
# saveDictJson(missed, missed_path)

# QuickCoy

# QuickCoy filtered 
dataset = 'DUD-E'
files = glob.glob(os.path.join(data_root, dataset, 'quickcoy_decoys', '*.csv'))
missed = {}
for i, file in enumerate(tqdm(files)):
    # file = files[0]
    
    df = pd.read_csv(file)
    target = GetFileName(file)
    
    raw_active_file = os.path.join(data_root, dataset, 'original_actives', f'{target}.csv')
    
    df = add_can_smiles(df, smiles_col='active', overwrite=True)
    df = add_can_smiles(df, smiles_col='decoy', overwrite=True)
    
    df_active = pd.read_csv(raw_active_file)
    
    dfs = []
    missed[target] = []
    for i, active in enumerate(tqdm(df_active.SMILES)):
        # i=0; active = df_active.SMILES[i]
        active_id = df_active.ID.values[i]
        try:
            df_sub = df.query(f'active=="{active}"')
            df_sub['active_id'] = active_id
            decoys = list(set(list(df_sub['decoy'])))
            
            decoy_ids = []
            for i, decoy in enumerate(decoys):
                # i = 0
                decoy = decoys[i]
                decoy_id = f'{target}-{active_id}-decoy-{str(i)}'
                df_sub.loc[df_sub['decoy']==decoy, 'ID'] = decoy_id
            dfs.append(df_sub)
        except:
            missed[target].append(active_id)
    
    try:
        if len(missed[target]) == 0:
            try:
                del missed[target]
            except:
                pass
    except:
        pass
        
    df_new = pd.concat(dfs)
    df_new.rename(columns={'decoy': 'SMILES', 'decoy_id': 'ID'}, inplace=True)
    df_new = adjust_order(df_new, pro=['ID', 'SMILES'])
    df_new.to_csv(file, index=False)
missed_path = os.path.join('/home/zdx/data/VS_dataset/DUD-E/QuickCoy_filtered_missed_active.json')
saveDictJson(missed, missed_path)
