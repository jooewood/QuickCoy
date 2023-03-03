#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes Apr 6 14:29:26 2022

@author: zdx
"""
# from operator import index
import numpy as np
import pandas as pd
import re
import os
import sys
import json
import math
import time
import pickle
import multiprocessing
from multiprocessing import Pool
from shutil import copy, rmtree
import shutil
from rdkit import Chem
from rdkit.Chem import DataStructs, PandasTools, rdMolHash, AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from moses.metrics import mol_passes_filters, QED, SA, logP
# from moses.utils import get_mol
from functools import partial
from rdkit.Chem import rdinchi
from os.path import basename, splitext, join, dirname
from glob import glob
from tqdm import tqdm
import soltrannet as stn
import biotite.sequence.io.fasta as fasta
import biotite.sequence.align as align
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import seaborn as sns
import matplotlib.pyplot as plt
# from copy import deepcopy
# from oddt.toolkits.rdk import readstring, readfile, Outputfile
from oddt.toolkits.ob import readstring, readfile, Outputfile
from collections import defaultdict
from sklearn.metrics import auc, roc_curve
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy
import pubchempy as pcp
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def tranDict(dict_):
    return {value:key for (key,value) in dict_.items()}

def smiles2smarts(smiles):
    """
    
    """
    if isinstance(smiles, list):
        smarts = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            smarts.append(Chem.MolToSmarts(mol))
    elif isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        smarts = Chem.MolToSmarts(mol)
    return smarts

def RxnSmartsFromSmiles(products, reactants):
    products_smarts = smiles2smarts(products)
    reactants_smarts = smiles2smarts(reactants)
    
    if len(products_smarts) > 1 and isinstance(products_smarts, list):
        products_smarts = '.'.join(products_smarts)
    if len(reactants_smarts) > 1 and isinstance(reactants_smarts, list):
        reactants_smarts = '.'.join(reactants_smarts)
    return '>>'.join([reactants_smarts, products_smarts])

def RxnFromSmiles(products, reactants):
    """
    products = 'Cc1nc(N(C)C)sc1-c1ccccc1'
    reactants = ['c1ccccc1C(=O)C(C)Cl', 'CN(C)C(=S)N']
    rxn_smarts = RxnFromSmiles(products, reactants)
    """
    if len(products) > 1 and isinstance(products, list):
        products = '.'.join(products)
    if len(reactants) > 1 and isinstance(reactants, list):
        reactants = '.'.join(reactants)
    return '>>'.join([reactants, products])

def get_substructure_cas(smiles):
    """
    Get CAS number of smiles
    """
    try:
        cas_rns = []
        results = pcp.get_synonyms(smiles, 'smiles', searchtype='substructure')
        for result in results:
            for syn in result.get('Synonym', []):
                match = re.match('(\d{2,7}-\d\d-\d)', syn)
                if match:
                    cas_rns.append(match.group(1))
        if isinstance(cas_rns, list):
            if len(cas_rns)>=1:
                cas_rns = cas_rns[0]
            else:
                cas_rns = None
        return cas_rns
    except:
        return None

def get_dataset_target_ids(data_root, dataset_name, 
                           target_folder_name='1_targets'):
    target_files = glob(os.path.join(data_root, dataset_name, 
                                          target_folder_name, '*.pdb'))
    target_ids = []
    for file in target_files:

        target_id = GetFileName(file)
        if dataset_name == 'LIT-PCBA':
            target_id = target_id.split('.')[0]
        target_ids.append(target_id)
        
    target_ids = list(set(target_ids))
    return target_ids

def auc_prc(y, y_pred, **kwargs):
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    return auc(recall, precision)

def get_time_string():
    time_string = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    return str(time_string)

def splitTrainTest(df, frac, random_seed=0, mode='cls', label_col='label'):  
    """
    frac = [0.8, 0.1, 0.1] [0.8, 0.2]
    """
    if len(frac) == 2:
        train_frac, val_frac = frac
        test_frac = 0
    
    if len(frac) == 3:
        train_frac, val_frac, test_frac = frac

    if mode == 'reg':
        if test_frac != 0:
            test = df.sample(frac = test_frac, replace = False, 
                            random_state = random_seed)
            train_val = df[~df.index.isin(test.index)]
        else:
            train_val = df
            test = None
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, 
                            random_state=random_seed)
        train = train_val[~train_val.index.isin(val.index)]
    elif mode == 'cls':
        labels = list(set(list(df[label_col].values)))

        dfs_train = []
        dfs_val = []
        if test_frac != 0:
            dfs_test = []

        for label in labels:
            df_sub = df[df[label_col]==label]
            if test_frac != 0:
                test_sub = df_sub.sample(frac = test_frac, replace = False, 
                                random_state = random_seed)
                dfs_test.append(test_sub)
                train_val_sub = df_sub[~df_sub.index.isin(test_sub.index)]
            else:
                train_val_sub = df_sub
                test = None
            val_sub = train_val_sub.sample(frac = val_frac/(1-test_frac), 
                replace = False, random_state=random_seed)
            train_sub = train_val_sub[~train_val_sub.index.isin(val_sub.index)]
            dfs_train.append(train_sub)
            dfs_val.append(val_sub)
        train = pd.concat(dfs_train)
        val = pd.concat(dfs_val)
        if test_frac != 0:
            test = pd.concatdfs_test()
    return train, val, test

"""
------------------
Evaluation of virtual screening
------------------
"""

def ranking_auc(df, pred_col='score', label_col='label', ascending=False, 
                sort=True): 
    """ ascending = True means sort from small to big
    Smina use ascending = True
    """
    if sort:
        df.sort_values(by=pred_col, ascending = ascending, inplace=True)
        df.reset_index(drop=True, inplace=True)
    auc_roc, auc_prc=np.nan, np.nan
    l = len(df)
    a_pos = list(df[df[label_col]==1].index)
    a_pos = np.array(a_pos) + 1
    ##Generate the seq of active cmpd
    a_seen = np.array(range(1,len(a_pos)+1))
    ##Calculate auc contribution of each active_cmpd
    d_seen = a_pos-a_seen
    ##Calculate AUC ROC
    a_total=len(a_pos)
    d_total = l - len(a_pos)
    contri_roc = d_seen/(a_total*d_total)
    auc_roc = 1.0 - np.sum(contri_roc)
    if auc_roc < 0:
        auc_roc = 0
    ##Calculate AUC PRC
    auc_prc = (1/a_total)*np.sum((1.0*a_seen)/a_pos) 
    if auc_prc < 0:
        auc_prc = 0
    return {'AUC_ROC':round(auc_roc,4), 'AUC_PRC': round(auc_prc, 4)}

def enrichment(a_pos, total_cmpd_number, top):
    ##Calculate total/active cmpd number at top% 
    top_cmpd_number = math.ceil(total_cmpd_number*top)
    top_active_number = 0
    for a in a_pos:
        if a>top_cmpd_number: break
        top_active_number += 1
    ##Calculate EF
    total_active_number = len(a_pos)
    ef = (1.0*top_active_number/top_cmpd_number)*(\
        total_cmpd_number/total_active_number)
    return round(ef, 4)
    
def enrichment_factor(df, pred_col='score', label_col='label', ascending=False,
                      ef_list=[0.01, 0.05, 0.1], sort=True):
    if sort:
        df.sort_values(by=pred_col, ascending=ascending, inplace=True)
        df.reset_index(drop=True, inplace=True)
    l = len(df)
    a_pos = df[df[label_col]==1].index
    a_pos = np.array(a_pos) + 1    
    ef_res = {}
    for ef in ef_list:
        ef_name = ''.join(['EF', '%d' % (ef*100)])
        ef_res[ef_name] = enrichment(a_pos, l,  ef)
    return ef_res

def eval_one_target(data, pred_col='score', label_col='label', sort=True,
                          ascending=False, ef_list=[0.01, 0.05, 0.1]):
    # Smina use ascending = True
    if isinstance(data, str):
        data = pd.read_csv(data)
    if not isinstance(data, pd.DataFrame):
        print('Please check your input.')
        return
    
    if sort:
        data.sort_values(by=pred_col, ascending=ascending, inplace=True)
        data.reset_index(drop=True, inplace=True)
    eval_dict = ranking_auc(data, pred_col=pred_col, label_col=label_col,
                            sort=False)
    eval_dict.update(enrichment_factor(data, ef_list=ef_list, sort=False,
                                       pred_col=pred_col, label_col=label_col))
    return eval_dict

def eval_one_dataset(data, pred_col='score', label_col='label', sort=True, 
    ascending=False, ef_list=[0.01, 0.05, 0.1], target_id_col='target_id',
    decoy_source=None, decoy_source_col='decoy_source', train_set=None,
    train_set_col='train_set', model_id_col='model_id', model_id=None
    ):
    targets = list(set(list(data[target_id_col].values)))
    eval_dicts = []
    for target_id in targets:
        sub_data = data[data[target_id_col]==target_id]
        sub_eval_dict = eval_one_target(
            data=sub_data, 
            pred_col=pred_col, 
            label_col=label_col, 
            sort=sort, 
            ascending=ascending, 
            ef_list=ef_list
            )
        eval_dicts.append(sub_eval_dict)
    df = pd.DataFrame(eval_dicts)
    if model_id is not None:
        df[model_id_col] = model_id
    df[target_id_col] = targets
    if decoy_source is not None:
        df[decoy_source_col] = decoy_source
    if train_set is not None:
        df[train_set_col] = train_set
    return df

def pdb_to_fasta(file, target_id='PDB', outfile=None, mode='w'):
    """Reads residue names of ATOM/HETATM records and exports them to a FASTA
    file.
    
    file = '/home/zdx/data/VS_dataset/DUD-E/0_raw/aa2ar/receptor.pdb'
    
    """
    fhandle = open(file, 'r')
    res_codes = [
        # 20 canonical amino acids
        ('CYS', 'C'), ('ASP', 'D'), ('SER', 'S'), ('GLN', 'Q'),
        ('LYS', 'K'), ('ILE', 'I'), ('PRO', 'P'), ('THR', 'T'),
        ('PHE', 'F'), ('ASN', 'N'), ('GLY', 'G'), ('HIS', 'H'),
        ('LEU', 'L'), ('ARG', 'R'), ('TRP', 'W'), ('ALA', 'A'),
        ('VAL', 'V'), ('GLU', 'E'), ('TYR', 'Y'), ('MET', 'M'),
        # Non-canonical amino acids
        # ('MSE', 'M'), ('SOC', 'C'),
        # Canonical xNA
        ('  U', 'U'), ('  A', 'A'), ('  G', 'G'), ('  C', 'C'),
        ('  T', 'T'),
    ]

    three_to_one = dict(res_codes)
    records = ('ATOM', 'HETATM')

    sequence = []  # list of chain sequences
    seen = set()
    prev_chain = None
    for line in fhandle:
        if line.startswith(records):

            chain_id = line[21]
            if chain_id != prev_chain:
                sequence.append([chain_id])
                prev_chain = chain_id

            res_uid = line[17:27]
            if res_uid in seen:
                continue

            seen.add(res_uid)

            aa_resn = three_to_one.get(line[17:20], 'X')
            sequence[-1].append(aa_resn)

    # Yield fasta format
    _olw = 60
    # Remove chain labels and merge into one single sequence
    sequence = [[r for c in sequence for r in c[1:]]]
        
    fasta_seq = f'>{target_id}' + '\n'

    protein_seq = ''
    for chain in sequence:
        seq = ''.join(chain)
        fmt_seq = [seq[i:i + _olw] + '\n' for i in range(0, len(seq), _olw)]
        fmt_seq = ''.join(fmt_seq)
        protein_seq += fmt_seq
        
    fasta_seq += protein_seq
    
    if outfile is not None:
        with open(outfile, mode) as f:
            f.write(fasta_seq)
    
    return fasta_seq, protein_seq.replace('\n', '')

def batchPDB2fasta(pdb_files, names=None, outfile=None):
    sequences = []
    
    flag = 0
    
    if names is None:
        names = []
        flag = 1
        
    for i, file in tqdm(enumerate(pdb_files), total=len(pdb_files)):
        if flag==1:
            name = GetFileName(file)
            names.append(name)
        else:
            name = names[i]
        if i == 0:
            _, protein_seq = pdb_to_fasta(file, name, outfile, 'w')
        else:
            _, protein_seq = pdb_to_fasta(file, name, outfile, 'a')
        sequences.append(protein_seq)
    return sequences, names

def MultiProteinAlignments(fasta_file=None, names=None, out_dir=None, dpi=300, 
                           identities_file=None, plot=False,
                            figsize=(10, 30), index_col=0, tight_layout = True,
                            method='average',
                            **kwargs):
    """
    >>> file = '/y/Aurora/Fernie/data/DUD-E_Kernie_MUV.fst'
    >>> out_dir = '/y/Aurora/Fernie/Report'
    >>> filename = 'DUD-E_Kernie_MUV'
    >>> identities = MultiProteinAliganments(file, out_dir)
    """
    if identities_file is not None:
        identities = pd.read_csv(identities_file)
    else:
        print('Getting sequences...')
        names = get_fasta_ids(fasta_file)
        fasta_reader = fasta.FastaFile()
        fasta_reader.read(fasta_file)
        sequences = list(fasta.get_sequences(fasta_reader).values())
        print('Finished.')
        
        # BLOSUM62
        print('Alignmenting ...')
        substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()
        # Matrix that will be filled with pairwise sequence identities
        identities = np.ones((len(sequences), len(sequences)))
        # Iterate over sequences
        
        for i in tqdm(range(len(sequences))):
            for j in range(i):
                # Align sequences pairwise
                alignment = align.align_optimal(
                    sequences[i], sequences[j], substitution_matrix
                )[0]
                # Calculate pairwise sequence identities and fill matrix
                identity = align.get_sequence_identity(alignment)
                identities[i,j] = identity
                identities[j,i] = identity
        identities = pd.DataFrame(identities, index = names, columns = names)
        identities = 1 - identities
        print('Finished.')
        
    if plot:
        print('Plotting ...')
        hc_linkage = hc.linkage(sp.distance.squareform(identities), 
                                method=method)
        plt.figure(dpi=dpi, figsize=figsize)
        plot = sns.clustermap(identities, row_linkage=linkage, 
            col_linkage=hc_linkage, 
            figsize=figsize, yticklabels=True, xticklabels=False, 
            col_cluster=True, row_cluster=True, **kwargs)
        fig = plot.fig
        if tight_layout:
            fig.tight_layout()
        print('Finished.')  
    
    if out_dir is not None:
        print('Saving...')
        figure_path = os.path.join(out_dir, 'clustermap.png')
        identities_path = os.path.join(out_dir, 'identities.csv')
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(identities_path): 
            identities.to_csv(identities_path, index=False)
            print(f"Succeed to save identities matrix into {identities_path}")
            
        print(f"Saving clustermap figure into {figure_path}")
        plt.savefig(figure_path) 
        plt.close()
        print('Finished')
    
    print('Converting full matrix to condensed matrix...')
    distance_matrix = identities.values
    distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    print('Finished.')
    
    return distance_matrix, list(identities.columns)

def hcluster(ytdist, labels, method="complete", plot=False, dpi=300,
             figsize=(20,50), leaf_font_size=25, orientation='left',
             fig_path=None, t=0.5, criterion='distance'):
    Z = linkage(ytdist, method=method)
    
    # Plot
    if plot:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        dendrogram(Z, labels=labels, leaf_font_size=leaf_font_size, 
                        orientation=orientation)
        fig.tight_layout()
        if fig_path is not None:
            plt.savefig(fig_path)
        else:
            plt.show()
    
    # Cluster
    cld = list(fcluster(Z, t=t, criterion=criterion))
    cluster_res = {}
    for label, c in zip(labels, cld):
        cluster_res[label] = c

    return cluster_res

def cluster2Fold(cluster_res, fold=3, sort=True, reverse=True, outfile=None):
    """
    Split dataset into different fold according to cluster result.
    cluster_res: dict, {'0':1, '1':2}
    """
    values = list(cluster_res.values())
    l = len(values)
    
    cluster_labels = list(set(values))
    label_count = {}
    for cluster_label in cluster_labels:
        label_count['%d'%cluster_label] = values.count(cluster_label)
    if sort:
        label_count = dict(sorted(label_count.items(), key=lambda x: x[1], 
                                  reverse=reverse))
    
    sorted_label = list(label_count.keys()) 
    sorted_label = list(map(int, sorted_label))
    
    fold_lens = []
    
    for i in range(0, fold):
        if i == fold-1:
            fold_lens.append(int(l/fold)+l%fold)
        else:
            fold_lens.append(int(l/fold))
    
    fold_res = {}
    for i in range(0, fold):
        # i = 0
        fold_volume = fold_lens[i]
        fold_res['%d'%i] = []
        
        while len(fold_res['%d'%i]) < fold_volume:
            for label in sorted_label:
                # label = sorted_label[0]
                keys = [k for k, v in cluster_res.items() if v == label]
                if len(fold_res['%d'%i]) + len(keys) <= fold_volume:
                    fold_res['%d'%i] += keys
                    for key in keys:
                        del cluster_res[key]
                    sorted_label.remove(label)
    if outfile is not None:
        saveDictJson(fold_res, outfile)
    return fold_res

def refinePDB(file, outfile=None):
    """
    Remove lines those not start from "ATOM".
    """

    with open(file, 'r') as f:
        lines = f.readlines()
        
    if outfile == None:
        outfile = file
        
    with open(outfile, 'w') as f:
        for l in lines:
            if l[:4]=='ATOM':
                f.write(l)

def concat_dfs(files):
    if isinstance(files[0], str):
        dfs = []
        for file in tqdm(files):
            dfs.append(pd.read_csv(file))
    else:
        dfs = files
    
    df = pd.concat(dfs)
    return df


def split_dataframe(df, chunksize = 1000): 
    chunks = list()
    num_chunks = len(df) // chunksize + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunksize:(i+1)*chunksize])
    return chunks

def ProcessBigDataFrame(file, func, outfile=None, chunksize=1000, sep=',', header=0, 
                        iterator=True):

    rows = sum(1 for _ in open(file, 'r'))
    if header is not None:
        rows = rows - 1
    
    if not isinstance(file, pd.DataFrame):
        df = split_dataframe(file)
    elif isinstance(file, str):
        df = pd.read_csv(file, sep=sep, header=header, iterator=iterator, 
                         chunksize=chunksize)
    
    with tqdm(total=rows, desc='Rows read: ') as bar:
        for i, chunk in enumerate(df):
            l = len(chunk)
            chunk = func(chunk)
            if outfile is not None:
                if i == 0:
                    chunk.to_csv(outfile, index=False)
                else:
                    chunk.to_csv(outfile, index=False, mode='a', header=None)
            bar.update(l)

def fileExist(file):
    if os.path.exists(file) and os.path.getsize(file) > 0:
        return True
    else:
        return False

def saveVariable(var, path):
    with open(path, 'wb') as fb:
        pickle.dump(var, fb)

def loadVariable(path):
    with open(path, 'rb') as fb:
        return pickle.load(fb)

def doe_score(actives, decoys):
    all_feat = list(actives) + list(decoys)
    up_p = np.percentile(all_feat, 95, axis=0)
    low_p = np.percentile(all_feat, 5, axis=0)
    norms = up_p - low_p
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1.

    active_norm = [act/norms for act in actives]
    decoy_norm = [dec/norms for dec in decoys]
    all_norm = active_norm + decoy_norm

    active_embed = []
    labels = [1] * (len(active_norm)-1) + [0] * len(decoy_norm)
    for i, act in enumerate(active_norm):
        comp = list(all_norm)
        del comp[i]
        dists = [100 - np.linalg.norm(c-act) for c in comp] # arbitrary large number to get scores in reverse order
        fpr, tpr, _ = roc_curve(labels, dists)
        fpr = fpr[::]
        tpr = tpr[::]
        a_score = 0
        for i in range(len(fpr)-1):
            a_score += (abs(0.5*( (tpr[i+1]+tpr[i])*(fpr[i+1]-fpr[i]) - (\
                fpr[i+1]+fpr[i])*(fpr[i+1]-fpr[i]) )))
        active_embed.append(a_score)

    #print(np.average(active_embed))
    return np.average(active_embed)


def dg_score_rev(actives, decoys):
    # Similar to DEKOIS
    # Lower is better (less like actives), higher is worse (more like actives)
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(\
        smi),3,useFeatures=True) for smi in actives] # Roughly FCFP_6
    decoys_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(\
        smi),3,useFeatures=True) for smi in decoys] # Roughly FCFP_6

    closest_sims = []
    closest_sims_id = []
    for decoy_fp in decoys_fps:
        active_sims = []
        for active_fp in active_fps:
            active_sims.append(DataStructs.TanimotoSimilarity(active_fp, 
                                                              decoy_fp))
        closest_sims.append(max(active_sims))
        closest_sims_id.append(np.argmax(active_sims))

    return closest_sims, closest_sims_id

def lads_score_v2(actives, decoys):
    # Similar to DEKOIS (v2)
    # Lower is better (less like actives), higher is worse (more like actives)
    # dg_scores, dg_ids = decoy_utils.dg_score_rev(set(in_mols), gen_mols)
    
    # unique actives
    # unique decoys
    
    active_fps = []
    active_info = {}
    info={}
    atoms_per_bit = defaultdict(int)
    for smi in actives:
        m = Chem.MolFromSmiles(smi)
        active_fps.append(AllChem.GetMorganFingerprint(m,3,useFeatures=True, 
                                                       bitInfo=info))
        for key in info:
            if key not in active_info:
                active_info[key] = info[key]
                env = Chem.FindAtomEnvironmentOfRadiusN(m, info[key][0][1], 
                                                        info[key][0][0])
                amap={}
                submol=Chem.PathToSubmol(m,env,atomMap=amap)
                if info[key][0][1] == 0:
                    atoms_per_bit[key] = 1
                else:
                    atoms_per_bit[key] = submol.GetNumHeavyAtoms()

    decoys_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi),3,
            useFeatures=True) for smi in decoys] # Roughly FCFP_6

    master_active_fp_freq = defaultdict(int)
    for fp in active_fps:
        fp_dict = fp.GetNonzeroElements()
        for k, v in fp_dict.items():
            master_active_fp_freq[k] += 1
    # Reweight
    for k in master_active_fp_freq:
        # Normalise
        master_active_fp_freq[k] /= len(active_fps)
        # Weight by size of bit
        master_active_fp_freq[k] *= atoms_per_bit[k]

    decoys_lads_avoid_scores = [sum([master_active_fp_freq[k] for k in\
        decoy_fp.GetNonzeroElements()])/len(decoy_fp.GetNonzeroElements())\
        for decoy_fp in decoys_fps]
    
    return decoys_lads_avoid_scores

def CountNonZeroDataFrame(df, drop=None):
    if drop is not None:
        columns = [x for x in df.columns if x not in drop]
    df = df[columns]
    res = {}
    for column in df.columns:
        res[column] = (df[column] != 0).sum()
    return res

"""
--------------------------------------------------------------------------------
File or File path operation
--------------------------------------------------------------------------------
"""
def GetFileFormat(file):
    if '.gz' in file:
        file = file.replace('.gz', '')
    format_ = os.path.splitext(os.path.basename(file))[1].split('.')[-1]
    return format_

def GetFileName(file):
    if '.gz' == file[-3:]:
        file = file.replace('.gz', '')
    name = os.path.splitext(os.path.basename(file))[0]
    return name

def GetFileNameFormat(file):
    return GetFileName(file), GetFileFormat(file)

def ChaneFileSuffix(file, suffix):
    if suffix[0] == '.':
        suffix = suffix[1:]
    return '.'.join([os.path.splitext(file)[0], suffix])

"""
--------------------------------------------------------------------------------
Molecular File converter
--------------------------------------------------------------------------------
"""
"""
========================== Convert file to mol ================================
"""

    
def file2mol(files, lazy=False, informat=None, protein=False):
    """
    files = ['/home/zdx/data/VS_dataset/DUD-E/1_targets/aa2ar.pdb', '/home/zdx/data/VS_dataset/DUD-E/1_targets/adrb1.pdb']
    file = '/home/zdx/data/VS_dataset/DUD-E/1_targets/adrb1.pdb'
    protein = True
    lazy = False
    mols = file2mol(files, protein=True)
    mol = mols[0]
    The format of input file is sdf
    Convert example.sdf to 1 molecule(s) format.
    Success!
    """
    if isinstance(files, str):
        files = [files]
        
    mols = []
    for file in files:
        if protein:
            target = 'protein(s)'
        else:
            target = 'molecule(s)'

        if informat is None:
            informat = GetFileFormat(file)
        try:
            print('The format of input file is', informat)
            assert informat in ['pdb', 'pdbqt', 'sdf', 'mol', 'mol2', 'smi', 'inchi']
        except:
            print('ERROR')
            print(f"Your file format {format} is not allowed. We support "
                  "pdb, pdbqt, sdf, mol, mol2, smi, inchi")
            return
        mols += list(readfile(informat, file, lazy=lazy))
        if protein:
            for mol in mols:
                mol.protein=True
                
    print('Success!\n')
    mols = [m for m in mols if m is not None]
    return mols

def string2mol(string=None, format_='smi'):
    """
    >>> mol = string2mol('C1CCC1')
    """
    try:
        assert format_ in ['smi']
    except:
        print(f'Your input format {format_} is not collect. We support smi')
    mol = readstring(format_, string)
    return [mol]

"""
========================== Convert mol to file ================================
"""
def mol2file(mols, outfile, format_=None, names=None, overwrite=True, size=None, 
             split=False, protein=False, addh_=False, make3d=False, 
             split_mode=False):
    """
    >>> mol2file(mols, 'example.sdf')
    mol = mols[0]
    mol2file(mols, '/home/zdx/project/MDZT-1003/SMS2/test/SMS2_00000.smina.sdf')
    
    The format of output file is sdf
    Saving 1 molecule(s) to example.sdf
    Success!
    """

    if protein:
        target = 'protein(s)'
    else:
        target = 'molecule(s)'
        
    if not isinstance(mols, list):
        mols = [mols]
    
    mols = [m for m in mols if m is not None]

    if format_ is None:
        informat = GetFileFormat(outfile)
    
    if informat == 'pdbqt':
        for mol in mols:
            try:
                mol.calccharges()
            except:
                mols.remove(mol)
    
    if addh_ or make3d:
        for mol in mols:
            try:
                if addh_:
                    mol.addh()
                if make3d:
                    mols.make3D()
            except:
                mols.remove(mol)
                
    try:
        print('The format of output file is', informat)
        print(f'Saving {len(mols)} {target} to {outfile}')
        assert informat in ['pdb', 'pdbqt', 'sdf', 'mol', 'mol2', 'smi', 'inchi']
    except:
        print('ERROR')
        print(f"Your output file format {format} is not allowed. We support "
              "pdb, pdbqt, sdf, mol, mol2, smi, inchi")
        return
    
    if names is not None:
        if len(names)!=len(mols):
            print('ERROR')
            print(f"Size of names isn't equal to size of {target}.")
            return
        else:
            for mol, name in tqdm(zip(mols, names)):
                mol.title = name
                
    if len(mols) == 0:
        print('ERROR')
        print(f'Threre is no {target} in the input mols list, please check.')
        return

    elif len(mols) == 1:
        mols[0].write(informat, outfile, overwrite=overwrite, size=size)
        # if informat == 'pdbqt' or informat == 'pdb':
        #     refinePDB(outfile)
            
    elif len(mols) > 1:
        if not split:
            writer = Outputfile(informat, outfile, overwrite=overwrite)
            for mol in tqdm(mols):
                writer.write(mol)
            writer.close()
        else:
            out_dir = dirname(outfile)
            file_name = GetFileName(outfile)
            out_dir = join(out_dir, file_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            if split_mode:
                mode_count = 0
            failed = []
            for i, mol in tqdm(enumerate(mols), total=len(mols)):
                
                if i == 0 and split_mode:
                    title = mol.title
                    new_title = '_'.join([mol.title, 'mode', str(mode_count)])
                    mode_count += 1
                    
                elif title != mol.title:
                    mode_count = 0
                    title = mol.title
                    new_title = '_'.join([mol.title, 'mode', str(mode_count)])
                    mode_count += 1

                else:
                    new_title = '_'.join([mol.title, 'mode', str(mode_count)])
                    mode_count += 1
                
                mol.title = new_title
                file = join(out_dir, '.'.join([mol.title, informat]))
                try:
                    mol.write(informat, file, overwrite=overwrite, size=size)
                    # if informat == 'pdbqt' or informat == 'pdb':
                    #     refinePDB(file)
                except:
                    failed.append(mol.title)
            print(failed)
    print('\nSuccess!\n')

def pdb2pdbqt(infiles, outfiles=None, protein=False, outformat='pdbqt'):
    if not isinstance(infiles, list):
        infiles = [infiles]
    
    if outfiles is not None and not isinstance(outfiles, list):
        outfiles = [outfiles]
        save_in_the_same_folder = False
    else:
        save_in_the_same_folder = True
        outfiles = []
        
    for i, infile in enumerate(infiles):
        mols = file2mol(infile, protein=protein)
        if mols[0] == None:
            continue
        if not save_in_the_same_folder:
            outfile = outfiles[i]
        else:
            outfile = ChaneFileSuffix(infile, outformat)
            outfiles.append(outfile)
        if not os.path.exists(outfile):
            try:
                mol2file(mols, outfile)
            except:
                os.system(f'obabel {infile} -O {outfile} --partialcharge')
        refinePDB(outfile)
    return outfiles

def same_folder_file(file, name, add=True):
    filename, fileformat = GetFileNameFormat(file)
    if add:
        return os.path.join(os.path.dirname(file),
                            f'{filename}_{name}.{fileformat}'
                            )
    else:
        return os.path.join(os.path.dirname(file),
                            f'{name}.{fileformat}'
                            )

def ispath(path):
    if isinstance(path, str):
        if os.path.exists(path):
            return True
    return False

def list1_in_list2(l1, l2):
    l1 = set(l1)
    l2 = set(l2)
    if len(l1) == len(l1 & l2):
        return True
    return False

def SaveDataFrame(df, outfile, index=False, sep=','):
    if outfile is None:
        return
    out_dir = os.path.dirname(outfile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(outfile, index=index, sep=sep)

def ReadDataFrame(file, sep=None, index_col=None, header=-1):
    """
    >>> df = ReadDataFrame(file)
    """
    if isinstance(file, pd.DataFrame):
       return file
    
    if not isinstance(file, str) or not os.path.exists(file):
        print('Input file not exists or not valid.')
    
    informat = GetFileFormat(file)
    
    if informat == 'smi':
        print('Input file is smi.')
        if sep is None:
            sep = ' '
        if header == -1:
            header = None
        df = pd.read_csv(file, sep=sep, header=header)
        # check the number of dataframe columns
        number_of_cols = df.shape[1]
        if number_of_cols == 1:
            df.columns = ['SMILES']
        elif number_of_cols == 2:
            df.columns = ['SMILES', 'ID']
            df = df[['ID', 'SMILES']]
    elif informat == 'xlsx':
        print('Input file is xlsx.')
        if index_col is not None:
            df = pd.read_excel(file, index_col=index_col)
        else:
            df = pd.read_excel(file)
    elif informat == 'csv':
        print('Input file is csv.')
        if sep is None:
            sep = ','
        if header == -1:
            header = 0
        if index_col is not None:
            df = pd.read_csv(file, index_col=index_col, sep=sep, header=header)
        else:
            df = pd.read_csv(file, sep=sep, header=header)
    else:
        if sep is not None and header != -1:
            if index_col is not None:
                df = pd.read_csv(file, index_col=index_col, sep=sep, 
                                 header=header)
            else:
                df = pd.read_csv(file, sep=sep, header=header)
        else:
            print('Only support smi, xlsx, csv, you should add sep= header=')
            return
        # with open(file, 'r') as f:
        #     line = f.readline().strip()
        # if '\t' in line:
        #     sep = '\t'
        # elif ',' in line:
        #     sep = ','
        # elif ' ' in line:
        #     sep = ' '
        # else:
        #     print('Can not recognize what the separator is. please provide it'
        #           ' by add argument "sep" ')
        #     return
    return df

def smi2csv(input_file, out_file=None, replace=False):
    df = ReadDataFrame(input_file)
    print('Load data.')
    print(df.head())
    if out_file is None:
        print('Not given output path, output file will saved in the same folder'
              ' of input file.')
        out_dir = os.path.dirname(input_file)
        filename = GetFileName(input_file)
        out_file = os.path.join(out_dir, f'{filename}.csv')
    if not replace:
        if os.path.exists(out_file):
            print('Outfile exists, if you want overwrite it, you should set'
                ' replace=True'
            )
            return
    print('Out:', out_file)
    print(df.head())
    df.to_csv(out_file, index=False)
    return df

def sdf2csv(file, outfile=None, idName='ID', smilesName='SMILES',removeHs=True):
    # file = '/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket69/docked_default_known_actives.sdf.gz'
    df = PandasTools.LoadSDF(file, idName=idName, smilesName=smilesName, 
        removeHs=removeHs)
    if None in df.columns:
        del df[None]
    if 'ROMol' in df.columns:
        del df['ROMol']
    if outfile is not None:
        df.to_csv(outfile, index=False)
    return df

def pdbqts2sdf(files, outfile):
    mols = file2mol(files)
    mol2file(mols, outfile)

def csv2smi(input_file, out_file=None, replace=False):
    """
    
    """
    if out_file is None:
        out_dir = os.path.dirname(input_file)
        filename = GetFileName(input_file)
        out_file = os.path.join(out_dir, f'{filename}.smi')
        
    if os.path.exists(out_file):
        return out_file
    
    informat = GetFileFormat(input_file)
    if informat != 'csv':
        print('Input file is not csv.')
        return
    df = pd.read_csv(input_file)
    print('Load data.')
    if not list1_in_list2(['SMILES', 'ID'], list(df.columns)):
        print('Input file must have SMILES and ID')
        return
    df = df[['SMILES', 'ID']]

    if not replace:
        if os.path.exists(out_file):
            print('Outfile exists, if you want overwrite it, you should set'
                ' replace=True'
            )
            return
    print('Out:', out_file)
    print('--------------')
    print('There are', len(df), 'molecules')
    print('--------------')
    df.to_csv(out_file, index=False, header=False, sep=' ')
    return out_file

def convet_between_smi_csv(input_file, out_file=None, replace=False):
    """
    
    """
    informat = GetFileFormat(input_file)
    print(f'Suffix of file is {informat}')
    if informat not in ['smi', 'csv']:
        print("File format is not 'smi' or 'csv', please check.")
    if informat == 'smi':
        smi2csv(input_file, out_file, replace)
    elif informat == 'csv':
        csv2smi(input_file, out_file, replace)

def verify_mol(mol, rep='SMILES'):
    if isinstance(mol, str):
        if rep=='SMILES':
            mol = Chem.MolFromSmiles(mol)
    if mol is None:
        print('input mol is invalid.')
        return None
    try:
        Chem.SanitizeMol(mol)
        inchi = MOL2InChI(mol)
        mol_re = InChI2MOL(inchi)
        inchi_re = MOL2InChI(mol_re)
        if inchi != inchi_re:
            print('input mol is invalid.')
            return None
    except:
        print('input mol is invalid.')
        return None
    print('input mol is valid.')
    return mol

# def autoFindsmile(df):
#     cols = df.columns
#     for col in cols:
#         if isinstance(col, str):
#             if 'C' in col or 'c' in col:
#                 s = df[col][0]
#                 mol = verify_mol(s)
#                 if mol is not None:
#                     print(mol)
#                     df.rename(columns={col:'SMILES'}, inplace=True)
#                     break
#     return df

"""
Remove duplicates molecules
"""
def remove_duplicate_mols(df, file=None, out=None, sep=None, header=None, 
                              col_names=None, inplace=False, save_out=False,
                              remain_InChI=False, donothing=False, 
                              smiles_col=None):
    """
    file = '/y/Aurora/Fernie/data/ligand_based_data/DUD-E/aa2ar.csv'
    """
    
    if smiles_col is not None:
        df.rename(columns={smiles_col:'SMILES'}, inplace=True)
    
    if 'SMILES' not in df.columns:
        print('The file must have have SMILES column.')
        return
    
    if 'label' in df.columns:
        df.sort_values(by='label', ascending = False, inplace=True)
    
    print('Before:', df.shape[0], 'moleculs.')
    
    if 'InChI' not in df.columns:
        df = SMILES2mol2InChI(df)
    else:
        print('Already have InChI')
    df.drop_duplicates('InChI', inplace=True)
    
    print('After drop duplicates:', df.shape[0], 'moleculs.')

    if not remain_InChI:
        del df['InChI']

    df.rename(columns={'SMILES':smiles_col}, inplace=True)
    if not save_out:
        return df

    else:
        if out is not None:
            df.to_csv(out, index=False)
        elif file is not None:
            if inplace:
                out = file
            else:
                file_dir = dirname(file)
                file_name, suffix = splitext(basename(file))
                file_name = file_name + '_drop_duplicates'
                out = join(file_dir, ''.join([file_name, suffix]))
        else:
            print('Not given a output path.')
            return
        if splitext(basename(out))[1] == '.smi':
            if 'InChI' in df.columns:
                del df['InChI']
        if splitext(basename(out))[1] == '.smi':
            header=False
        df.to_csv(out, index=False, header=header)
        if inplace:
            print(f'Success to replace raw input file {out}.\n')
        else:
            print(f'Success to save out to {out}.\n')


def drop_multi_files_duplicate_mols(files=None, input_dir=None, 
                                    output_dir = None,
                                    suffix=None, donothing=False,
                                    sep=None, header=None, 
                                    col_names=None, inplace=True, 
                                    save_out=True, remain_InChI=True):
    """
    input_dir = '/y/Aurora/Fernie/data/ligand_based_data/MUV'
    drop_multi_files_duplicate_mols(input_dir=input_dir, suffix='csv')
    """
    if suffix == 'smi':
        sep = ' '
        header = None
    
    if files is None and input_dir is not None:
        files = glob(join(input_dir, f'*.{suffix}'))
    else:
        print('Please check your input.')
        return
    dfs = []
    failed = []
    for file in tqdm(files):
        try:
            df = remove_duplicate_mols(file=file, sep=sep, header=header, 
                                  col_names=col_names, inplace=inplace, 
                                  save_out=save_out, remain_InChI=remain_InChI,
                                  donothing=donothing)
            dfs.append(df)
        except:
            failed.append(file)
        
    print(failed)
    return dfs

def merge_files(files=None, input_dir=None, out=None, suffix='csv', donothing=True):
    """
    input_dir = '/y/Aurora/Fernie/data/ligand_based_data/MUV'
    out = '/y/Aurora/Fernie/data/ligand_based_data/MUV.csv'
    merge_files(input_dir=input_dir, out=out)
    """
    if files is None and input_dir is not None:
        dfs = drop_multi_files_duplicate_mols(input_dir=input_dir, 
            suffix=suffix, donothing=donothing)
        df = pd.concat(dfs)
        df = remove_duplicate_mols(df=df, out=out)
        return df
    else:
        print('Please check your input.')
        return

"""
=========================== parallelize apply =================================
"""

def parallelize_dataframe(df, func, **kwargs):
    CPUs = multiprocessing.cpu_count()

    num_partitions = int(CPUs*0.8) # number of partitions to split dataframe
    num_cores = int(CPUs*0.8) # number of cores on your machine

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    func = partial(func, **kwargs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def judge_whether_is_dir(path):
    if not os.path.isdir(path):
        return False
    else:
        return True

def remain_path_of_dir(x):
    return list(filter(judge_whether_is_dir, x))

"""
-------------------------------------------------------------------------------
String to mol 
-------------------------------------------------------------------------------
"""
def InChI2MOL(inchi):
    try:
        mol = Chem.inchi.MolFromInchi(inchi)
        Chem.SanitizeMol(mol)
        if not mol == None:
            return mol
        else:
            return np.nan
    except:
        return np.nan
    
def RefineSMILES(smiles):
    try:
        if '.' in smiles:
            frags = smiles.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smiles = frags[ix]
        if smiles.count('C') + smiles.count('c') < 2:
            return np.nan
        return smiles
    except:
        return np.nan
    
def SMILES2MOL(smiles):
    try:
        if smiles.count('C') + smiles.count('c') < 2:
            return np.nan
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        if not mol == None:
            return mol
        else:
            return np.nan
    except:
        return np.nan

def remove_invalid_smiles(smiles_list):
    mols = list(map(SMILES2MOL, smiles_list))
    mols = filter(lambda x: str(x)!='nan', mols)
    return list(map(MOL2SMILES, mols))

"""
-------------------------------------------------------------------------------
Apply string to mol
-------------------------------------------------------------------------------
"""
def apply_refineSMILES(df):
    df['SMILES'] = df['SMILES'].apply(RefineSMILES)
    return df

def apply_SMILES2MOL(df):
    df['ROMol'] = df['SMILES'].apply(SMILES2MOL)
    return df

def apply_InChI2MOL(df):
    df['ROMol'] = df['InChI'].apply(InChI2MOL)
    return df

"""
-------------------------------------------------------------------------------
Mol to string
-------------------------------------------------------------------------------
"""
def toCanSmiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def add_can_smiles(df, smiles_col='SMILES', out_col='can_smiles', 
                   overwrite=False):
    
    if isinstance(df, str):
        df = pd.read_csv(df)
    
    smiles = df[smiles_col]
    
    can_smiles = []
    
    pre = smiles[0]
    for i, s in enumerate(tqdm(smiles)):
        if i != 0:
            if s == pre:
                can_smiles.append(can_smiles[i-1])
            else:
                can_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
                pre = s
        else:
            can_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
            pre = s
        
    if overwrite:
        df[smiles_col] = can_smiles
    else:
        df[out_col] = can_smiles
    return df

def MOL2SMILES(mol):
    try:
        # Chem.Kekulize(mol)
        sm = Chem.MolToSmiles(mol) # , kekuleSmiles=True
        return sm
    except:
        return np.nan

def MOL2InChI(mol):
    try:
        inchi, retcode, message, logs, aux = rdinchi.MolToInchi(mol)
        return inchi
    except:
        return np.nan

def MOL2ECFP4(mol, nbits=2048, radius=2, useFeatures=False):
    try:
        res = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, 
                                                    nBits=nbits, 
                                                    useFeatures=useFeatures)
        return np.array(res)
    except:
        return np.nan

"""
-------------------------------------------------------------------------------
apply mol to string
-------------------------------------------------------------------------------
"""
def apply_MOL2SMILES(df):
    df['SMILES'] = df.ROMol.apply(MOL2SMILES)
    return df

def apply_MOL2InChI(df):
    df['InChI'] = df.ROMol.apply(MOL2InChI)
    return df

def apply_mol2ECFP4(df, nbits=2048, radius=2, useFeatures=False):
    # mol2ECFP4V2 = partial(MOL2ECFP4, nbits=nbits, radius=radius)
    df['ECFP4'] = df['ROMol'].apply(MOL2ECFP4, nbits=nbits, radius=radius,
                                    useFeatures=useFeatures)
    return df

"""
-------------------------------------------------------------------------------
Parallel convert from one molecular respresentation to another.
-------------------------------------------------------------------------------
"""

def add_mol(df):
    if 'SMILES' in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
    elif 'InChI' in df.columns:
        df = parallelize_dataframe(df, apply_InChI2MOL)
    df.dropna(inplace=True)
    return df

def add_ECFP4(df, **kwargs):
    if 'ROMol' not in df.columns:
        df = add_mol(df)
    df = parallelize_dataframe(df, apply_mol2ECFP4, **kwargs)
    df.dropna(inplace=True)
    del df['ROMol']
    return df

def add_inchi(df):
    if 'ROMol' not in df.columns:
        df = add_mol(df)
    df = parallelize_dataframe(df, apply_MOL2InChI)
    del df['ROMol']
    return df
    
def InChI2mol2SMILES(df):
    if 'ROMol' not in df.columns:
        df = parallelize_dataframe(df, apply_InChI2MOL)
    df = parallelize_dataframe(df, apply_MOL2SMILES)
    del df['ROMol']
    return df

def SMILES2mol2InChI(df):
    if 'ROMol' not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
    df = parallelize_dataframe(df, apply_MOL2InChI)
    del df['ROMol']
    return df

"""
-------------------------------------------------------------------------------
Property functions
-------------------------------------------------------------------------------
"""

def judge_whether_has_rings_4(mol):
    r = mol.GetRingInfo()
    if len([x for x in r.AtomRings() if len(x)==4]) > 0:
        return False
    else:
        return True
    
def add_whether_have_4_rings(data):
    """4 rings"""
    data['4rings'] = data['ROMol'].apply(judge_whether_has_rings_4)
    return data

def four_rings_filter(df):
    df = parallelize_dataframe(df, add_whether_have_4_rings)
    df = df[df['4rings']==True]
    del df['4rings']
    return df

def analyze_logS(x):
    """
    logS = log (solubility measured in mol/l).
    >=0 highly soluble
    -2 to 0,  soluble
    −2 to −4, slightly soluble
    < -4, insoluble
    """
    if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
        if x >= 0:
            return 'highly soluble'
        elif -2 <= x and x <=0:
            return 'soluble'
        elif -4 <= x and x < -2:
            return 'slightly soluble'
        elif x < -4:
            return 'insoluble'
    else:
        return np.nan

def batch_analyze_logS(values):
    analysis = list(map(analyze_logS, values))
    return analysis

def LOGS(smiles):
    if isinstance(smiles, str):
        smiles = [smiles]
    else:
        smiles = list(smiles)
    preds = list(stn.predict(smiles))
    preds = [x[0] for x in preds]
    return preds

def add_solubility(df_or_file):
    if isinstance(df_or_file, str):
        informat = GetFileFormat(df_or_file)
        if informat == 'csv':
            df = pd.read_csv(df_or_file)
    elif isinstance(df_or_file, pd.DataFrame):
        df = df_or_file
    smiles = list(df.SMILES.values)
    df['logS'] = LOGS(smiles)
    df['Solubility'] = batch_analyze_logS(list(df.logS.values))
    return df

def AtomNumber(mol):
    try:
        res = mol.GetNumAtoms(onlyExplicit=False)
        return res
    except:
        return np.nan

def HeavyAtomNumber(mol):
    try:
        res = mol.GetNumHeavyAtoms()
        return res
    except:
        return np.nan
    
def BoronAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#5]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def CarbonAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan

    
def NitrogenAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def OxygenAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def FluorineAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#9]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def PhosphorusAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#15]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def SulfurAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def ChlorineAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#17]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan

def BromineAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#35]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def IodineAtomNumber(mol):
    try:
        res = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#53]"), 
                                          maxMatches=mol.GetNumAtoms()))
        return res
    except:
        return np.nan
    
def StereoCenters(mol):
    try:
        res = len(Chem.FindMolChiralCenters(mol,force=True,
                                            includeUnassigned=True))
        return res
    except:
        return np.nan

def AromaticRingsNumber(mol):
    try:
        res = Chem.Lipinski.NumAromaticRings(mol)
        return res
    except:
        return np.nan

def PositiveCharge(mol):
    # mol = Chem.AddHs(mol)
    try:
        positive_charge = 0
        for atom in mol.GetAtoms():
            charge = float(atom.GetFormalCharge())
            positive_charge += max(charge, 0)
        return positive_charge
    except:
        return np.nan

def NegativeCharge(mol):
    # mol = Chem.AddHs(mol)
    try:
        negative_charge = 0
        for atom in mol.GetAtoms():
            charge = float(atom.GetFormalCharge())
            negative_charge -= min(charge, 0)
        return negative_charge
    except:
        return np.nan

def FormalNetCharge(mol):
    # mol = Chem.AddHs(mol)
    try:
        res = Chem.rdmolops.GetFormalCharge(mol)
        return res
    except:
        return np.nan

def MW(mol):
    try:
        res = Chem.Descriptors.ExactMolWt(mol)
        return round(res,2)
    except:
        return np.nan

def HBA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBA(mol)
        return res
    except:
        return np.nan

def HBD(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBD(mol)
        return res
    except:
        return np.nan

def TPSA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcTPSA(mol)
        return round(res,2)
    except:
        return np.nan

def NRB(mol):
    try:
        res =  Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        return res
    except:
        return np.nan
    
def get_num_rings(mol):
    try:
        r = mol.GetRingInfo()
        res = len(r.AtomRings())
        return res
    except:
        return np.nan

def get_num_rings_6(mol):
    try:
        r = mol.GetRingInfo()
        res = len([x for x in r.AtomRings() if len(x) > 6])
        return res
    except:
        return np.nan

def LOGP(mol):
    try:
        res = logP(mol)
        return round(res,2)
    except:
        return np.nan
    
def MCF(mol):
    """
    Keep molecules whose MCF=True
    MCF=True means toxicity. but toxicity=True is not bad if the patient is dying.
    """
    try:
        res = mol_passes_filters(mol)
        return res
    except:
        return np.nan

def synthesis_availability(mol):
    """
    0-10. smaller, easier to synthezie.
    not very accurate.
    """
    try:
        res = SA(mol)
        return round(res,2)
    except:
        return np.nan
    
def estimation_drug_likeness(mol):
    """
    0-1. bigger is better.
    """
    try:
        res = QED(mol)
        return round(res,2)
    except:
        return np.nan

def get_scaffold_mol(mol):
    try: 
        res = GetScaffoldForMol(mol)
        return res
    except:
        return np.nan

def add_atomic_scaffold_mol(df):
    df['atomic_scaffold_mol'] = df.ROMol.apply(get_scaffold_mol)
    return df

def get_scaffold_inchi(mol):
    try: 
        scaffold_mol = GetScaffoldForMol(mol)
        inchi = MOL2InChI(scaffold_mol)
        return inchi
    except:
        return np.nan

def get_scaffold_smiles(mol):
    try: 
        scaffold_mol = GetScaffoldForMol(mol)
        smiles = MOL2SMILES(scaffold_mol)
        return smiles
    except:
        return np.nan

def add_scaffold_inchi(df):
    df['scaffold_inchi'] = df.ROMol.apply(get_scaffold_inchi)
    return df

def molSimilarity(s1, s2, radius=3, useFeatures=True, nBits=2048):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(SMILES2MOL(s1), 
                                                nBits=nBits,
                                                radius=radius, 
                                                useFeatures=useFeatures)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(SMILES2MOL(s2), 
                                                nBits=nBits,
                                                radius=radius,
                                                useFeatures=useFeatures)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def apply_molSimilarity(df, s2, smiles_col='SMILES', radius=3, nBits=2048, 
                        useFeatures=True, label='tanimoto'):
    molSimilarity_v = partial(molSimilarity, s2=s2, radius=radius, nBits=nBits, 
                              useFeatures=useFeatures)
    df[label] = df[smiles_col].apply(molSimilarity_v)
    return df

def parallel_molSimilarity(df, s2, smiles_col='SMILES', radius=3, nBits=2048, 
                           useFeatures=True, label='tanimoto'):
    apply_molSimilarity_v = partial(apply_molSimilarity, s2=s2,
                                    smiles_col=smiles_col,
                                    radius=radius, 
                                    nBits=nBits, 
                                    useFeatures=useFeatures,
                                    label=label)
    df = parallelize_dataframe(df, apply_molSimilarity_v)
    return df

"""
from rdkit import Chem
mol = Chem.MolFromSmiles('c1(c2nc(N=C(N)N)sc2)cn(c(c1)C)C')
res = molFeatures(mol)
"""
Descriptors = {
    'atom_n': AtomNumber, # 1
    'heavy_n': HeavyAtomNumber, # 2
    'boron_n': BoronAtomNumber, # 3
    'carbon_n': CarbonAtomNumber, # 4
    'nitrogen_n': NitrogenAtomNumber, # 5
    'oxygen_n': OxygenAtomNumber, # 6
    'fluorine_n': FluorineAtomNumber, # 7
    'phosphorus_n': PhosphorusAtomNumber, # 8
    'sulfur_n': SulfurAtomNumber, # 9
    'chlorine_n': ChlorineAtomNumber, # 10
    'bromine_n': BromineAtomNumber, # 11
    'iodine_n': IodineAtomNumber, # 12
    'logP': LOGP,  # 13
    'HBA': HBA, # 14
    'HBD': HBD, # 15
    'rings': get_num_rings, # 16
    'stereo_centers': StereoCenters, # 17
    'MW': MW, # 18
    'aromatic_rings': AromaticRingsNumber, # 19
    'NRB': NRB, # 20
    'pos_charge': PositiveCharge, # 21
    'neg_charge': NegativeCharge, # 22
    'formal_net_charge': FormalNetCharge, # 23
    'TPSA': TPSA, # 24
    'SA': synthesis_availability, # 25
    'QED': estimation_drug_likeness, # 26
    'MCF': MCF, # 27
    'scaffold_inchi': get_scaffold_inchi, # 28
    'scaffold_smiles': get_scaffold_smiles, # 29
    'InChI': MOL2InChI # 30
    }


def molFeatures(smiles_or_mol, pro=['MW', 'logP', 'TPSA', 'QED', 'SA', 'HBA',
                                    'HBD', 'NRB', 'rings', 'atom_n', 'heavy_n',
                                    'carbon_n', 'nitrogen_n', 'oxygen_n', 
                                    'fluorine_n', 'sulfur_n', 'chlorine_n',
                                    'bromine_n', 'stereo_centers', 
                                    'aromatic_rings']):
    
    # l = len(pro)
    # print("Feature number:", l)
    if isinstance(smiles_or_mol, str):
        mol = SMILES2MOL(smiles_or_mol)
        if str(mol)=='nan':
            return None, None
    else:
        mol = smiles_or_mol
        
    res = {}
    if not isinstance(pro, list):
        pro = [pro] 
    for p in pro:
        res[p] = Descriptors[p](mol)
    return res
    

def add_descriptors(df, pro=['MW', 'logP', 'HBA', 'HBD',
                             'TPSA', 'NRB', 'MCF', 'SA', 'QED',
                             'rings', 'scaffold_smiles']): 
    for p in pro:
        # if p not in ['atom_n', 'heavy_n', 'boron_n', 'carbon_n', 'nitrogen_n',
        #              'oxygen_n', 'fluorine_n', 'phosphorus_n', 'sulfur_n',
        #              'chlorine_n', 'bromine_n', 'iodine_n', 'logP', 'HBA', 
        #              'HBD', 'rings', 'stereo_centers', 'MW', 'aromatic_rings', 
        #              'NRB', 'pos_charge', 'neg_charge', 'formal_net_charge',
        #              'TPSA', 'SA', 'QED', 'MCF', 'scaffold_inchi', 
        #              'scaffold_smiles', 'InChI']:
        #     continue
        if p not in df.columns:
            df[p] = df.ROMol.apply(Descriptors[p])
    return df

def validity_filter(df):
    print("Start to remove invalid SMILES...")
    df = parallelize_dataframe(df, apply_refineSMILES)
    df.dropna(subset=['SMILES'], inplace=True)
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
    df.dropna(subset=['ROMol'], inplace=True)
    print("Finished.")
    return df

def remove_list1_from_list2(l1, l2):
    return [x for x in l2 if x not in l1]

def adjust_order(df, pro=
    ['ID', 'SMILES', 'smina', 'S', 'MW', 'logP', 'logS', 'Solubility', 'TPSA', 
     'QED', 'SA', 'HBA', 'HBD', 'NRB', 'MCF', 'rings', 'scaffold_smiles',
    'standard_relation', 'standard_value', 'standard_type', 'standard_units',
    ]):
    
    cols = [col for col in pro if col in df.columns] +\
        [col for col in df.columns if col not in pro]
    return df[cols]

def add_features(df, sort=True, remove_na=False, remove_mol=True,
                 order=True,
                 pro=['MW', 'logP', 'HBA', 'HBD', 'logS',
                      'TPSA', 'NRB', 'MCF', 'SA', 'QED',
                      'rings', 'scaffold_smiles'], smiles_col='SMILES'):

    if set(df.columns) >= set(pro):
        # print('Features already exist.')
        return df

    if smiles_col != 'SMILES':
        df.rename(columns={smiles_col: 'SMILES'}, inplace=True)

    if 'SMILES' not in df.columns:
        print("The column includes SMILES should be named 'SMILES'.")
        return
    
    if "ROMol" not in df.columns:
        df = validity_filter(df)

    if 'logS' in pro and 'logS' not in df.columns:
        df = add_solubility(df)
        pro.remove('logS')

    add_descriptors_v = partial(add_descriptors, pro=pro)
    df = parallelize_dataframe(df, add_descriptors_v)

    if remove_mol:
        del df['ROMol']
    if remove_na:
        df.dropna(subset=pro, inplace=True)
    if sort:
        if 'smina' in df.columns:
            df.sort_values('smina', inplace=True)
        elif 'S' in df.columns:
            df.sort_values('S', inplace=True)
        elif 'QED' in df.columns:
            df.sort_values('QED', ascending=False, inplace=True)
        elif 'standard_type' in df.columns and 'standard_value' in df.columns:
            df.sort_values(['standard_type', 'standard_value'], inplace=True)
    if order:
        df = adjust_order(df)
    if smiles_col != 'SMILES':
        return df.rename(columns={'SMILES': smiles_col})
    else:
        return df

def add_decoy_properties(df, sort=False, remove_na=True, 
                           remove_mol=True, order=False,
                           drop_dup = False,
                           pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
                                'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
                                'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
                                'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
                                'HBD', 'rings', 'stereo_centers', 'MW', # 4
                                'aromatic_rings', 'NRB', 'pos_charge', # 3
                                'neg_charge', 'formal_net_charge', 'TPSA', # 3
                                'SA', 'QED' # 2
                               ], smiles_col='SMILES', id_col='ID'):
    
    # if id_col is not None and id_col in df.columns:
    #     df.drop_duplicates(subset=[id_col], inplace=True)
    #     cols = [id_col] + pro
    #     df_pro = add_features(df, sort=sort, remove_na=remove_na, order=order,
    #         remove_mol=remove_mol, pro=pro, smiles_col=smiles_col)
    #     df_pro_merge = df[[id_col]].merge(df_pro, how='left', on=id_col)
    #     if remove_na:
    #         df_pro_merge.dropna(inplace=True)
    #     return df_pro_merge
    # else:
    if drop_dup:
        return add_features(df, sort=sort, remove_na=remove_na, order=order,
            remove_mol=remove_mol, 
            pro=pro, smiles_col=smiles_col).drop_duplicates(subset=[id_col])
    else:
        return add_features(df, sort=sort, remove_na=remove_na, order=order,
            remove_mol=remove_mol, pro=pro, smiles_col=smiles_col)

def property_filter(df, condition):
    """
    -----------------
    descriptor filter
    -----------------
    """
    print('descriptor filter')
    df = parallelize_dataframe(df, apply_SMILES2MOL)
    df.dropna(inplace=True)
    df = four_rings_filter(df)
    df = add_features(df)
    df[['MW', 'logP', 'TPSA', 'SA', 'QED']] = df[['MW', 'logP', 'TPSA', 'SA',\
        'QED']].apply(lambda x: round(x, 3))
    df = df.query(condition)
    df = df.reset_index(drop=True)
    return df

def fingerprint_similarity(line): 
    tanimoto = DataStructs.FingerprintSimilarity(line[0], line[1])
    return tanimoto

def molecule_in_patent(sample_fingerprint, patent_fingerprints, l, ths):
    fp_list = [sample_fingerprint] * int(l)
    matrix = pd.DataFrame({'SMILES':fp_list, 'patent':patent_fingerprints})
    matrix['tanimoto'] = matrix.apply(fingerprint_similarity, axis=1)
    if len(matrix.query('tanimoto%s' % ths)) > 0:
        return True
    else:
        return False

def add_patent(df, patent_fingerprints, l, ths):
    molecule_in_patentv2 = partial(molecule_in_patent, 
                                   patent_fingerprints=patent_fingerprints, 
                                   l=l, ths=ths)
    df['patent'] = df['ECFP4'].apply(molecule_in_patentv2)
    return df

def hard_patent_filter(df, patent, ths='==1'):
    """
    -------------------------------
    Remove molcules those in patent
    -------------------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
        df.dropna(inplace=True)
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, apply_SMILES2MOL)
        patent = patent.dropna()
    df = parallelize_dataframe(df, apply_mol2ECFP4)
    patent = parallelize_dataframe(patent, apply_mol2ECFP4)
    patent_fingerprints = patent.ECFP4
    l = len(patent_fingerprints)
    df = parallelize_dataframe(df, add_patent, 
                               patent_fingerprints=patent_fingerprints, 
                               l=l, ths=ths)
    df = df[df['patent']==False]
    del df['patent']
    df = df.reset_index(drop=True)
    del df['ECFP4']
    return df, patent

def soft_patent_filter(df, patent, ths='>0.85'):
    """
    -----------------------------------
    Remove molcules Tc > 0.85 in patent
    -----------------------------------
    """
    df, patent = hard_patent_filter(df, patent, ths)
    return df, patent

def substruct_match(df):
    # Chem.MolFromSmiles
    return df.ROMol.HasSubstructMatch(df.patent_scaffold_mol)

def add_substructure_match(matrix, outname='remain'):
    matrix[outname] = matrix.apply(substruct_match, axis=1)
    return matrix

def scaffold_in_patent(mol, patent_scaffolds, l):
    mol_list = [mol] * int(l)
    matrix = pd.DataFrame({'ROMol':mol_list, 
                           'patent_scaffold_mol':list(patent_scaffolds)})
    matrix = add_substructure_match(matrix)
    if len(matrix.query('remain==True')) > 0:
        return False
    else:
        return True 
    
def judge_substructure(df, col, patent_scaffolds, l, outname='remain'):
    scaffold_in_patentv2 = partial(scaffold_in_patent, 
                     patent_scaffolds=patent_scaffolds,
                     l=l)
    df[outname] = df[col].apply(scaffold_in_patentv2)
    return df

def atom_scaffold_filter(df, patent, col = 'atomic_scaffold_mol'):
    """
    ----------------------
    atomic scaffold filter
    ----------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
        df.dropna(inplace=True)
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, apply_SMILES2MOL)
        patent.dropna(inplace=True)
    if "atomic_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_atomic_scaffold_mol)
        df.dropna(inplace=True)
    if "atomic_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_atomic_scaffold_mol)
        patent.dropna(inplace=True)
    patent_scaffolds = set(patent[col])
    l = len(patent_scaffolds)
    df = parallelize_dataframe(df, judge_substructure, 
                               col = col,
                               patent_scaffolds=patent_scaffolds, l=l)
    df = df[df['remain']==True]
    del df['remain']
    df = df.reset_index(drop=True)
    return df, patent

def get_graph_scaffold_mol(atomic_scaffold_mol):
    try:
        #atomic_scaffold_mol.Compute2DCoords()
        graph_scaffold_mol = MurckoScaffold.MakeScaffoldGeneric( 
            atomic_scaffold_mol)
        return graph_scaffold_mol
    except:
        return np.nan
    
def add_graph_scaffold(df, col='atomic_scaffold_mol', 
                       outname='graph_scaffold_mol'):
    df[outname] = df[col].apply(get_graph_scaffold_mol)
    return df

def grap_scaffold_filter(df, patent, col='graph_scaffold_mol'):
    """
    ----------------------
    graph scaffold filter
    ----------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, apply_SMILES2MOL)
        df.dropna(inplace=True)
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, apply_SMILES2MOL)
        patent.dropna(inplace=True)
    if "atomic_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_atomic_scaffold_mol)
        df.dropna(inplace=True)
    if "atomic_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_atomic_scaffold_mol)
        patent.dropna(inplace=True)
    if "graph_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_graph_scaffold)
        df.dropna(inplace=True)
    if "graph_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_graph_scaffold)
        patent.dropna(inplace=True)
    df, patent = atom_scaffold_filter(df, patent, col = col)
    return df, patent

def atomic_scaffold_smiles(df):
    if "atomic_scaffold_smiles" not in df.columns:
        df["atomic_scaffold_smiles"] = df["atomic_scaffold_mol"].apply(MOL2SMILES)
    return df

def graph_scaffold(df):
    if "graph_scaffold" not in df.columns:
        df["graph_scaffold"] = df["graph_scaffold_mol"].apply(MOL2SMILES)
    return df

def save_file(df, path):
    if "ROMol" in df.columns:
        del df["ROMol"]
    if "atomic_scaffold_mol" in df.columns:
        df = parallelize_dataframe(df, atomic_scaffold_smiles)
        del df['atomic_scaffold_mol']
    if "graph_scaffold_mol" in df.columns:
        df = parallelize_dataframe(df, graph_scaffold)
        del df["graph_scaffold_mol"]
    df.to_csv(path, index=False)


def filter_molecule(input_file, output_dir, condition_file, patent_file):
    try:
        with open(condition_file, 'r') as f:
            condition = f.readline()
            condition = condition.strip()
    except:
        print("Read condition file failed.")
        return
    try:
        df = pd.read_csv(input_file)
    except:
        print("Read compound file failed.")
        return
    try:
        patent = pd.read_csv(patent_file)
    except:
        print("Read patent file failed.")
        return
    df = property_filter(df, condition)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_file(df, os.path.join(output_dir, "property_filter.csv"))
    df, patent = hard_patent_filter(df, patent)
    save_file(df, os.path.join(output_dir, "hard_pat_filter.csv"))
    df, patent = soft_patent_filter(df, patent)
    save_file(df, os.path.join(output_dir, "soft_pat_filter.csv"))
    df, patent = atom_scaffold_filter(df, patent)
    save_file(df, os.path.join(output_dir, "atom_pat_filter.csv"))
    df, patent = grap_scaffold_filter(df, patent)
    save_file(df, os.path.join(output_dir, "grap_pat_filter.csv"))

"""
-------------------
Protein
-------------------
"""
def get_fasta_ids(file):
    """
    >>> names = get_fasta_ids(file)
    """

    with open(file, 'r') as f:
        lines = f.readlines()
    names = []
    for line in lines:
        line = line.strip()
        if '>' in line:
            names.append(line.replace('>', ''))
    print(f'There are {len(names)} proteins.')
    return names

def seq2fasta(seqs, names, out):
    """
    >>> seq2fasta(seqs, names, out)
    """
    with open(out, 'w') as f:
        for name, seq in zip(names, seqs):
            print(f'>{name}', file=f)
            print(seq, file=f)
    print(f'Succeed to write {len(names)} proteins into {out}')

def df2fasta(df, out, seq_col, name_col):
    seqs = list(df[seq_col].values)
    names = list(df[name_col].values)
    seq2fasta(seqs, names, out)

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in tqdm(os.listdir(src)):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def delete_a_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

def delete_a_folder(dir_path):
    try:
        if os.path.exists(dir_path):
            rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

def build_new_folder(dir_path, overwrite=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        if overwrite:
            delete_a_folder(dir_path)
            os.makedirs(dir_path)
        
def display_dict(dict_):
    for key,value in dict_.items():
        print(key, ":", value)

def try_copy(source, target):
    try:
       copy(source, target)
    except IOError as e:
       print("Unable to copy file. %s" % e)
       exit(1)
    except:
       print("Unexpected error:", sys.exc_info())
       exit(1)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def saveDictJson(dict_, outfile):
    with open(outfile, "w") as outfile:
        json.dump(dict_, outfile, cls=NpEncoder, indent=4)
        
def readJson(json_file):
    with open(json_file) as f:
       data = json.load(f)
    return data