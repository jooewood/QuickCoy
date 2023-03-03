#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:23:09 2021

@author: zdx
"""
import matplotlib
# matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
import seaborn as sns
from utils import add_features, adjust_order, CountNonZeroDataFrame, \
    build_new_folder, saveDictJson
import matplotlib.pyplot as plt



def vioCompare(df, x_col, y_col, hue, palette="muted", figsize=(10,8),
               dpi=300, outfile=None, ylim=None, plot_type='box'):
    
    plt.figure(figsize=figsize, dpi=dpi)
    if plot_type == 'vio':
        g = sns.violinplot(x=x_col, y=y_col, hue=hue,
                       data=df, palette=palette)
    if plot_type == 'box':
        g = sns.boxplot(x=x_col, y=y_col, hue=hue,
                       data=df, palette=palette)
        
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if ylim is not None:
        plt.ylim(ylim)
    if outfile is not None:
        plt.savefig(outfile, dpi=dpi, format='png')
    else:
        plt.show()

def equal_list(l1, l2):
    interset = set(l1) & set(l2)
    if len(interset) == len(set(l2)):
        return True
    else:
        return False

def reshape_df(raw, columns, x_col='x', y_col='y', type_=None, 
               type_col=None):
    
    exclude_cols = list(set(raw.columns) - set(columns))
    
    for i, col in enumerate(columns):
        # i = 0
        # col = 'AUC_ROC'
        if i==0:
            df = pd.DataFrame({
                y_col : raw[col].values,
                x_col:[col] * len(raw)
                })
            df = pd.concat([df, raw[exclude_cols].reset_index(drop=True)], axis=1)
        else:
            tmp = pd.DataFrame({
                y_col: raw[col].values,
                x_col:[col] * len(raw)
                })
            tmp = pd.concat([tmp, raw[exclude_cols].reset_index(drop=True)], axis=1)
            df = pd.concat([df, tmp], sort=False)
    if type_ is not None:
        df[type_col] = type_
    return df

def df_feature_comparison(df_left, df_right=None, columns=None, output_file=None,  
                          title=None, type_col='data',
                          left_name="Real", right_name="AI Design", 
                          draweps=False, figsize=(20, 8), dpi=300):


    if 'S' in df_left.columns and 'S' in df_right:
        columns.append('S')

    if not equal_list(df_left.columns, columns): 
        df_left_add_feature = add_features(df_left)
    else:
        df_left_add_feature = adjust_order(df_left)
    if not equal_list(df_right.columns, columns): 
        df_right_add_feature = add_features(df_right)
    else:
        df_right_add_feature = adjust_order(df_right)

    df_left = df_left_add_feature[columns]
    df_right = df_right_add_feature[columns]
    
    df_left_re = reshape_df(df_left, columns, type_=left_name, type_col=type_col)
    df_right_re = reshape_df(df_right, columns, type_=right_name, type_col=type_col)

    df = pd.concat([df_left_re, df_right_re], sort=False)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(1, len(columns))
    for i,col in enumerate(columns):
        tmp = df.query('x=="%s"' % col)
        g = sns.violinplot(x="x", y="y", data=tmp, hue="data", 
                           hue_order=[left_name, right_name],
                       split=True, palette="Set3", ax=axes[i])
        if i==0:
            handles, labels = axes[i].get_legend_handles_labels()
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].legend('')
    if title is not None:
        plt.title(title)
    g.legend(handles, labels, loc='best')
    # fig.legend(handles, labels, loc='top right', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    
    if output_file is not None:
        out_dir = os.path.dirname(output_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if draweps:
            if '.eps' not in output_file:
                output_file += '.eps'
            fig.savefig(output_file, dpi=dpi, format='eps')
        
        if '.png' not in output_file:
            if '.eps' in output_file:
                output_file = output_file.replace('.eps', '')
            output_file += '.png'
        fig.savefig(output_file, dpi=dpi, format='png')
    else:
        plt.show()
    return df_left_add_feature, df_right_add_feature

def DataFrameDistribution(df, drop=None, columns=None, outfile=None,
                          figsize=(20, 8), dpi=300, plot_type='vio'):
    if drop is None and columns is None:
        columns = df.columns
        df_pro = df
    else:   
        if drop is not None:
            columns = [x for x in df.columns if x not in drop]
        df_pro = df[columns]
        
    print(columns)
        
    colors = sns.color_palette("Paired")
    l = len(colors)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(1, len(columns))
    for i,column in enumerate(columns):
        color = colors[(i+1)%l]
        if plot_type == 'vio':
            sns.violinplot(y=df_pro[column], ax=axes[i], color=color)
        elif plot_type == 'box':
            sns.boxplot(y=df_pro[column], ax=axes[i], color=color)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('')
        axes[i].legend('')
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=dpi, format='png')
    else:
        plt.show()
    return df_pro.describe()

def xDistribution(x, bins=10, density=False, save_path=None, 
                  title=None, dpi=300, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.hist(x, bins=bins, density=density)
    if title is not None:
        plt.title(title)
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=dpi, format='png')
    
def BarPlot(labels, values, mode='h', outfile=None, figsize=(10,10), xlabel=None, 
            ylabel=None, title=None, dpi=300, log=True):
    
    labels = np.array(list(labels))
    values = np.array(list(values))
    
    fig, ax = plt.subplots(figsize =figsize)
    if mode == 'h':
        bars = plt.barh(labels, values, log=log)
        for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}')
            
    elif mode == 'v':
        bars = plt.bar(labels, values, log=log)
        for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.tight_layout()
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile, dpi=dpi, format='png')

def DfNonZero(df, out_dir=None, drop=None, columns=None, figsize1=(10,10), 
              figsize2=(20,10), only=False, dpi=300):
    res = CountNonZeroDataFrame(df, drop=drop)
    
    if only:
        return res
    if out_dir != None:
        build_new_folder(out_dir)
        barplot_path = os.path.join(out_dir, 'barplot.png')
        dis_path = os.path.join(out_dir, 'distribution.png')
        saveDictJson(res, os.path.join(out_dir, 'ZeroCount.json'))
    else:
        barplot_path = None
        dis_path = None
    BarPlot(res.keys(), res.values(), figsize=figsize1, outfile=barplot_path,
            dpi=dpi)
    # Distribution
    DataFrameDistribution(df, drop=drop, columns=columns, figsize=figsize2,
                          outfile=dis_path, dpi=300)
    return res

def mol_file_properties_comparison(left_path, right_path, output_file, title=None, 
        extra=None,
        columns = ['MW', 'logP', 'logS', 'SA', 'QED', 'TPSA', 'HBA', 'HBD', 
                   'NRB', 'rings'],
        left_name="Real", right_name="AI Design", 
        draweps=False):
    
    df_left = pd.read_csv(left_path)
    df_right = pd.read_csv(right_path)
    df_left, df_right = df_feature_comparison(df_left=df_left, 
                          df_right=df_right, columns=columns, 
                          title=title, output_file=output_file,
                          left_name=left_name, right_name=right_name,
                          draweps=draweps
                          )
    df_left.to_csv(left_path, index=False)
    df_right.to_csv(right_path, index=False)
