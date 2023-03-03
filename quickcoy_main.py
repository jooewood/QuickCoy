#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 19:02:04 2022

@author: zdx
"""

import os
import re
import time
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
import random
from abc import ABC, abstractmethod
from collections import UserList, defaultdict
import math
from utils import add_features, remove_invalid_smiles,\
    saveVariable, loadVariable, molFeatures, ReadDataFrame, build_new_folder,\
    GetFileName, add_decoy_properties
from plot import DataFrameDistribution
import argparse
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_model_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_model.pt'
    )
def get_log_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_log.csv'
    )
def get_config_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_config.pt'
    )
def get_vocab_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_vocab.pt'
    )
def get_generation_path(config, model):
    return os.path.join(
        config.checkpoint_dir,
        model + config.experiment_suff + '_generated.csv'
    )

def get_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default=None)
    parser.add_argument('--generation_dir', default=None)
    parser.add_argument('--active_id_col', default='active_id')
    parser.add_argument('--active_smiles_col', default='active')
    parser.add_argument('--smiles_col',  default='SMILES', type=str)
    parser.add_argument('--property_num', default=None, type=int)
    parser.add_argument('--id_col', default='ID', type=str)
    parser.add_argument('--pro_cols', 
        default=['MW', 'logP', 'TPSA', 'QED', 'SA', 'HBA', 'HBD', 'NRB',
                 'rings', 'atom_n', 'heavy_n', 'carbon_n', 'nitrogen_n',
                 'oxygen_n', 'fluorine_n', 'sulfur_n', 'chlorine_n',
                 'bromine_n', 'stereo_centers', 'aromatic_rings'], nargs='*')
    parser.add_argument('--init_decoy_n', default=1024, type=int)
    parser.add_argument('--max_len', default=120, type=int)
    parser.add_argument('--test_load', default=None,
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path',
                        type=str, required=False,
                        help='Path to scaffold test molecules csv')
    parser.add_argument('--ptest_path',
                        type=str, required=False,
                        help='Path to precalculated test npz')
    parser.add_argument('--ptest_scaffolds_path',
                        type=str, required=False,
                        help='Path to precalculated scaffold test npz')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for checkpoints')
    parser.add_argument('--n_samples', type=int, default=30000,
                        help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device index in form `cuda:N` (or `cpu`)')
    parser.add_argument('--metrics', type=str, default='metrics.csv',
                        help='Path to output file with metrics')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training dataset')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of testing dataset')
    parser.add_argument('--experiment_suff', type=str, default='',
                        help='Experiment suffix to break ambiguity')


    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')

    parser.add_argument('--train_load',
                            type=str,
                            help='Input data in csv format to train')
    parser.add_argument('--val_load', type=str,
                            help="Input data in csv format to validation")
    parser.add_argument('--model_save',
                            type=str, default=None,
                            help='Where to save the model')
    parser.add_argument('--save_frequency',
                            type=int, default=20,
                            help='How often to save the model')
    parser.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    parser.add_argument('--config_save',
                            type=str, required=False,
                            help='Where to save the config')
    parser.add_argument('--vocab_save',
                            type=str, required=False,
                            help='Where to save the vocab')
    parser.add_argument('--vocab_load',
                            type=str, required=False,
                            help='Where to load the vocab; '
                                 'otherwise it will be evaluated')
    # Model
    parser.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    parser.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    parser.add_argument('--q_d_h',
                           type=int, default=256,
                           help='Encoder h dimensionality')
    parser.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    parser.add_argument('--q_dropout',
                           type=float, default=0.5,
                           help='Encoder layers dropout')
    parser.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    parser.add_argument('--d_n_layers',
                           type=int, default=3,
                           help='Decoder number of layers')
    parser.add_argument('--d_dropout',
                           type=float, default=0.2,
                           help='Decoder layers dropout')
    parser.add_argument('--d_z',
                           type=int, default=128,
                           help='Latent vector dimensionality')
    parser.add_argument('--d_d_h',
                           type=int, default=512,
                           help='Decoder hidden dimensionality')
    parser.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')

    # Train
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--n_batch',
                           type=int, default=128,
                           help='Batch size')
    parser.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    parser.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    parser.add_argument('--kl_w_start',
                           type=float, default=0,
                           help='Initial kl weight value')
    parser.add_argument('--kl_w_end',
                           type=float, default=0.05,
                           help='Maximum kl weight value')
    parser.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    parser.add_argument('--lr_n_period',
                           type=int, default=10,
                           help='Epochs before first restart in SGDR')
    parser.add_argument('--lr_n_restarts',
                           type=int, default=10,
                           help='Number of restarts in SGDR')
    parser.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    parser.add_argument('--lr_end',
                           type=float, default=3 * 1e-4,
                           help='Maximum lr weight value')
    parser.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    parser.add_argument('--n_workers',
                           type=int, default=12,
                           help='Number of workers for DataLoaders') 
    return parser


class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        if isinstance(key, slice):
            return Logger(self.data[key])
        ldata = self.sdata[key]
        if isinstance(ldata[0], dict):
            return Logger(ldata)
        return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)

    def save(self, path):
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)

class KLAnnealer:
    def __init__(self, n_epoch, config):
        self.i_start = config.kl_start
        self.w_start = config.kl_w_start
        self.w_max = config.kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc

class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer, config):
        self.n_period = config.lr_n_period
        self.n_mult = config.lr_n_mult
        self.lr_end = config.lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end

class Trainer(ABC):
    @property
    def n_workers(self):
        n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model):
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device

    def get_dataloader(self, dataset, collate_fn=None, shuffle=True):
        if collate_fn is None:
            collate_fn = dataset.default_collate
        return DataLoader(dataset, batch_size=self.config.n_batch,
                          shuffle=shuffle,
                          num_workers=self.n_workers, collate_fn=collate_fn,
                          worker_init_fn=set_torch_seed_to_all_gens
                          if self.n_workers > 0 else None)

    def get_collate_fn(self, model):
        return None

    @abstractmethod
    def get_vocabulary(self, data):
        pass

    @abstractmethod
    def fit(self, model, train_data, val_data=None):
        pass

class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        if self.size > 0:
            return self.data[:self.size].mean()
        return 0.0

class VAETrainer(Trainer):
    def __init__(self, config):
        self.config = config
        self.config_epoch = config.epoch

    def get_vocabulary(self, data):
        return OneHotVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            return tensors

        return collate

    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        for x, c, l, data in tqdm_data:
            c = c.to(model.device)
            x = x.t()
            x = tuple(data.to(model.device) for data in x)
            # Forward
            kl_loss, recon_loss = model(x, c)
            loss = kl_weight * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model),
                                self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'klw={kl_weight:.5f} lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix

    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device
        n_epoch = self._n_epoch()
        print('Total epoch:', n_epoch)

        optimizer = optim.Adam(self.get_optim_params(model),
                               lr=self.config.lr_start)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer,
                                                   self.config)

        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            
            kl_weight = kl_annealer(epoch)
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch,
                                        tqdm_data, kl_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            # New =============================================================
            # if epoch == 0:
            #     best_kl_loss = postfix['kl_loss']
            #     best_recon_loss = postfix['recon_loss']
            # else:
            #     if postfix['kl_loss'] < best_kl_loss and \
            #         postfix['recon_loss'] < best_recon_loss:
            #             if self.config.model_save is not None:
            print('Save checkpoint.')
            model = model.to('cpu')
            torch.save(model.state_dict(),
                        self.config.model_save[:-3] +
                        '_{0:03d}.pt'.format(epoch))
            model = model.to(device)
            # =================================================================
            # if (self.config.model_save is not None) and \
            #         (epoch % self.config.save_frequency == 0):
            # if postfix
            #     model = model.to('cpu')
            #     torch.save(model.state_dict(),
            #                self.config.model_save[:-3] +
            #                '_{0:03d}.pt'.format(epoch))
            #     model = model.to(device)

            # Epoch end
            lr_annealer.step()

    def fit(self, model, X_train, y_train, X_test=None, y_test=None):
        logger = Logger() if self.config.log_file is not None else None
        train_dataset = StringDataset(model.vocabulary, X_train, y_train)
        train_loader = self.get_dataloader(train_dataset, shuffle=True)
        if X_test is not None and y_test is not None:
            val_dataset = StringDataset(model.vocabulary, X_test, y_test)
            val_loader = self.get_dataloader(val_dataset, shuffle=False)
        else:
            val_loader = None

        self._train(model, train_loader, val_loader, logger)
        return model

    def _n_epoch(self):
        return min(self.config_epoch, sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        ))

def set_torch_seed_to_all_gens(_):
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)

def get_dataset(path, smiles_col='SMILES', id_col='ID',pro_cols=None,frac=0.9,
                random_state=0, split=True):
    """
    path = '/home/zdx/data/decoy_generation/smiles_MW_logP_HD_HA_TPSA.csv'
    
    Loads dataset

    Arguments:
        smiles_col (str): 
        pro_cols (list): 

    Returns:
        list with SMILES strings
        list with labels 
    """
    df = pd.read_csv(path)
    if pro_cols is None:
        pro_cols = [x for x in df.columns if x not in [smiles_col, id_col]]
    property_num = len(pro_cols)
    
    if split:
        all_smiles = df[smiles_col].values
        
        train = df.sample(frac=frac, random_state=random_state)
        test = df.drop(train.index)
    
        X_train = train[smiles_col].values
        y_train = train[pro_cols].values
        X_test = test[smiles_col].values
        y_test = test[pro_cols].values
        return X_train, y_train, X_test, y_test, property_num, all_smiles
    
    else:
        X_train = df[smiles_col].values
        y_train = df[pro_cols].values
        return X_train, y_train, property_num

class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

_atoms = ['Cl', 'Br', 'Na', 'Si', 'Li', 'Se', 'Mg', 'Zn']

def get_tokenizer_re(atoms):
    return re.compile('('+'|'.join(atoms)+r'|\%\d\d|.)')

_atoms_re = get_tokenizer_re(_atoms)

def smiles_tokenizer(line, atoms=None):
    """
    Tokenizes SMILES string atom-wise using regular expressions. While this
    method is fast, it may lead to some mistakes: Sn may be considered as Tin
    or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
    specify a set of two-letter atoms explicitly.

    Parameters:
         atoms: set of two-letter atoms for tokenization
    line = s
    """
    try:
        if atoms is not None:
            reg = get_tokenizer_re(atoms)
        else:
            reg = _atoms_re
        return reg.split(line)[1::2]
    except:
        reg = None
        return reg

class CharVocab:
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            tokenlized_string = smiles_tokenizer(string)
            if tokenlized_string is not None:
                chars.update(tokenlized_string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=True, add_eos=True):
        ids = [self.char2id(c) for c in smiles_tokenizer(string)]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string
    
class OneHotVocab(CharVocab):
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))

class StringDataset:
    def __init__(self, vocab, data, y):
        """
        Creates a convenient Dataset with SMILES tokinization

        Arguments:
            vocab: CharVocab instance for tokenization
            data (list): SMILES strings for the dataset
        """
        self.vocab = vocab
        self.tokens = [vocab.string2ids(s) for s in data]
        self.data = data
        self.y = y
        self.bos = vocab.bos
        self.eos = vocab.eos

    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return len(self.tokens)

    def __getitem__(self, index):
        """
        Prepares torch tensors with a given SMILES.

        Arguments:
            index (int): index of SMILES in the original dataset

        Returns:
            A tuple (with_bos, with_eos, smiles), where
            * with_bos is a torch.long tensor of SMILES tokens with
                BOS (beginning of a sentence) token
            * with_eos is a torch.long tensor of SMILES tokens with
                EOS (end of a sentence) token
            * smiles is an original SMILES from the dataset
        """
        return torch.tensor(self.tokens[index], dtype=torch.long),\
               self.y[index],\
               self.data[index]

    def default_collate(self, batch, return_data=False):
        """
        Simple collate function for SMILES dataset. Joins a
        batch of objects from StringDataset into a batch

        Arguments:
            batch: list of objects from StringDataset
            pad: padding symbol, usually equals to vocab.pad
            return_data: if True, will return SMILES used in a batch

        Returns:
            with_bos, with_eos, lengths [, data] where
            * with_bos: padded sequence with BOS in the beginning
            * with_eos: padded sequence with EOS in the end
            * lengths: array with SMILES lengths in the batch
            * data: SMILES in the batch

        Note: output batch is sorted with respect to SMILES lengths in
            decreasing order, since this is a default format for torch
            RNN implementations
        """
        tokens, y, data = list(zip(*batch))
        # Get lengths
        lengths = [len(x) for x in tokens]
        # Get order
        order = np.argsort(lengths)[::-1]
        # Sort
        lengths = [lengths[i] for i in order]
        y = [y[i] for i in order]
        tokens = [tokens[i] for i in order]
        # padding
        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, padding_value=self.vocab.pad
        )
        data = np.array(data)[order]
        return tokens, torch.FloatTensor(y), lengths, data

class VAE(nn.Module):
    def __init__(self, vocab, config, property_num):
        super().__init__()
        
        self.property_num = property_num
        
        self.vocabulary = vocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if config.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last+self.property_num, config.d_z)
        self.q_logvar = nn.Linear(q_d_last+self.property_num, config.d_z)

        # Decoder
        if config.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + config.d_z + self.property_num,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )
        
        self.decoder_lat = nn.Linear(config.d_z+self.property_num, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def forward(self, x, c):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x, c)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z, c)

        return kl_loss, recon_loss

    def forward_encoder(self, x, c):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        c = c.view(-1, self.property_num)
        h = torch.cat((h, c), dim=1) 
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x, z, c):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z = torch.cat((z, c), dim=1)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )

        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def sample(self, c, max_len=120, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        c = y_test
        
        """
        if not torch.is_tensor(c):
            c = torch.FloatTensor(c)
            
        c = c.to(self.device)
        
        n_batch = c.size()[0]
        # print('Batch:', n_batch)
        
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z = torch.cat((z, c), dim=1)
            
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=self.device)
            
            
            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            generated = remove_invalid_smiles(list(map(self.tensor2string, 
                                                        new_x)))
            return generated

def GenDecoy(active, model, active_id=None, init_decoy_n=1000, n_batch=1024, 
             max_len=120, active_id_col='active_id', max_n_batch=10,
             active_smiles_col='active',
                  pro=['MW', 'logP', 'TPSA', 'QED', 'SA', 'HBA', 'HBD', 'NRB',
                       'rings', 'atom_n', 'heavy_n', 'carbon_n', 'nitrogen_n',
                       'oxygen_n', 'fluorine_n', 'sulfur_n', 'chlorine_n',
                       'bromine_n', 'stereo_centers', 'aromatic_rings']):
    s = time.time()
    if isinstance(active, list):
        print('Do not accept a list of SMILES.')
        return None
    _, c = molFeatures(active, pro=pro)
    if c is None:
        print('Invalid active SMILES.')
        return None
    c = np.repeat([c], n_batch, axis=0)
    generated = set()
    count = 0
    while len(generated) < init_decoy_n:
        generated = generated|set(model.sample(c, max_len=max_len))
        count += 1
        if count > max_n_batch:
            break
    e = time.time()
    label = pd.DataFrame([c[0]])
    label.columns = pro
    decoy_n = len(generated)
    if active_id is None:
        df = pd.DataFrame({active_smiles_col:active,'decoy':generated})
    else:
        df = pd.DataFrame({active_id_col:active_id, active_smiles_col:active,
            'decoy':list(generated)})
    res = {}
    res['generated_df'] = df
    res['decoy_n'] = decoy_n
    res['validity'] = round((decoy_n/(n_batch*count)), 2)
    res['time_cost'] = round((e-s), 2)
    res['condition_label'] = label.rename(index={0:'label'})
    res['init_gen_n'] = n_batch * count
    return res

def batchGenDecoy(df_or_file, model, outfile=None, target_name=None, 
                  active_id_col='active_id', 
                  active_smiles_col='active', 
                  init_decoy_n=1000, n_batch=1024, max_len=120, 
                  pro=['MW', 'logP', 'TPSA', 'QED', 'SA', 'HBA', 'HBD', 'NRB',
                       'rings', 'atom_n', 'heavy_n', 'carbon_n', 'nitrogen_n',
                       'oxygen_n', 'fluorine_n', 'sulfur_n', 'chlorine_n',
                       'bromine_n', 'stereo_centers', 'aromatic_rings']):
    
    active_df = ReadDataFrame(df_or_file)
    active_df = active_df[[active_id_col, active_smiles_col]]
    active_df.drop_duplicates(subset=active_id_col, inplace=True)
    active_n = len(active_df)
    
    gen_decoy_dfs = []
    decoy_lens = []
    validitys = []
    time_costs = [] 
    init_gen_ns = []
    for id_, active in tqdm(zip(active_df[active_id_col].values, 
                           active_df[active_smiles_col].values), 
                            total=len(active_df[active_id_col].values)):
        try:
            res_sub = GenDecoy(active=active, model=model, active_id=id_, 
                init_decoy_n=init_decoy_n, n_batch=n_batch, max_len=max_len,
                pro=pro, active_id_col=active_id_col, 
                active_smiles_col=active_smiles_col)
            if res_sub is not None: 
                gen_decoy_dfs.append(res_sub['generated_df'])
                decoy_lens.append(res_sub['decoy_n'])
                validitys.append(res_sub['validity'])
                time_costs.append(res_sub['time_cost'])
                init_gen_ns.append(res_sub['init_gen_n'])
            else:
                decoy_lens.append(np.nan)
                validitys.append(np.nan)
                time_costs.append(np.nan)
                init_gen_ns.append(np.nan)
        except:
            decoy_lens.append(np.nan)
            validitys.append(np.nan)
            time_costs.append(np.nan)
            init_gen_ns.append(np.nan)
    res = pd.concat(gen_decoy_dfs)
    decoy_n = len(res)
    summary_df = active_df
    summary_df['decoy_count'] = decoy_lens
    summary_df['validity'] = validitys
    summary_df['time_cost'] = time_costs
    summary_df['init_gen_n'] = init_gen_ns
    if target_name is not None:
        summary_df['target'] = target_name
        
    summary_df = add_decoy_properties(summary_df, order=False,
                                        smiles_col=active_smiles_col)
        
    if outfile is not None:
        out_dir = os.path.dirname(outfile)
        filename = GetFileName(outfile)
        res.to_csv(outfile, index=False)
        summary_df.to_csv(
            os.path.join(out_dir, f'{filename}.quickcoy.summary.csv'),
            index=False)
    return summary_df, active_n, decoy_n

def multiTargetGenDecoy(files, model, out_dir=None, active_id_col='active_id', 
                  active_smiles_col='active', 
                  init_decoy_n=1000, n_batch=1024, max_len=120, 
                  pro=['MW', 'logP', 'TPSA', 'QED', 'SA', 'HBA', 'HBD', 'NRB',
                       'rings', 'atom_n', 'heavy_n', 'carbon_n', 'nitrogen_n',
                       'oxygen_n', 'fluorine_n', 'sulfur_n', 'chlorine_n',
                       'bromine_n', 'stereo_centers', 'aromatic_rings']):
    build_new_folder(out_dir)
    dfs = []
    targets = []
    active_ns = []
    decoy_ns = []
    for i, file in enumerate(tqdm(files)):
        # file = '/home/zdx/data/decoy_generation/DeepCoy_decoys/DeepCoy-DUDE-SMILES/dude-target-comt-decoys-final.csv'
        filename = GetFileName(file)
        target_name = filename.split('-')[2]
        targets.append(target_name)
        outfile = os.path.join(out_dir, f'{filename}.csv')
        sub_summary_df, active_n, decoy_n = batchGenDecoy(file, model=model, 
                      outfile=outfile, 
                      active_id_col=active_id_col, 
                      active_smiles_col=active_smiles_col, 
                      init_decoy_n=init_decoy_n, n_batch=n_batch,
                      max_len=max_len)
        dfs.append(sub_summary_df)
        active_ns.append(active_n)
        decoy_ns.append(decoy_n)
    active_summary = pd.concat(dfs)
    dataset_summary = pd.DataFrame({
        'target':targets,
        'active_n':active_ns,
        'decoy_n':decoy_ns
        })
    if out_dir is not None:
        active_summary.to_csv(os.path.join(out_dir, '1_active_summary.csv'),
                              index=False)
        dataset_summary.to_csv(os.path.join(out_dir, '0_dataset_summary.csv'),
                              index=False)
    return active_summary, dataset_summary

def evaluate(active, model, pro):
    generated_decoys, label = GenDecoy(active, model, pro=pro)
    df = pd.DataFrame({"SMILES":generated_decoys})
    df  = add_features(df, sort=False, remove_na=True, pro=pro)
    df_pro = df[pro]
    DataFrameDistribution(df_pro, figsize=(50,10))
    df_pro = pd.concat([df_pro.describe(), 
                        df_pro.mode().rename(index={0:'mode'})])
    df_pro = pd.concat([df_pro, label])
    return df_pro.loc[['count', 'min', '25%', '50%', '75%', 'max', 'std', 
                       'mean', 'mode', 'label']]

def lossPlot(data, loss_name, outdir=None, dpi=300):
    if isinstance(data, str):
        data = pd.read_csv(data)
    fig, ax = plt.subplots()
    
    # l = range(10)
    for key, grp in data.groupby(['mode']):
        ax = grp.plot(ax=ax, kind='line', x='epoch', y=loss_name, label=key, 
                      logy=True)
    
    plt.legend(loc='best')
    plt.title(loss_name)
    if outdir is not None:
        outfile = os.path.join(outdir, f'{loss_name}.png')
        fig.savefig(outfile, dpi=dpi, format='png')
    else:
        plt.show()


"""
# Train
--checkpoint_dir
--train_load
--pro_cols
--debug

# Predict
--test_load
--dataset_dir
--generation_dir
--model_save
"""

if __name__ == "__main__":

    parser = get_main_parser()
    config = parser.parse_known_args()[0]
    
    if config.model_save is not None or config.generation_dir is not None:
        if config.model_save is None:
            config.model_save = '/home/zdx/project/decoy_generation/result/REAL_202/model/quickcoy_model_099.pt'
        print('Predicting...')
        config.checkpoint_dir = os.path.dirname(config.model_save)
        model_name = ''
        config.vocab_save = get_vocab_path(config, model_name)
        config.log_file = get_log_path(config, model_name)
        config.config_save = get_config_path(config, model_name)
        vocab = loadVariable(config.vocab_save)
        loaded_config = loadVariable(config.config_save)
        
        set_seed(loaded_config.seed)
        device = torch.device(config.device)
    
        if device.type.startswith('cuda'):
            torch.cuda.set_device(device.index or 0)

        model = VAE(vocab, loaded_config, loaded_config.property_num).to(device)
        model.load_state_dict(torch.load(config.model_save))
        
        if config.dataset_dir is not None:
            files = glob.glob(os.path.join(config.dataset_dir, '*.csv'))
            multiTargetGenDecoy(files, model=model, 
                out_dir=config.generation_dir, 
                active_id_col=config.active_id_col, 
                active_smiles_col=config.active_smiles_col,
                init_decoy_n=config.init_decoy_n, 
                n_batch = config.n_batch,
                max_len=config.max_len,
                pro=config.pro_cols)
        elif config.test_load is not None:
            filename = GetFileName(config.test_load)
            outfile = os.path.join(config.generation_dir, f'{filename}.csv')
            batchGenDecoy(config.test_load, model=model,
                outfile=outfile,
                active_id_col=config.active_id_col, 
                active_smiles_col=config.active_smiles_col,
                init_decoy_n=config.init_decoy_n, 
                n_batch = config.n_batch,
                max_len=config.max_len,
                pro=config.pro_cols)
            
        
    elif config.checkpoint_dir is not None:
        print('Training...')
        model_name = ''
        config.model_save = get_model_path(config, model_name)
        config.config_save = get_config_path(config, model_name)
        config.vocab_save = get_vocab_path(config, model_name)
        config.log_file = get_log_path(config, model_name)
    
        if not os.path.exists(config.checkpoint_dir):
            os.mkdir(config.checkpoint_dir)
    
        set_seed(config.seed)
        device = torch.device(config.device)
    
        if device.type.startswith('cuda'):
            torch.cuda.set_device(device.index or 0)
        if config.test_load is None:
            X_train, y_train, X_test, y_test, property_num, all_smiles = \
                get_dataset(path=config.train_load,
                            smiles_col=config.smiles_col,
                            id_col=config.id_col,
                            pro_cols=config.pro_cols)
        else:
            X_train, y_train, property_num = \
                get_dataset(path=config.train_load,
                            smiles_col=config.smiles_col,
                            id_col=config.id_col,
                            pro_cols=config.pro_cols,
                            split=False)
            X_test, y_test, _ = \
                get_dataset(path=config.test_load,
                            smiles_col=config.smiles_col,
                            id_col=config.id_col,
                            pro_cols=config.pro_cols,
                            split=False)
            all_smiles = np.concatenate((X_train, X_test), axis=0)
            
        config.property_num = property_num
        print('Property Numer:', property_num)
        saveVariable(config, config.config_save)
        trainer = VAETrainer(config)
        vocab = trainer.get_vocabulary(all_smiles)
        saveVariable(vocab, config.vocab_save)
        model = VAE(vocab, config, property_num).to(device)
        trainer.fit(model, X_train, y_train, X_test, y_test)
        lossPlot(config.log_file, 'kl_loss', config.checkpoint_dir)
        lossPlot(config.log_file, 'recon_loss', config.checkpoint_dir)
        lossPlot(config.log_file, 'loss', config.checkpoint_dir)