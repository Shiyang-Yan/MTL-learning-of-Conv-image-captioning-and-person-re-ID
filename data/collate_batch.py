# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, language, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    #print (torch.stack(language, dim = 0).size())
    return torch.stack(imgs, dim=0), torch.stack(language, dim=0), pids


def val_collate_fn(batch):
    imgs, language, pids, camids, _ = zip(*batch)
    #print (torch.stack(imgs, dim = 0).size())
    return torch.stack(imgs, dim=0), torch.stack(language, dim=0), pids, camids
