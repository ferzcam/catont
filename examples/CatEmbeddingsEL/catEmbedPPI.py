#!/usr/bin/env python
import click as ck
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys
import logging
import torch as th

import mowl
mowl.init_jvm("10g")

from mowl.datasets.base import PathDataset
from mowl.datasets.builtin import PPIYeastSlimDataset, PPIYeastDataset

sys.path.append("../../")
from models.ppiEL import PPIEL



@ck.command()
@ck.option(
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    logging.info(f"Number of cores detected: {os.cpu_count()}")

    if species == "yeast":
        ds = PathDataset("data_old/yeast/yeast-classes.owl", "data_old/yeast/valid.owl", "data_old/yeast/test.owl")
        #ds = PPIYeastDataset()
        lr = 1e-1
        embedding_size = 80 #100
        #milestones = [20,50, 90,150, 180,400,  600, 800, 1000, 1300, 1600, 20001001] #only_nf4\
        gamma = 0.8
        margin = 5
        epochs = 10000
        step = 40
        milestones = [i*step for i in range(epochs//step)]
        milestones.append("70000000000")

    elif species == "human":
        ds = PathDataset("data_old/human/human-classes.owl", "data_old/human/valid.owl", "data_old/human/test.owl")
        lr = 1e-1
        embedding_size = 80
        
        #milestones = [20,50, 90,150, 180,400,  600, 800, 1000, 1300, 1600, 20001001] #only_nf4\
        gamma = 0.7
        margin = 0#0.5
        epochs = 1500
        step = 60
        milestones = [i*step for i in range(epochs//step)]
 
                                                                        

        
    model = PPIEL(
        ds, 
        128,#4096*8, #4096*4, #bs 
        embedding_size, #embeddings size
        lr, #lr ##1e-3 yeast, 1e-5 human
        epochs, #epochs
        500, #num points eval ppi
        milestones,
        dropout = 0.3,
        decay = 0,
        gamma = gamma,
        eval_ppi = True,
        hom_set_size = 3,
        depth =  3,
        margin = margin,
        seed = 0,
        early_stopping = 20000,
        device = "cuda:0"
    )

    model.train()
    model.evaluate()
    return

if __name__ == '__main__':
    main()
