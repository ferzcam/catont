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
import pickle as pkl
import ray
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch

from path import Path                                                      

import mowl
mowl.init_jvm("10g")

from mowl.datasets.base import PathDataset
from mowl.datasets.builtin import PPIYeastSlimDataset, PPIYeastDataset

sys.path.append(Path("../../").abspath())
from models.ppiEL_tune import PPIEL, train_and_evaluate



@ck.command()
@ck.option(
    '--species', '-sp', help="Nome of species",
    default = "yeast"
)

def main(species):
    logging.info(f"Number of cores detected: {os.cpu_count()}")

    
    
    if species == "yeast":
        ds = PathDataset(Path("data_old/yeast/yeast-classes.owl").abspath(), Path("data_old/yeast/valid.owl").abspath(), Path("data_old/yeast/test.owl").abspath())

        #ds = PPIYeastDataset()
        search_space = {
            "batch_size": tune.choice([4096, 4096*2]), #tune.choice([2048, 4096, 4096*2]), 
            "embedding_size": tune.choice([25, 50, 100, 150]),
            "max_lr": tune.choice([1e-1, 1e-2]),
            "min_lr": tune.choice([1e-3, 1e-4, 1e-5]),
            "step_size_up": tune.choice([50, 75, 100]),
            "epochs": tune.choice([2000, 3000]),#tune.choice([1000, 2000, 3000]),
            "optimizer": tune.choice(["adam", "rmsprop"]),
            "margin": tune.choice([2, 4, 6]),
            "dropout": tune.choice([0.1, 0.2, 0.3, 0.4]),
            "decay": tune.choice([0, 0.001, 0.005]),
            "hom_set": tune.choice([1, 2, 3, 4]),
            "depth": tune.choice([1,2,3,4])
        }

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

    
        
    train_ppi_el(search_space, ds = ds, device = "cuda", seed = 0, species = species)

def train_ppi_el(search_space, ds=None, device = 'cuda', seed = 0, species = "yeast"):
    model = PPIEL(
        ds, 
        species,
        seed = seed,
        device = device
    )

    training_data, validation_data, testing_edges, to_filter, num_classes, num_obj_props = model.load_data()
    class_index_dict = model.class_index_dict
    object_property_index_dict = model.object_property_index_dict

    ray.init(log_to_driver = False)
    algo = OptunaSearch()

    data = dict()
    data["training_data"] = training_data
    data["validation_data"] = validation_data
    data["testing_edges"] = testing_edges
    data["to_filter"] = to_filter
    data["num_classes"] = num_classes
    data["num_obj_props"] = num_obj_props
    data["class_index_dict"] = class_index_dict
    data["object_property_index_dict"] = object_property_index_dict


    tuner = tune.run(
        tune.with_parameters(train_and_evaluate, data = data, device = device),
        metric = "mean_rank",
        mode = "min",
        search_alg = algo,
        config = search_space,
        resources_per_trial = {"gpu": 1},
        stop = {"training_iteration": 1},
        num_samples = 100,
        log_to_file = True
    )

    return


        

    

if __name__ == '__main__':
    main()
