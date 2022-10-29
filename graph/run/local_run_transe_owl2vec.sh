#!/bin/bash

for epochs_f in 20 40 60
do
    for emb_size in 50 100 200
    do
	python run_cge_transe.py -case go -g owl2vecstar -epf ${epochs_f} -esize ${emb_size} -eps 1 -lr 1 --margin 1 -dev cuda -train
    done
done
