import torch as th
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
import argparse
import tqdm

import mowl
mowl.init_jvm("10g")
from mowl.owlapi.defaults import BOT, TOP

import sys
sys.path.append("../../")

from src.finvec.dbpedia.category_qa import Category
from src.finvec.dbpedia.ppc import ppc
from src.finvec.dbpedia.evaluate import evaluate_e, evaluate_c
from src.finvec.dbpedia.data import TrainDataset, ValidDataset, TestDataset, load_train_and_test_data, load_test_data    

def get_mapper(root):
    kg_data_all = pd.read_csv(root + 'mid/kg_data_all.csv', index_col=0)
    is_data_all = pd.read_csv(root + 'mid/is_data_all.csv', index_col=0)
    is_data_train = pd.read_csv(root + 'mid/is_data_train.csv', index_col=0)
    ot = pd.read_csv(root + 'mid/ot.csv', index_col=0)

    e_from_kg = set(kg_data_all['h'].unique()) | set(kg_data_all['t'].unique())
    e_from_is = set(is_data_all['h'].unique())
    c_from_is = set(is_data_all['t'].unique())
    c_from_ot = set(ot['h'].unique()) | set(ot['t'].unique())

    e = e_from_kg | e_from_is
    c = c_from_ot | c_from_is
    r = set(kg_data_all['r'].unique())

    e_dict = dict(zip(e, range(len(e))))
    c_dict = dict(zip(c, range(len(e))))
    r_dict = dict(zip(r, range(len(r))))
    print(f'E: 0--{len(e) - 1}, C: 0--{len(c) - 1}, R: {len(r)}')
    return e_dict, c_dict, r_dict, ot, is_data_train


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DBpedia', type=str)
    parser.add_argument('--root', default='data/', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=3000, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--valid_interval', default=50, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--tolerance', default=3, type=int)
    return parser.parse_args(args)


def train():
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    seed_everything(cfg.seed)

    save_root = 'tmp/'
    device = th.device(f'cuda:{cfg.gpu}' if th.cuda.is_available() else 'cpu')
    e_dict, c_dict, r_dict, ot, is_data_train = get_mapper(cfg.root + cfg.dataset + '/')
    train_ot = ppc(ot, e_dict, c_dict, r_dict, query_type='ot', answer_type=None, flag=None)
    train_is = ppc(is_data_train, e_dict, c_dict, r_dict, query_type='is', answer_type=None, flag=None)

    train_e_1p, train_c_1p, train_filter_e_1p, train_filter_c_1p, test_e_1p, test_c_1p, test_filter_e_1p, test_filter_c_1p, valid_dl_1p_e, valid_dl_1p_c, test_dl_1p_e, test_dl_1p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='1p')
    train_e_2p, train_c_2p, train_filter_e_2p, train_filter_c_2p, test_e_2p, test_c_2p, test_filter_e_2p, test_filter_c_2p, valid_dl_2p_e, valid_dl_2p_c, test_dl_2p_e, test_dl_2p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2p')
    train_e_3p, train_c_3p, train_filter_e_3p, train_filter_c_3p, test_e_3p, test_c_3p, test_filter_e_3p, test_filter_c_3p, valid_dl_3p_e, valid_dl_3p_c, test_dl_3p_e, test_dl_3p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3p')
    train_e_2i, train_c_2i, train_filter_e_2i, train_filter_c_2i, test_e_2i, test_c_2i, test_filter_e_2i, test_filter_c_2i, valid_dl_2i_e, valid_dl_2i_c, test_dl_2i_e, test_dl_2i_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2i')
    train_e_3i, train_c_3i, train_filter_e_3i, train_filter_c_3i, test_e_3i, test_c_3i, test_filter_e_3i, test_filter_c_3i, valid_dl_3i_e, valid_dl_3i_c, test_dl_3i_e, test_dl_3i_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3i')
    test_e_pi, test_c_pi, test_filter_e_pi, test_filter_c_pi, valid_dl_pi_e, valid_dl_pi_c, test_dl_pi_e, test_dl_pi_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='pi')
    test_e_ip, test_c_ip, test_filter_e_ip, test_filter_c_ip, valid_dl_ip_e, valid_dl_ip_c, test_dl_ip_e, test_dl_ip_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='ip')
    test_e_2u, test_c_2u, test_filter_e_2u, test_filter_c_2u, valid_dl_2u_e, valid_dl_2u_c, test_dl_2u_e, test_dl_2u_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2u')
    test_e_up, test_c_up, test_filter_e_up, test_filter_c_up, valid_dl_up_e, valid_dl_up_c, test_dl_up_e, test_dl_up_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='up')

    train_data = th.cat([train_ot, train_is, train_e_1p, train_c_1p, train_e_2p, train_c_2p, train_e_3p, train_c_3p, train_e_2i, train_c_2i, train_e_3i, train_c_3i], dim=0)
    train_filters_e = {'1p': train_filter_e_1p, '2p': train_filter_e_2p, '3p': train_filter_e_3p, '2i': train_filter_e_2i, '3i': train_filter_e_3i}
    train_filters_c = {'1p': train_filter_c_1p, '2p': train_filter_c_2p, '3p': train_filter_c_3p, '2i': train_filter_c_2i, '3i': train_filter_c_3i}
    train_dataset = TrainDataset(e_dict, c_dict, train_data, num_ng=cfg.num_ng, filters={'e': train_filters_e, 'c': train_filters_c})
    train_dl = th.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.bs, num_workers=cfg.num_workers, shuffle=True, drop_last=True)

    individuals_idx = train_is[:,6].numpy()
    concepts_idx = train_is[:,7].numpy()

    ind_to_concept = [len(c_dict) + 1 for i in range(len(e_dict))]


    for i,ind in enumerate(individuals_idx):
        ind_to_concept[ind] = concepts_idx[i]

    individual_to_concept_dict = th.tensor(ind_to_concept, device=device) #{i: c for i, c in zip(individuals_idx, concepts_idx)}

    model = Category(cfg.emb_dim, e_dict, c_dict, r_dict, individual_to_concept_dict, num_negatives=cfg.num_ng)
    model = model.to(device)
    tolerance = cfg.tolerance
    max_rr = 0
    optimizer = th.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:')
        model.train()
        avg_loss = []
        if cfg.verbose == 1:
            train_dl = tqdm.tqdm(train_dl)
        for batch in train_dl:
            batch = batch.to(device)
            qe_losses, qc_losses, cc_loss, ec_loss = model(batch)
            qe_loss = sum(qe_losses) / len(qe_losses)
            # qc_loss = sum(qc_losses) / len(qc_losses)
            loss = qe_loss
            # loss = (qe_loss + qc_loss + cc_loss + ec_loss) / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss)/len(avg_loss), 4)}')
        if (epoch + 1) % cfg.valid_interval == 0:
            model.eval()
            print('Validating Entity Answering:')
            _, rr_1p_e, h1_1p_e, h3_1p_e, _ = evaluate_e(model, valid_dl_1p_e, test_filter_e_1p, device, query_type='1p')
            _, rr_2p_e, h1_2p_e, h3_2p_e, _ = evaluate_e(model, valid_dl_2p_e, test_filter_e_2p, device, query_type='2p')
            _, rr_3p_e, h1_3p_e, h3_3p_e, _ = evaluate_e(model, valid_dl_3p_e, test_filter_e_3p, device, query_type='3p')
            _, rr_2i_e, h1_2i_e, h3_2i_e, _ = evaluate_e(model, valid_dl_2i_e, test_filter_e_2i, device, query_type='2i')
            _, rr_3i_e, h1_3i_e, h3_3i_e, _ = evaluate_e(model, valid_dl_3i_e, test_filter_e_3i, device, query_type='3i')
            _, rr_pi_e, h1_pi_e, h3_pi_e, _ = evaluate_e(model, valid_dl_pi_e, test_filter_e_pi, device, query_type='pi')
            #_, rr_ip_e, h1_ip_e, h3_ip_e, _ = evaluate_e(model, valid_dl_ip_e, test_filter_e_ip, device, query_type='ip')
            #_, rr_2u_e, h1_2u_e, h3_2u_e, _ = evaluate_e(model, valid_dl_2u_e, test_filter_e_2u, device, query_type='2u')
            #_, rr_up_e, h1_up_e, h3_up_e, _ = evaluate_e(model, valid_dl_up_e, test_filter_e_up, device, query_type='up')
            #mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e, rr_pi_e, rr_ip_e, rr_2u_e, rr_up_e]) / 9, 3)
            #mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e, h1_pi_e, h1_ip_e, h1_2u_e, h1_up_e]) / 9, 3)
            #mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e, h3_pi_e, h3_ip_e, h3_2u_e, h3_up_e]) / 9, 3)
            mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e]) / 5, 3)
            mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e]) / 5, 3)
            mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e]) / 5, 3)
            print(f'Entity Answering Mean: \n MRR: {mrr_e}, H1: {mh1_e}, H3: {mh3_e}')
            # print('Validating Concept Answering:')
            # _, rr_1p_c, h1_1p_c, h3_1p_c, _ = evaluate_c(model, valid_dl_1p_c, test_filter_c_1p, device, query_type='1p')
            # _, rr_2p_c, h1_2p_c, h3_2p_c, _ = evaluate_c(model, valid_dl_2p_c, test_filter_c_2p, device, query_type='2p')
            # _, rr_3p_c, h1_3p_c, h3_3p_c, _ = evaluate_c(model, valid_dl_3p_c, test_filter_c_3p, device, query_type='3p')
            # _, rr_2i_c, h1_2i_c, h3_2i_c, _ = evaluate_c(model, valid_dl_2i_c, test_filter_c_2i, device, query_type='2i')
            # _, rr_3i_c, h1_3i_c, h3_3i_c, _ = evaluate_c(model, valid_dl_3i_c, test_filter_c_3i, device, query_type='3i')
            # _, rr_pi_c, h1_pi_c, h3_pi_c, _ = evaluate_c(model, valid_dl_pi_c, test_filter_c_pi, device, query_type='pi')
            # _, rr_ip_c, h1_ip_c, h3_ip_c, _ = evaluate_c(model, valid_dl_ip_c, test_filter_c_ip, device, query_type='ip')
            # _, rr_2u_c, h1_2u_c, h3_2u_c, _ = evaluate_c(model, valid_dl_2u_c, test_filter_c_2u, device, query_type='2u')
            # _, rr_up_c, h1_up_c, h3_up_c, _ = evaluate_c(model, valid_dl_up_c, test_filter_c_up, device, query_type='up')
            # mrr_c = round(sum([rr_1p_c, rr_2p_c, rr_3p_c, rr_2i_c, rr_3i_c, rr_pi_c, rr_ip_c, rr_2u_c, rr_up_c]) / 9, 3)
            # mh1_c = round(sum([h1_1p_c, h1_2p_c, h1_3p_c, h1_2i_c, h1_3i_c, h1_pi_c, h1_ip_c, h1_2u_c, h1_up_c]) / 9, 3)
            # mh3_c = round(sum([h3_1p_c, h3_2p_c, h3_3p_c, h3_2i_c, h3_3i_c, h3_pi_c, h3_ip_c, h3_2u_c, h3_up_c]) / 9, 3)
            # print(f'Concept Answering Mean: \n MRR: {mrr_c}, H1: {mh1_c}, H3: {mh3_c}')

            mrr_e = h1_1p_e
            if mrr_e >= max_rr:
                max_rr = mrr_e
                tolerance = cfg.tolerance
            else:
                tolerance -= 1

            th.save(model.state_dict(), save_root + (str(epoch + 1)))

        if tolerance == 0:
            print(f'Best performance at epoch {epoch - cfg.tolerance * cfg.valid_interval + 1}')
            model.load_state_dict(th.load(save_root + str(epoch - cfg.tolerance * cfg.valid_interval + 1)))
            model.eval()
            print('Testing Entity Answering:')
            _, rr_1p_e, h1_1p_e, h3_1p_e, _ = evaluate_e(model, test_dl_1p_e, test_filter_e_1p, device, query_type='1p')
            _, rr_2p_e, h1_2p_e, h3_2p_e, _ = evaluate_e(model, test_dl_2p_e, test_filter_e_2p, device, query_type='2p')
            _, rr_3p_e, h1_3p_e, h3_3p_e, _ = evaluate_e(model, test_dl_3p_e, test_filter_e_3p, device, query_type='3p')
            _, rr_2i_e, h1_2i_e, h3_2i_e, _ = evaluate_e(model, test_dl_2i_e, test_filter_e_2i, device, query_type='2i')
            _, rr_3i_e, h1_3i_e, h3_3i_e, _ = evaluate_e(model, test_dl_3i_e, test_filter_e_3i, device, query_type='3i')
            _, rr_pi_e, h1_pi_e, h3_pi_e, _ = evaluate_e(model, test_dl_pi_e, test_filter_e_pi, device, query_type='pi')
            # _, rr_ip_e, h1_ip_e, h3_ip_e, _ = evaluate_e(model, test_dl_ip_e, test_filter_e_ip, device, query_type='ip')
            # _, rr_2u_e, h1_2u_e, h3_2u_e, _ = evaluate_e(model, test_dl_2u_e, test_filter_e_2u, device, query_type='2u')
            # _, rr_up_e, h1_up_e, h3_up_e, _ = evaluate_e(model, test_dl_up_e, test_filter_e_up, device, query_type='up')
            # mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e, rr_pi_e, rr_ip_e, rr_2u_e, rr_up_e]) / 9, 3)
            # mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e, h1_pi_e, h1_ip_e, h1_2u_e, h1_up_e]) / 9, 3)
            # mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e, h3_pi_e, h3_ip_e, h3_2u_e, h3_up_e]) / 9, 3)
            mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e, rr_pi_e]) / 6, 3)
            mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e, h1_pi_e]) / 6, 3)
            mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e, h3_pi_e]) / 6, 3)
            print(f'Entity Answering Mean: \n MRR: {mrr_e}, H1: {mh1_e}, H3: {mh3_e}')
            # print('Testing Concept Answering:')
            # _, rr_1p_c, h1_1p_c, h3_1p_c, _ = evaluate_c(model, test_dl_1p_c, test_filter_c_1p, device, query_type='1p')
            # _, rr_2p_c, h1_2p_c, h3_2p_c, _ = evaluate_c(model, test_dl_2p_c, test_filter_c_2p, device, query_type='2p')
            # _, rr_3p_c, h1_3p_c, h3_3p_c, _ = evaluate_c(model, test_dl_3p_c, test_filter_c_3p, device, query_type='3p')
            # _, rr_2i_c, h1_2i_c, h3_2i_c, _ = evaluate_c(model, test_dl_2i_c, test_filter_c_2i, device, query_type='2i')
            # _, rr_3i_c, h1_3i_c, h3_3i_c, _ = evaluate_c(model, test_dl_3i_c, test_filter_c_3i, device, query_type='3i')
            # _, rr_pi_c, h1_pi_c, h3_pi_c, _ = evaluate_c(model, test_dl_pi_c, test_filter_c_pi, device, query_type='pi')
            # _, rr_ip_c, h1_ip_c, h3_ip_c, _ = evaluate_c(model, test_dl_ip_c, test_filter_c_ip, device, query_type='ip')
            # _, rr_2u_c, h1_2u_c, h3_2u_c, _ = evaluate_c(model, test_dl_2u_c, test_filter_c_2u, device, query_type='2u')
            # _, rr_up_c, h1_up_c, h3_up_c, _ = evaluate_c(model, test_dl_up_c, test_filter_c_up, device, query_type='up')
            # mrr_c = round(sum([rr_1p_c, rr_2p_c, rr_3p_c, rr_2i_c, rr_3i_c, rr_pi_c, rr_ip_c, rr_2u_c, rr_up_c]) / 9, 3)
            # mh1_c = round(sum([h1_1p_c, h1_2p_c, h1_3p_c, h1_2i_c, h1_3i_c, h1_pi_c, h1_ip_c, h1_2u_c, h1_up_c]) / 9, 3)
            # mh3_c = round(sum([h3_1p_c, h3_2p_c, h3_3p_c, h3_2i_c, h3_3i_c, h3_pi_c, h3_ip_c, h3_2u_c, h3_up_c]) / 9, 3)
            # print(f'Concept Answering Mean: \n MRR: {mrr_c}, H1: {mh1_c}, H3: {mh3_c}')
            # break


if __name__ == '__main__':
    train()
