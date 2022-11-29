import os
from mowl.kge import KGEModel
from mowl.projection import TaxonomyWithRelsProjector
from pykeen.models import TransD
from mowl.projection import Edge
import time
import random
from tqdm import tqdm
import torch as th
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def train(graph, graph_method, dataset, out_model_file = None, emb_size = None, epochs = None, lr = None, batch_size = None, seed = 0):
    """Projects an ontology into a graph and trains a word2vec model.
    
    :param projector: Projector to use
    :param dataset: mOWL dataset to use
    """
    pykeen_triples = Edge.as_pykeen(graph, create_inverse_triples=True)

    candidate_embedding_dim = 2*len(graph)//(len(dataset.classes.as_str) + len(dataset.object_properties.as_str))
    embedding_dim = emb_size# max(20, candidate_embedding_dim)

    print(f"Embedding dim: {embedding_dim}")

    pykeen_model = graph_method(triples_factory = pykeen_triples, embedding_dim = embedding_dim, random_seed = seed)
    model = KGEModel(pykeen_triples, 
                    pykeen_model, 
                    epochs = epochs, 
                    batch_size = batch_size,
                    device = "cuda:0",
                    lr = lr,
                    model_filepath = out_model_file
                    )

    if not os.path.exists(out_model_file):
        print("Training model")
        model.train()
    else:
        print("Loading model")
        model.load_best_model()

    return model

def generate_scores(model, onto, eval_results_file):

    model.load_best_model()

    entities_to_id = {k:v for k, v in model.triples_factory.entity_to_id.items() if not k.startswith("http://mowl")}
    relations_to_id = model.triples_factory.relation_to_id


    with open(f"../data/positives_{onto}.txt", "r") as f:

        with open(eval_results_file, "w") as f_out:
            n_pos = 0
            for line in f.readlines():
            
                src, dst = line.strip().split("\t")
                if not src in entities_to_id:
                    print(f"Source entity {src} not in training set")
                    continue
                if not dst in entities_to_id:
                    print(f"Destination entity {dst} not in training set")
                    continue
                n_pos += 1
                src = entities_to_id[src]
                rel = relations_to_id["http://subclassof"]
                dst = entities_to_id[dst]

                point = [src, rel, dst]
                point = th.tensor(point, dtype=th.long, device=model.device).unsqueeze(0)
                pos_score = th.sigmoid(model.score_method_tensor(point)).detach().item()
                f_out.write(f"1\t{pos_score}\n")

    print(f"Positive triples: {n_pos}")
    with open(f"../data/negatives_{onto}.txt", "r") as f:

        with open(eval_results_file, "a") as f_out:
            n_neg = 0
            for line in f.readlines():
                src, dst = line.strip().split("\t")
                if not src in entities_to_id:
                    print(f"Source entity {src} not in training set")
                    continue
                if not dst in entities_to_id:
                    print(f"Destination entity {dst} not in training set")
                    continue

                n_neg += 1
                src = entities_to_id[src]
                rel = relations_to_id["http://subclassof"]
                dst = entities_to_id[dst]

                point = [src, rel, dst]
                point = th.tensor(point, dtype=th.long, device=model.device).unsqueeze(0)
                pos_score = th.sigmoid(model.score_method_tensor(point)).detach().item()
                f_out.write(f"0\t{pos_score}\n")

    print(f"Negative triples: {n_neg}")
    #scores = undecidable_scores + scores


def compute_metrics(eval_tsv_file):
    with open(eval_tsv_file, "r") as f:
        eval_data = [line.strip().split("\t") for line in f.readlines()]
    
    eval_data = [(int(label), float(score)) for label, score in eval_data]

    n_pos = n_neg = len(eval_data)//2

    labels = [label for label, score in eval_data]
    preds = [score for label, score in eval_data]

        
    mae_pos = round(sum(preds[:n_pos]) / n_pos, 4)
    auc = round(roc_auc_score(labels, preds), 4)
    aupr = round(average_precision_score(labels, preds), 4)
    
    precision, recall, _ = precision_recall_curve(labels, preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    fmax = round(np.max(f1_scores), 4)
    
    return mae_pos, auc, aupr, fmax, len(eval_data)

def predict_inferences(model, dataset):
    """Predict subsumption inferences
    :param model: trained model
    :param dataset: mOWL dataset to get the classes from.
    """
    #Potential bottleneck, check time
    entities = dataset.classes.as_str
    entities_idx = [model.triples_factory.entity_to_id[entity] for entity in entities if entity in model.triples_factory.entity_to_id]
    relation_to_id = model.triples_factory.relation_to_id

    start = time.time()
    inferences = []
    rel_id = relation_to_id["http://subclassof"]
    for sub in tqdm(entities_idx, total = len(entities_idx)):
        data = np.zeros((len(entities_idx), 3), dtype = np.int64)
        data[:, 0] = sub
        data[:, 1] = rel_id
        data[:, 2] = np.arange(len(entities_idx))
        data_as_tensor = th.tensor(data, dtype = th.long, device = model.device).unsqueeze(0)
        sim = model.score_method_tensor(data_as_tensor)

        for point, val in zip(data, sim):
            inferences.append((point[0], point[1], point[2], val))

    end = time.time()
    print("Time computing predictions: {}".format(end - start))

    with open("data/inferences.txt", "w") as f:
        for sub, sup, sim in inferences:
            f.write("{}\t{}\t{}\n".format(sub, sup, sim))




