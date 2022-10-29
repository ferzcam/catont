import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.walking import DeepWalk
from mowl.projection import Edge
from tqdm import tqdm
import os
import numpy as np
import pickle as pk
import click as ck
import pandas as pd
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

from mowl.kge import KGEModel
from src.OpenKE.openke.module.model import TransE as T2
from pykeen.models import TransE


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        
    def forward(self, a, b):
        x = th.cat([a, b], dim=1)
        x = self.fc1(x).squeeze()
        return th.sigmoid(x)

class SubsumptionDataset(Dataset):
    def __init__(self, data, labels):
        #assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

ROOT_DIR = "../case_studies/"

@ck.command()
@ck.option('--case-study', '-case', type = ck.Choice(["go", "foodon", "helis"]))
@ck.option('--graph-type', '-g', type = ck.Choice(["owl2vecstar", "categorical"]))
@ck.option('--epochs_first', '-epf', type = int, default = 10)
@ck.option('--epochs_second', '-eps', type = int, default = 10)
@ck.option('--embedding-size', '-esize', type = int, default = 10)
@ck.option('--lr', '-lr', type = float, default = 0.001)
@ck.option('--margin', '-m', type = float, default = 1.0)
@ck.option('--device', '-dev', type = ck.Choice(["cpu", "cuda"]), default = "cpu")
@ck.option('--seed', '-seed', type = int, default = 42)
@ck.option('--train', '-train', is_flag = True)
@ck.option('--test', '-test', is_flag = True)

def main(case_study, graph_type, epochs_first, epochs_second, embedding_size, lr, margin, device, seed, train, test):

    seed_everything(seed)
    
    if case_study == "go":
        root = ROOT_DIR + "go_subsumption/"
        graph_prefix = root + "go.train."
    elif case_study == "foodon":
        root = ROOT_DIR + "foodon_subsumption/"
        graph_prefix = root + "foodon-merged.train."
    elif case_study == "helis":
        root = ROOT_DIR + "helis_memmbership/"
    else:
        raise ValueError(f"Invalid case study: {case_study}")

    if graph_type == "owl2vecstar":
        graph_path = graph_prefix + "only.graph.projection.edgelist"
        subclassof_rel = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    elif graph_type == "categorical":
        graph_path = graph_prefix + "cat.projection.edgelist"
        subclassof_rel = "http://subclassof"
    outdir_transe = root + "cat/" + f"graph_{graph_type}_epf_{epochs_first}_esize_{embedding_size}/"
    output_dir = root + "cat/" + f"graph_{graph_type}_epf_{epochs_first}_esize_{embedding_size}_eps_{epochs_second}_lr_{lr}_margin_{margin}/"

    if not os.path.exists(outdir_transe):
        os.makedirs(outdir_transe)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Configuration:")
    print("\tCase study: ", case_study)
    print("\tGraph: ", graph_type)
    print("\tEpochs first: ", epochs_first)
    print("\tEpochs second: ", epochs_second)
    print("\tEmbedding size: ", embedding_size)
    print("\tLearning rate: ", lr)
    print("\tMargin: ", margin)
    print("\tDevice: ", device)
    print("\tSeed: ", seed)
    print("\tRoot directory: ", root)
    print("\tOutput directory: ", output_dir)

    transe_outfile = outdir_transe + "transe.model"
    
    if train:
        print("Training...")

        graph = pd.read_csv(graph_path, sep = "\t", header = None)
        graph.columns = ["h", "r", "t"]
        classes = list(set(graph["h"].tolist() + graph["t"].tolist()))
        classes.sort()
        class_to_id = {c: i for i, c in enumerate(classes)}
        relations = list(set(graph["r"].tolist()))
        relations.sort()
        relation_to_id = {r: i for i, r in enumerate(relations)}

        edges = [Edge(row["h"], row["r"], row["t"]) for _, row in graph.iterrows()]
        triples_factory = Edge.as_pykeen(edges, create_inverse_triples=True, entity_to_id=class_to_id, relation_to_id=relation_to_id)
        

        if not os.path.exists(root + f"{graph_type}_relations.txt"):
            with open(root + f"{graph_type}_relations.txt", "w") as f:
                for r in relations:
                    f.write(r + "\n")
        
        pk_model = TransE(triples_factory=triples_factory, random_seed=seed, embedding_dim=embedding_size)

        model = KGEModel(triples_factory, pk_model, epochs_first, batch_size=1024*4, device=device, lr=lr)
        model.train()
        ent_embs = model.class_embeddings_dict
        rel_embs = model.object_property_embeddings_dict
        
        with open(outdir_transe+ "ent_embs", "wb") as f:
            pk.dump(ent_embs, f)
        with open(outdir_transe + "rel_embs", "wb") as f:
            pk.dump(rel_embs, f)
        
    if test:
        print("Testing...")
        classes = pd.read_csv(root + "classes.txt", sep = "\t", header = None)
        classes.columns = ["class"]
        classes = classes["class"].values
                
        relations_file = pd.read_csv(root + f"{graph_type}_relations.txt", sep = "\t", header = None)
        relations_file.columns = ["relation"]
        relations = relations_file["relation"].values
        relations = sorted(relations)
        relation_to_id = {r: i for i, r in enumerate(relations)}
        
        with open(outdir_transe + "ent_embs", "rb") as f:
            ent_embs = pk.load(f)
        with open(outdir_transe + "rel_embs", "rb") as f:
            rel_embs = pk.load(f)

        
        
        vocab = set(ent_embs.keys())
        classes = [c for c in classes if c in vocab]
        classes.sort()
        class_to_id = {c: i for i, c in enumerate(classes)}
        embeddings = np.array([ent_embs[c] for c in classes])
        
        
        df = pd.read_csv(root+"train.csv", header = None)
        df.columns = ["source", "target", "label"]

        not_found = 0
        found = 0
        
        print("Getting embeddings from First TransE model...")
        positives = {"batch_h": np.array([]), "batch_t": np.array([]), "batch_r": np.array([]), "mode": "normal"}
        negatives = {"batch_h": np.array([]), "batch_t": np.array([]), "batch_r": np.array([]), "mode": "normal"}
                
        for i, row in tqdm(df.iterrows(), total = len(df)):
            source = row["source"]
            target = row["target"]
            label = row["label"]
            relation = relation_to_id[subclassof_rel]

            if source in vocab and target in vocab:
                source = class_to_id[source]
                target = class_to_id[target]

                if label == 1:
                    positives["batch_h"] = np.append(positives["batch_h"], source)
                    positives["batch_t"] = np.append(positives["batch_t"], target)
                    positives["batch_r"] = np.append(positives["batch_r"], relation)

                elif label == 0:
                    negatives["batch_h"] = np.append(negatives["batch_h"], source)
                    negatives["batch_t"] = np.append(negatives["batch_t"], target)
                    negatives["batch_r"] = np.append(negatives["batch_r"], relation)
                else:
                    raise ValueError(f"Invalid label: {label}")
                
                found += 1
            else:
                not_found += 1

        print("Number of triples found: ", found)
        print(f"Number of classes not found: {not_found}")

        positives["batch_h"] = th.tensor(positives["batch_h"], dtype = th.long, device=device)
        positives["batch_t"] = th.tensor(positives["batch_t"], dtype = th.long, device=device)
        positives["batch_r"] = th.tensor(positives["batch_r"], dtype = th.long, device=device)
        negatives["batch_h"] = th.tensor(negatives["batch_h"], dtype = th.long, device=device)
        negatives["batch_t"] = th.tensor(negatives["batch_t"], dtype = th.long, device=device)
        negatives["batch_r"] = th.tensor(negatives["batch_r"], dtype = th.long, device=device)


        # Validation data:
        df = pd.read_csv(root+"valid.csv", header = None)
        df.columns = ["source", "target"]

        not_found = 0
        found = 0
        validation_data = {"batch_h": np.array([]), "batch_t": np.array([]), "batch_r": np.array([]), "mode": "normal"}
        for i, row in tqdm(df.iterrows(), total = len(df)):
            source = row["source"]
            target = row["target"]
            relation = relation_to_id[subclassof_rel]

            if source in vocab and target in vocab:
                source = class_to_id[source]
                target = class_to_id[target]

                validation_data["batch_h"] = np.append(validation_data["batch_h"], source)
                validation_data["batch_t"] = np.append(validation_data["batch_t"], target)
                validation_data["batch_r"] = np.append(validation_data["batch_r"], relation)

                found += 1
            else:
                not_found += 1

        print("Number of triples found: ", found)
        print(f"Number of classes not found: {not_found}")
        validation_data["batch_h"] = th.tensor(validation_data["batch_h"], dtype = th.long, device=device)
        validation_data["batch_t"] = th.tensor(validation_data["batch_t"], dtype = th.long, device=device)
        validation_data["batch_r"] = th.tensor(validation_data["batch_r"], dtype = th.long, device=device)
        
        
        model = T2(len(classes), len(relations), dim = embedding_size)
        model.to(device)
        assert model.ent_embeddings.weight.data.shape == (len(classes), embedding_size)
        
        model.ent_embeddings.weight.data = th.from_numpy(embeddings).to(device)
        model.rel_embeddings.weight.data = th.from_numpy(np.array(list(rel_embs.values()))).to(device)

        optimizer = th.optim.Adam(model.parameters(), lr = lr)
#        criterion = nn.MarginRankingLoss(margin = margin, reduction = "mean")
        criterion = nn.LogSigmoid()
        best_loss = float("inf")
        for epoch in tqdm(range(epochs_second)):
            model.train()
            
            pos_dist  = model(positives).sum()
            neg_dist  = model(negatives).sum()
            targets = -th.ones_like(pos_dist).to(device)
            loss = - th.nn.functional.logsigmoid(pos_dist - neg_dist + margin).mean()
            #loss = -criterion(pos_dist-neg_dist)
            #loss = criterion(pos_dist, neg_dist, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()/len(positives["batch_h"])

            model.eval()
            with th.no_grad():
                val_dist = model(validation_data)
                val_dist = th.mean(val_dist).item()/len(validation_data["batch_h"])

            if best_loss >  train_loss:
                best_loss = train_loss
                th.save(model.state_dict(), output_dir + "model.th")

            if epoch % 100 == 0:
                print("Epoch: ", epoch, "Loss: ", train_loss, "Val dist: ", val_dist)
            

        # Evaluate
        model.load_state_dict(th.load(output_dir + "model.th"))
        
        # Classes in ontology
                        

        # Intersection of classes
        all_classes = np.array(list(class_to_id.values()))
        relation = relation_to_id[subclassof_rel]
        relations = np.array([relation] * len(all_classes))

        test_data = {"batch_t": th.tensor(all_classes, dtype = th.long, device=device), "batch_r": th.tensor(relations, dtype = th.long, device=device), "mode": "normal"}

        
        # Test data
        df = pd.read_csv(root+"test.csv", header = None)
        df.columns = ["source", "target"]

        inferred_ancestors = dict()
        with open(root+"inferred_ancestors.txt", "r") as f:
            for line in f.readlines():
                all_infer_classes = line.strip().split(",")
                cls = all_infer_classes[0]
                ancestors = all_infer_classes
                inferred_ancestors[cls] = ancestors
        
        
        hits1 = 0
        hits5 = 0
        hits10 = 0
        mrr = 0
        fhits1 = 0
        fhits5 = 0
        fhits10 = 0
        fmrr = 0
        

        ignored = 0
        model.eval()
        with th.no_grad():
            for i, row in tqdm(df.iterrows(), total = len(df)):
                source = row["source"]
                target = row["target"]

                if not source in vocab or not target in vocab:
                    ignored += 1
                    continue

                source_id = class_to_id[source]
                target_id = class_to_id[target]

                sources = np.array([source_id] * len(all_classes))
                test_data["batch_h"] = th.tensor(sources, dtype = th.long, device=device)

                scores = model(test_data)
                # create a mask for inferred ancestors
                mask = th.ones_like(scores).to(device)
                if source in inferred_ancestors:
                    for ancestor in inferred_ancestors[source]:
                        if ancestor in class_to_id:
                            ancestor_id = class_to_id[ancestor]
                            mask[ancestor_id] = 100000

                fscores = (scores + 1) * mask                
                # Sort scores
                orders = th.argsort(scores, descending = False)
                forders = th.argsort(fscores, descending = False)
            
                # Get index of target
                target_rank = th.where(orders == target_id)[0][0].item()
                ftarget_rank = th.where(forders == target_id)[0][0].item()

                if target_rank == 0:
                    hits1 += 1
                if target_rank < 5:
                    hits5 += 1
                if target_rank < 10:
                    hits10 += 1
                mrr += 1/(target_rank+1)

                if ftarget_rank == 0:
                    fhits1 += 1
                if ftarget_rank < 5:
                    fhits5 += 1
                if ftarget_rank < 10:
                    fhits10 += 1
                fmrr += 1/(ftarget_rank+1)

                if i % 1000 == 0:
                    print(f"Evaluated {i} triples")
                    print("Hits@1\tHits@5\tHits@10\tMRR\tFHits@1\tFHits@5\tFHits@10\tFMRR")
                    tmp_hits1 = hits1/(i+1)
                    tmp_hits5 = hits5/(i+1)
                    tmp_hits10 = hits10/(i+1)
                    tmp_mrr = mrr/(i+1)
                    tmp_fhits1 = fhits1/(i+1)
                    tmp_fhits5 = fhits5/(i+1)
                    tmp_fhits10 = fhits10/(i+1)
                    tmp_fmrr = fmrr/(i+1)
                    print(f"{tmp_hits1:.4f}\t{tmp_hits5:.4f}\t{tmp_hits10:.4f}\t{tmp_mrr:.4f}\t{tmp_fhits1:.4f}\t{tmp_fhits5:.4f}\t{tmp_fhits10:.4f}\t{tmp_fmrr:.4f}")

            hits1 = hits1/(len(df)-ignored)
            hits5 = hits5/(len(df)-ignored)
            hits10 = hits10/(len(df)-ignored)
            mrr = mrr/(len(df)-ignored)
            fhits1 = fhits1/(len(df)-ignored)
            fhits5 = fhits5/(len(df)-ignored)
            fhits10 = fhits10/(len(df)-ignored)
            fmrr = fmrr/(len(df)-ignored)

            with open(root + f"{graph_type}_transe_results", "a") as f:
                f.write(f"{epochs_first},{epochs_second},{embedding_size},{lr},{margin},{hits1},{hits5},{hits10},{mrr},{fhits1},{fhits5},{fhits10},{fmrr}\n")
            
            print("Ignored: ", ignored)
            print(f"Hits@1\tHits@5\tHits@10\tMRR\tFHits@1\tFHits@5\tFHits@10\tFMRR")
            print(f"{hits1:.4f}\t{hits5:.4f}\t{hits10:.4f}\t{mrr:.4f}\t{fhits1:.4f}\t{fhits5:.4f}\t{fhits10:.4f}\t{fmrr:.4f}")
                                                
            
if __name__ == "__main__":
    main()
