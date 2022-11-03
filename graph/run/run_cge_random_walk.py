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
import gensim


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
@ck.option('--num-walks', '-nwalks', type = int, default = 20)
@ck.option('--walk-length', '-wlen', type = int, default = 5)
@ck.option('--alpha', '-alpha', type = float, default = 0.1, help = "Probability of restart random walk")
@ck.option('--epochs_w2v', '-epw2v', type = int, default = 15)
@ck.option('--window-size', '-wsize', type = int, default = 5)
@ck.option('--num-workers', '-nworkers', type = int, default = 16)
@ck.option('--embedding-size', '-esize', type = int, default = 10)
@ck.option('--lr', '-lr', type = float, default = 0.001)
@ck.option('--device', '-dev', type = ck.Choice(["cpu", "cuda"]), default = "cpu")
@ck.option('--train', '-train', is_flag = True)
@ck.option('--test', '-test', is_flag = True)

def main(case_study, graph_type, num_walks, walk_length, alpha, epochs_w2v, window_size, num_workers, embedding_size, lr, device, train, test):

    seed_everything(42)
    
    if case_study == "go":
        root = ROOT_DIR + "go_subsumption/"
        graph_prefix = root + "go.train."
    elif case_study == "foodon":
        root = ROOT_DIR + "foodon_subsumption/"
        graph_prefix = root + "foodon-merged.train."
    elif case_study == "helis":
        root = ROOT_DIR + "helis_membership/"
        graph_prefix = root + "helis_v1.00.train."
    else:
        raise ValueError(f"Invalid case study: {case_study}")

    if graph_type == "owl2vecstar":
        graph_path = graph_prefix + "only.graph.projection.edgelist"
    elif graph_type == "categorical":
        if case_study == "go":
            graph_path = graph_prefix + "cat.projection.bi.edgelist"
        else:
            graph_path = graph_prefix + "cat.projection.edgelist"
            
    outdir_walks_and_w2v = root + "cat/" + f"graph_{graph_type}_nwalks_{num_walks}_wlen_{walk_length}_epw2v_{epochs_w2v}_wsize_{window_size}_esize_{embedding_size}/"
    output_dir = root + "cat/" + f"graph_{graph_type}_nwalks_{num_walks}_wlen_{walk_length}_epw2v_{epochs_w2v}_wsize_{window_size}_esize_{embedding_size}_lr_{lr}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(outdir_walks_and_w2v):
        os.makedirs(outdir_walks_and_w2v)

    print("Configuration:")
    print("\tCase study: ", case_study)
    print("\tGraph: ", graph_type)
    print("\tNumber of walks: ", num_walks)
    print("\tWalk length: ", walk_length)
    print("\tAlpha: ", alpha)
    print("\tEpochs w2v: ", epochs_w2v)
    print("\tWindow size: ", window_size)
    print("\tNumber of workers: ", num_workers)
    print("\tEmbedding size: ", embedding_size)
    print("\tLearning rate: ", lr)
    print("\tRoot directory: ", root)
    print("\tOutput directory: ", output_dir)

    walk_outfile = outdir_walks_and_w2v + "walks.txt"
    w2v_outfile  = outdir_walks_and_w2v + "w2v.model"

    if train and not os.path.exists(w2v_outfile) and not os.path.exists(walk_outfile):
        graph = pd.read_csv(graph_path, sep = "\t", header = None)
        graph.columns = ["h", "r", "t"]
        edges = [Edge(h, r, t) for h, r, t in graph.values]
                                        
        walker = DeepWalk(num_walks, walk_length, alpha, outfile = walk_outfile, workers = num_workers)
        walker.walk(edges)

        sentences = gensim.models.word2vec.LineSentence(walk_outfile)
        
        w2v_model = gensim.models.Word2Vec(sentences, vector_size=embedding_size, window=window_size, min_count=1, workers=num_workers, sg=1, epochs=epochs_w2v)

        w2v_model.save(w2v_outfile)
        
    if test:
        print("Testing...")
        classes = pd.read_csv(root + "classes.txt", sep = "\t", header = None)
        classes.columns = ["class"]
        classes = classes["class"].values
        classes = sorted(classes)

        w2v_model = gensim.models.Word2Vec.load(w2v_outfile)
        embeddings = w2v_model.wv
        
        
        df = pd.read_csv(root+"train.csv", header = None)
        df.columns = ["source", "target", "label"]


        not_found = 0
        found = 0
        source_embs = []
        target_embs = []
        labels = []

        vocab = embeddings.index_to_key
        print("Getting embeddings from Word2Vec model...")
        for i, row in tqdm(df.iterrows(), total = len(df)):
            source = row["source"]
            target = row["target"]
            label = row["label"]

            if source in vocab and target in vocab:
                source_embs.append(embeddings[source])
 
                target_embs.append(embeddings[target])
                labels.append(label)
                found += 1
            else:
                not_found += 1

        print("Number of triples found: ", found)
        print(f"Number of classes not found: {not_found}")

        source_embs = np.array(source_embs)
        target_embs = np.array(target_embs)
        
        # Shuffle data
        indexes = np.arange(len(source_embs))
        #indexes_shuf
        np.random.shuffle(indexes)
        source_embs = source_embs[indexes]
        target_embs = target_embs[indexes]

        labels = np.array(labels)
        labels = labels[indexes]

        source_embs = th.tensor(source_embs, dtype = th.float32, device = device)
        target_embs = th.tensor(target_embs, dtype = th.float32, device = device)
        labels = th.tensor(labels, dtype = th.float32, device = device)


        vdf = pd.read_csv(root+"valid.csv", header = None)
        vdf.columns = ["source", "target"]
    
        not_found = 0
        found = 0
        vsource_embs = []
        vtarget_embs = []
    
        print("Getting embeddings from Word2Vec model...")
        for i, row in tqdm(vdf.iterrows(), total = len(vdf)):
            source = row["source"]
            target = row["target"]
        
            if source in vocab and target in vocab:
                vsource_embs.append(embeddings[source])
                vtarget_embs.append(embeddings[target])
                found += 1
            else:
                not_found += 1

        print("Number of validation triples found: ", found)
        print(f"Number of validation classes not found: {not_found}")

        vsource_embs = np.array(vsource_embs)
        vtarget_embs = np.array(vtarget_embs)

        # Shuffle data
        vindexes = np.arange(len(vsource_embs))
        #indexes_shuf
        np.random.shuffle(vindexes)
        vsource_embs = vsource_embs[vindexes]
        vtarget_embs = vtarget_embs[vindexes]

        vsource_embs = th.tensor(vsource_embs, dtype = th.float32, device = device)
        vtarget_embs = th.tensor(vtarget_embs, dtype = th.float32, device = device)
        
        model = MLP(embedding_size*2)
        model = model.to(device)
        optimizer = th.optim.Adam(model.parameters(), lr = lr)
        criterion = nn.BCELoss()
        criterion_no_reduce = nn.BCELoss(reduction = "none")

        early_stop_limit = 3
        best_loss = float("inf")
        vloss_prev = float("inf")
        for epoch in tqdm(range(4000)):
            model.train()
            #for i, (batch_data, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(source_embs, target_embs)
            loss = criterion(output, labels.float().to(device))
            loss.backward()
            optimizer.step()
            train_loss = loss.item()


            model.eval()
            with th.no_grad():
                voutput = model(vsource_embs, vtarget_embs)
                vloss= criterion(voutput, th.ones(voutput.shape).to(device = th.device(device)))
        
            if best_loss > vloss.item():
                best_loss = vloss
                th.save(model.state_dict(), output_dir + "model.th")
            if (epoch+1)%200 == 0:
                print("Epoch: ", epoch, "Loss: ", train_loss, "VLoss: ", vloss.item())
            if vloss_prev < vloss:
                early_stopping_limit -= 1
            else:
                early_stopping_limit = 3
                vloss_prev = vloss
            if early_stopping_limit == 0:
                print(f"Best epoch {epoch}")
                break

            

        # Evaluate
        model.load_state_dict(th.load(output_dir + "model.th"))
        # Classes in ontology
        classes = sorted(classes)

        # Classes in word2vec model
        model_classes = vocab

        # Intersection of classes
        model_classes = [c for c in classes if c in model_classes]
        model_classes_to_idx = {c: i for i, c in enumerate(model_classes)}

        all_embeddings = embeddings[model_classes]
        print("All embeddings shape: ", all_embeddings.shape)
        all_embeddings = th.from_numpy(all_embeddings).to(device = th.device(device))
        
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
        for i, row in tqdm(df.iterrows(), total = len(df)):
            source = row["source"]
            target = row["target"]

            if not source in vocab or not target in vocab:
                ignored += 1
                continue

            source_emb = embeddings[source]
            # replicate source as many times as there are targets
            source_emb = np.tile(source_emb, (len(model_classes), 1))
            source_emb = th.from_numpy(source_emb).to(device = th.device(device))
            assert source_emb.shape == all_embeddings.shape

            # Compute scores
            scores = model(source_emb, all_embeddings)
            scores = criterion_no_reduce(scores, th.ones(scores.shape).to(device = th.device(device))).squeeze()
            # create mask for inferred ancestors
            mask = th.ones(scores.shape).to(device = th.device(device))
            if source in inferred_ancestors:
                for ancestor in inferred_ancestors[source]:
                    if ancestor in model_classes_to_idx:
                        mask[model_classes_to_idx[ancestor]] = 10000
            
            fscores = (scores + 1) * mask
            # Sort scores
            orders = th.argsort(scores, descending = False)
            forder = th.argsort(fscores, descending = False)
            
            # Get index of target
            
            target_idx = model_classes_to_idx[target]
            
            target_rank = th.where(orders == target_idx)[0][0].item()
            ftarget_rank = th.where(forder == target_idx)[0][0].item()
            
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
            
            if i%1000 == 0:
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
        with open(root + f"{graph_type}_results", "a") as f:
            f.write(f"{num_walks},{walk_length},{epochs_w2v},{window_size},{embedding_size},{lr},{hits1},{hits5},{hits10},{mrr},{fhits1},{fhits5},{fhits10},{fmrr}\n")
        
        print("Ignored: ", ignored)
        print(f"Hits@1\tHits@5\tHits@10\tMRR\tFHits@1\tFHits@5\tFHits@10\tFMRR")
        print(f"{hits1:.4f}\t{hits5:.4f}\t{hits10:.4f}\t{mrr:.4f}\t{fhits1:.4f}\t{fhits5:.4f}\t{fhits10:.4f}\t{fmrr:.4f}")
                                
        

            
if __name__ == "__main__":
    main()
