from multiprocessing.sharedctypes import Value
import sys
sys.path.append('../')
import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.projection import DL2VecProjector, OWL2VecStarProjector
from src.cat_projection import CategoricalProjector
from mowl.projection import Edge
from pykeen.models import TransE
from tqdm import tqdm

from src.train import train, generate_scores, compute_metrics
import click as ck

@ck.command()
@ck.option("--projector", "-p", default="categorical", type=ck.Choice(["categorical", "dl2vec", "owl2vecstar"]))
@ck.option("--graph-method", "-g", default="transe", type=ck.Choice(["complex", "distmult", "transe", "hole"]))
def run(projector, graph_method):
    if projector == "categorical":
        proj = CategoricalProjector()
        data_root = "data/categorical"
    elif projector == "dl2vec":
        proj = DL2VecProjector(bidirectional_taxonomy=True)
        data_root = "data/dl2vec"
    elif projector == "owl2vecstar":
        proj = OWL2VecStarProjector(bidirectional_taxonomy=True, include_literals = True, only_taxonomy = False)
        data_root = "data/owl2vecstar"

    if graph_method == "transe":
        graph_m = TransE
        data_root += "_transe"
    else:
        raise ValueError(f"Graph method {graph_method} not supported")

    dataset = PathDataset("../data/flopo.owl", testing_path = "../data/flopo-inferred.owl")
    graph = proj.project(dataset.ontology)
    #HPO
    emb_sizes = [32, 64, 128]
    epochs = [100, 150, 200]
    lrs = [0.01, 0.001, 0.0001, 0.00001]
    batch_sizes = [1024, 2048, 4096]

    out_hpo_file = f"data/{projector}_{graph_method}_hpo.txt"
    with open(out_hpo_file, "w") as f:
        f.write(f"{projector}\t{graph_method}\tEmb size\tEpochs\tLR\tBatch size\tMAE\tAUC\tAUPR\tFMax\n")

    for emb_size in tqdm(emb_sizes, total=len(emb_sizes)):
        for ep in tqdm(epochs, total=len(epochs)):
            for lr in tqdm(lrs, total=len(lrs)):
                for bs in tqdm(batch_sizes, total=len(batch_sizes)):
                    print(f"Running with emb size: {emb_size}, epochs: {ep}, lr: {lr}, batch_size: {bs}")
                    model = train(graph, graph_m, dataset, out_model_file = f"{data_root}_model_emb{emb_size}_ep{ep}_lr{lr}_bs{bs}.pt", emb_size = emb_size, epochs = ep, batch_size = bs, lr = lr)

                    eval_result_file = f"{data_root}_scores_emb{emb_size}_ep{ep}_lr{lr}_bs{bs}.txt"
                    generate_scores(model, dataset, eval_result_file)
            
                    mae_pos, auc, aupr, fmax = compute_metrics(eval_result_file)
                    with open(out_hpo_file, "a") as f:
                        f.write(f"{projector}\t{graph_method}\t{emb_size}\t{ep}\t{lr}\t{bs}\t{mae_pos}\t{auc}\t{aupr}\t{fmax}\n")
                    #print("MAE pos: {}, AUC: {}, AUPR: {}, Fmax: {}".format(mae_pos, auc, aupr, fmax))


if __name__ == "__main__":
    run()