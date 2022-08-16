from mowl.base_models.elmodel import EmbeddingELModel
from .elmodule import CatELModule
from .evaluate import CatEmbeddingsPPIEvaluatorTune
from models.utils import seed_everything
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np

from mowl.projection.factory import projector_factory
import tempfile
import ray
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch



class PPIEL(EmbeddingELModel):

    def __init__(
            self, 
            dataset,
            species,
            seed = -1,
            device = "cpu",
            batch_size = 4096
    ):
        
        super().__init__(dataset, batch_size, extended = False)
        
        self.species = species
        self.seed = seed
        self.device = device
        
        self.data_root = f"data/models/{species}/"


        self._loaded = False

        self.top = "http://www.w3.org/2002/07/owl#Thing"
        self.bottom = "http://www.w3.org/2002/07/owl#Nothing"

        
        if seed>=0:
            seed_everything(seed)

        self.valid_evaluator = None

        #self.testing_data = self.testing_datasets.get_gci_datasets()
        self.num_classes = len(self.class_index_dict)
        self.num_obj_props = len(self.object_property_index_dict)


        self.projector = projector_factory("taxonomy_rels", relations = ["http://interacts"])
        
    def load_data(self):
        training_data = self.training_datasets.get_gci_datasets()
        validation_data = self.validation_datasets.get_gci_datasets()

        testing_edges = self.projector.project(self.dataset.testing)
        to_filter = self.projector.project(self.dataset.ontology)

        return training_data, validation_data, testing_edges, to_filter, self.num_classes, self.num_obj_props

def init_model(num_classes, num_obj_props, hom_set_size, embedding_size, dropout, depth, device):
    model = CatELModule(
        num_classes, #number of ontology classes
        num_obj_props, #number of ontology object properties
        hom_set_size,
        embedding_size,
        dropout = dropout,
        depth = depth,
    ).to(device)

    return model

def train(config, training_data = None, validation_data = None, num_classes = None, num_obj_props = None, device = None):
    
    file_ = tempfile.NamedTemporaryFile(delete=False)
    file_name = file_.name
        
    model_filepath = file_name
    print(f"model will be saved in {model_filepath}")

        
    model = init_model(num_classes, num_obj_props, config["hom_set"], config["embedding_size"],  config["dropout"],  config["depth"], device)
    th.save(model.state_dict(), model_filepath)
    optimizer = config["optimizer"](model.parameters(), lr=config["max_lr"])
    best_loss = float('inf')
    best_mr = float('inf')
        
    scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, config["min_lr"], config["max_lr"], config["step_size_up"], cycle_momentum = False)
                
    size_datasets = [(k,len(v)) for k,v in training_data.items()]
    total_samples = sum([v for k,v in size_datasets])
    weights = {k: v/total_samples for k,v in size_datasets}
    print(size_datasets)
        
    for epoch in trange(config["epochs"]):
        model.train()

        train_loss = 0
        loss = 0
            
        for gci_name, gci_dataset in training_data.items():
            if len(gci_dataset) == 0:
                continue
            if config["batch_size"]>len(gci_dataset):
                replace = True
            else:
                replace = False
            rand_index = np.random.choice(len(gci_dataset), size = config["batch_size"], replace = replace)
            rand_index = th.as_tensor(rand_index, device = device)
            data = gci_dataset[rand_index]

            #Positive scores computation
            pos_scores = model(data, gci_name)

            #Negative scores computation
            idxs_for_negs = np.random.choice(num_classes, size = len(data), replace = True)
            rand_index = th.tensor(idxs_for_negs, device = device)
            neg_data = th.cat([data[:,:-1], rand_index.unsqueeze(1)], dim = 1)                  
            neg_scores = model(neg_data, gci_name, neg = True)
            
            #Loss computation
            assert pos_scores.shape == neg_scores.shape, f"{pos_scores.shape}, {neg_scores.shape}"
            log_loss = pos_scores - neg_scores + config["margin"]
            log_loss = - th.mean(F.logsigmoid(-log_loss))
            loss += log_loss*weights[gci_name]

            scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        train_loss += loss.detach().item()

        loss = 0
        with th.no_grad():
            model.eval()
            valid_loss = 0
            gci2_data = validation_data["gci2"][:]
            loss = model(gci2_data, "gci2")
            loss = th.mean(loss)
            valid_loss += loss.detach().item()

        checkpoint = epoch + 10
        if best_loss > train_loss:
            best_loss = train_loss
            th.save(model.state_dict(), model_filepath)
        print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

            # if (epoch + 1) % checkpoint == 0:
            #     metrics = self.eval_valid()
            #     mr = metrics["mean_rank"]
            #     if best_mr > mr:
            #         best_mr = mr
            #         th.save(self.model.state_dict(), self.model_filepath)
                
    return file_name

def evaluate(config, model_filepath, testing_edges, to_filter, num_classes, num_obj_props, device, class_index_dict, object_property_index_dict):
    model = init_model(num_classes, num_obj_props, config["hom_set"], config["embedding_size"],  config["dropout"],  config["depth"], device)
        
    print('Load the best model', model_filepath)
    model.load_state_dict(th.load(model_filepath))
    with th.no_grad():
        model.eval()

        eval_method = model.gci2_loss

        evaluator = CatEmbeddingsPPIEvaluatorTune(eval_method, to_filter, class_index_dict, object_property_index_dict, device = device)
        evaluator(testing_edges)
        evaluator.print_metrics()
        return evaluator.get_metrics()

        return training_data, validation_data, testing_edges, to_filter, num_classes, num_obj_props
def train_and_evaluate(config, checkpoint_dir = None, training_data= None, validation_data=None, testing_edges = None, to_filter = None, num_classes=None, num_obj_props=None, class_index_dict=None, object_property_index_dict=None, device = None):
    print("training...")
    file_name = train(config, training_data, validation_data, num_classes, num_obj_props, device = device)
    print("evaluating...")
    metrics = evaluate(config, file_name, testing_edges, to_filter, num_classes, num_obj_props, device, class_index_dict, object_property_index_dict)
    tune.report(mean_rank = metrics["mean_rank"])
    return
