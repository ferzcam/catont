from mowl.base_models.elmodel import EmbeddingELModel
from .elmodule import CatELModule
from .evaluate import CatEmbeddingsPPIEvaluator
from models.utils import seed_everything
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import tempfile
class PPIEL(EmbeddingELModel):

    def __init__(
            self, 
            dataset, 
            batch_size, 
            embedding_size,
            max_lr,
            min_lr,
            step_size_up,
            epochs,
            optimizer,
            margin,
            dropout,
            decay,
            hom_set_size,
            depth,
            species,
            seed = -1,
            device = "cpu"):
        
        super().__init__(dataset, batch_size, extended = False)
        
        self.embedding_size = embedding_size
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size_up = step_size_up
        self.epochs = epochs
        self.optimizer = optimizer,
        self.margin = margin
        self.dropout = dropout
        self.decay = decay
        self.hom_set_size = hom_set_size
        self.depth = depth
        self.species = species
        self.seed = seed
        self.device = device
        
        self.data_root = f"data/models/{species}/"

        file_ = tempfile.NamedTemporaryFile()
        self.file_name = file_.name
        
        self.model_filepath = self.file_name
        print(f"model will be saved in {self.model_filepath}")

        self._loaded = False

        self.top = "http://www.w3.org/2002/07/owl#Thing"
        self.bottom = "http://www.w3.org/2002/07/owl#Nothing"

        
        if seed>=0:
            seed_everything(seed)

        self.valid_evaluator = None


    def init_model(self):
        self.model = CatELModule(
            len(self.class_index_dict), #number of ontology classes
            len(self.object_property_index_dict), #number of ontology object properties
            self.hom_set_size,
            self.embedding_size,
            dropout = self.dropout,
            depth = self.depth,
        ).to(self.device)


            
    def train(self,):
        self.init_model()
        
        optimizer = self.optimizer(self.model.parameters(), lr=self.max_lr)
        best_loss = float('inf')
        best_mr = float('inf')
        
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, self.min_lr, self.max_lr, self.step_size_up)
        
        
        size_datasets = [(k,len(v)) for k,v in self.training_datasets.get_gci_datasets().items()]
        total_samples = sum([v for k,v in size_datasets])
        weights = {k: v/total_samples for k,v in size_datasets}
        print(size_datasets)
        
        for epoch in trange(self.epochs):
            self.model.train()

            train_loss = 0
            loss = 0
            
            for gci_name, gci_dataset in self.training_datasets.get_gci_datasets().items():
                if len(gci_dataset) == 0:
                    continue
                if self.batch_size>len(gci_dataset):
                    replace = True
                else:
                    replace = False
                rand_index = np.random.choice(len(gci_dataset), size = self.batch_size, replace = replace)
                rand_index = th.as_tensor(rand_index, device = self.device)
                data = gci_dataset[rand_index]

                #Positive scores computation
                pos_scores = self.model(data, gci_name)

                #Negative scores computation
                idxs_for_negs = np.random.choice(len(self.class_index_dict), size = len(data), replace = True)
                rand_index = th.tensor(idxs_for_negs, device = self.device)
                neg_data = th.cat([data[:,:-1], rand_index.unsqueeze(1)], dim = 1)                  
                neg_scores = self.model(neg_data, gci_name, neg = True)
                assert pos_scores.shape == neg_scores.shape, f"{pos_scores.shape}, {neg_scores.shape}"
                log_loss = pos_scores - neg_scores + self.margin
                log_loss = - th.mean(F.logsigmoid(-log_loss))
                loss += log_loss*weights[gci_name]

                scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.detach().item()

            loss = 0
            with th.no_grad():
                self.model.eval()
                valid_loss = 0
                gci2_data = self.validation_datasets.get_gci_datasets()["gci2"][:]
                loss = self.model(gci2_data, "gci2")
                loss = th.mean(loss)
                valid_loss += loss.detach().item()

            checkpoint = epoch + 10
            if best_loss > train_loss:
                best_loss = train_loss
                th.save(self.model.state_dict(), self.model_filepath)
            print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

            # if (epoch + 1) % checkpoint == 0:
            #     metrics = self.eval_valid()
            #     mr = metrics["mean_rank"]
            #     if best_mr > mr:
            #         best_mr = mr
            #         th.save(self.model.state_dict(), self.model_filepath)
                



    def eval_method(self, data):
        return self.model.gci2_loss(data)

    def eval_valid(self):
        if self.valid_evaluator is None:
            eval_method = self.model.gci2_loss
            self.valid_evaluator = CatEmbeddingsPPIEvaluator(self.dataset.validation, eval_method, self.dataset.ontology, self.class_index_dict, self.object_property_index_dict, device = self.device, num_points = self.num_points_eval)            
        with th.no_grad():
            self.model.eval()
            self.valid_evaluator()
            self.valid_evaluator.print_metrics()
            return self.valid_evaluator.get_metrics()
        
    def evaluate(self):
        self.init_model()
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        with th.no_grad():
            self.model.eval()

            eval_method = self.model.gci2_loss

            evaluator = CatEmbeddingsPPIEvaluator(self.dataset.testing, eval_method, self.dataset.ontology, self.class_index_dict, self.object_property_index_dict, device = self.device)
            evaluator()
            evaluator.print_metrics()
            return evaluator.get_metrics
