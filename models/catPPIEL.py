from mowl.base_models.elmodel import EmbeddingELModel
from .elmodule import CatELModule
from .evaluate import CatEmbeddingsPPIEvaluator
from models.utils import seed_everything
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np

class CatPPIEL(EmbeddingELModel):

    def __init__(
            self, 
            dataset, 
            batch_size, 
            embedding_size,
            lr,
            epochs,
            num_points_eval,
            milestones,
            dropout = 0,
            decay = 0,
            gamma = None,
            eval_ppi = False,
            hom_set_size = 1,
            depth = 1,
            margin = 0,
            seed = -1,
            early_stopping = 10,
            species = "yeast",
            device = "cpu"):
        
        super().__init__(dataset, batch_size, extended = False)
        
        self.embedding_size = embedding_size
        self.lr = lr
        self.epochs = epochs
        self.num_points_eval = num_points_eval
        self.milestones = milestones
        self.dropout = dropout
        self.decay = decay
        self.gamma = gamma
        self.eval_ppi = eval_ppi
        self.hom_set_size = hom_set_size
        self.depth = depth
        self.margin = margin
        self.early_stopping = early_stopping
        self.device = device
        self.dataset = dataset

        
        self.species = species
        milestones_str = "_".join(str(m) for m in milestones[-10:])
        self.data_root = f"data/models/{species}/"
        self.file_name = f"bs{self.batch_size}_emb{self.embedding_size}_lr{lr}_epochs{epochs}_eval{num_points_eval}_mlstns_{milestones_str}_drop_{self.dropout}_decay_{self.decay}_gamma_{self.gamma}_evalppi_{self.eval_ppi}_margin{self.margin}_hs_{self.hom_set_size}_depth_{self.depth}.th"
        self.model_filepath = self.data_root + self.file_name
        self.predictions_file = f"data/predictions/{self.species}/" + self.file_name
        self.labels_file = f"data/labels/{self.species}/" + self.file_name
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
        
        optimizer = th.optim.RMSprop(self.model.parameters(), lr=self.lr)
        best_loss = float('inf')
        best_mr = float('inf')
        #scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.milestones, gamma = self.gamma)
        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, 1e-6, 1e-1,70)
        bs = 4096*4
        zeros = th.zeros((bs), requires_grad = False).to(self.device)
        ones = th.ones((bs), requires_grad = False).to(self.device)
        size_datasets = [(k,len(v)) for k,v in self.training_datasets.get_gci_datasets().items()]
        total_samples = sum([v for k,v in size_datasets])
        weights = {k: v/total_samples for k,v in size_datasets}
        print(size_datasets)
        #print(self.model)
        for epoch in range(self.epochs): #trange(self.epochs):
            self.model.train()

            train_loss = 0
            loss = 0
            
            criterion = nn.MSELoss()
            # Notice how we use the ``training_datasets`` variable directly
            #and every element of it is a pair (GCI name, GCI tensor data).
            for gci_name, gci_dataset in self.training_datasets.get_gci_datasets().items():
                if len(gci_dataset) == 0:
                    continue
                if bs>len(gci_dataset):
                    replace = True
                else:
                    replace = False
                rand_index = np.random.choice(len(gci_dataset), size = bs, replace = replace)
                rand_index = th.as_tensor(rand_index, device = self.device)
                data = gci_dataset[rand_index]
                
                #pos_scores = th.mean(self.model(gci_dataset[:], gci_name))
                pos_scores = self.model(data, gci_name)
                assert pos_scores.shape == zeros.shape, f"{pos_scores.shape}, {zeros.shape}"
#                mse_loss = criterion(pos_scores, zeros)
#                mse_loss = th.mean(mse_loss)
#                loss += mse_loss
#                print(gci_name, mse_loss)
                if gci_name == "gci2" or True:
                    prots = [self.class_index_dict[p] for p in self.dataset.evaluation_classes]
                    idxs_for_negs = np.random.choice(len(self.class_index_dict), size = len(data), replace = True)
                    rand_index = th.tensor(idxs_for_negs, device = self.device)
                    #data = gci_dataset[:]
                    neg_data = th.cat([data[:,:-1], rand_index.unsqueeze(1)], dim = 1)                  
                    neg_scores = self.model(neg_data, gci_name, neg = True)
                    #mse_loss = criterion(neg_scores, th.ones(neg_scores.shape, requires_grad = False).to(self.device))
                    assert pos_scores.shape == neg_scores.shape, f"{pos_scores.shape}, {neg_scores.shape}"
                    log_loss = pos_scores - neg_scores + self.margin
                    log_loss = - th.mean(F.logsigmoid(-log_loss))

                    loss += log_loss*weights[gci_name]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.detach().item()

            loss = 0
            with th.no_grad():
                self.model.eval()
                valid_loss = 0
                gci2_data = self.validation_datasets.get_gci_datasets()["gci2"][:]
                loss = self.model(gci2_data, "gci2")
                loss = th.mean(loss)
                #mse_loss = th.mean(criterion(loss, th.zeros(loss.shape, requires_grad = False).to(self.device)))
                valid_loss += loss.detach().item()

            checkpoint = 100
            if best_loss > valid_loss:
                best_loss = valid_loss
                #th.save(self.model.state_dict(), self.model_filepath)
            if (epoch + 1) % checkpoint == 0:
                metrics = self.eval_valid()
                mr = metrics["mean_rank"]
                if best_mr > mr:
                    th.save(self.model.state_dict(), self.model_filepath)
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')



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
