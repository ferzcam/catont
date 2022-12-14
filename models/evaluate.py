from mowl.evaluation.base import AxiomsRankBasedEvaluator
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import logging
import numpy as np
from scipy.stats import rankdata
import torch as th
import random
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from mowl.evaluation.base import compute_rank_roc
class CatEmbeddingsPPIEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            eval_method,
            axioms_to_filter,
            class_name_indexemb,
            rel_name_indexemb,
            device = "cpu",
            verbose = False,
            num_points = None
    ):
        self.num_points = num_points
        super().__init__(eval_method, axioms_to_filter, device, verbose)

        self.class_name_indexemb = class_name_indexemb
        self.relation_name_indexemb = rel_name_indexemb
        
        self._loaded_training_scores = False
        self._loaded_eval_data = False
        self._loaded_ht_data = False

        
        
    def _load_head_tail_entities(self):
        if self._loaded_ht_data:
            return

        ents, _ = Edge.getEntitiesAndRelations(self.axioms)
        ents_filter, _ = Edge.getEntitiesAndRelations(self.axioms_to_filter)

        entities = list(set(ents) | set(ents_filter))

        self.head_entities = set()
        for e in entities:
            if e in self.class_name_indexemb:
                self.head_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)       

        self.tail_entities = set()
        for e in entities:
            if e in self.class_name_indexemb:
                self.tail_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)

        self.head_name_indexemb = {k: self.class_name_indexemb[k] for k in self.head_entities}
        self.tail_name_indexemb = {k: self.class_name_indexemb[k] for k in self.tail_entities}

        self.head_indexemb_indexsc = {v: k for k, v in enumerate(self.head_name_indexemb.values())}
        self.tail_indexemb_indexsc = {v: k for k, v in enumerate(self.tail_name_indexemb.values())}
                
        self._loaded_ht_data = True

        

    def _load_training_scores(self):
        if self._loaded_training_scores:
            return self.training_scores

        self._load_head_tail_entities()
        
        training_scores = np.ones((len(self.head_entities), len(self.tail_entities)), dtype=np.int32)

        if self._compute_filtered_metrics:
        # careful here: c must be in head entities and d must be in tail entities
            for axiom in self.axioms_to_filter:
                c, _, d = axiom.astuple()
                if (not c in self.head_entities) or not (d in self.tail_entities):
                    continue
            
                c, d = self.head_name_indexemb[c], self.tail_name_indexemb[d]
                c, d = self.head_indexemb_indexsc[c], self.tail_indexemb_indexsc[d]
            
                training_scores[c, d] = 10000

            logging.info("Training scores created")

        self._loaded_training_scores = True
        return training_scores
        
        
    def _init_axioms(self, axioms):

        if axioms is None:
            return None
        
        projector = projector_factory("taxonomy_rels", relations = ["http://interacts"])

        edges = projector.project(axioms)

        if self.num_points is None:
            return edges # List of Edges
        else:
            return random.sample(edges, self.num_points)

    def _init_axioms_to_filter(self, axioms):

        if axioms is None:
            return None
        
        projector = projector_factory("taxonomy_rels", relations = ["http://interacts"])

        edges = projector.project(axioms)
        return edges # List of Edges

        
    def compute_axiom_rank(self, axiom):
        
        self.training_scores = self._load_training_scores()

        c, r, d = axiom.astuple()
        
        if not (c in self.head_entities) or not (d in self.tail_entities):
            return None, None, None

        # Embedding indices
        c_emb_idx, d_emb_idx = self.head_name_indexemb[c], self.tail_name_indexemb[d]

        # Scores matrix labels
        c_sc_idx, d_sc_idx = self.head_indexemb_indexsc[c_emb_idx], self.tail_indexemb_indexsc[d_emb_idx]

        r = self.relation_name_indexemb[r]

        data = [[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]
        data = np.array(data)
        data = th.as_tensor(data, device=self.device)

        res = self.eval_method(data).squeeze().cpu().detach().numpy()
        
        #self.testing_predictions[c_sc_idx, :] = res                                                                                
        index = rankdata(res, method='average')
        rank = index[d_sc_idx]

        findex = rankdata((res * self.training_scores[c_sc_idx, :]), method='average')
        frank = findex[d_sc_idx]

        return rank, frank, len(self.tail_entities)

    def get_metrics(self):
        return self._metrics


class CatEmbeddingsPPIEvaluatorTune(CatEmbeddingsPPIEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_axioms(self, axioms):
        return axioms

    def _init_axioms_to_filter(self, axioms):
        return axioms
        
    
class CatEmbeddingsIntersectionEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            eval_method,
            class_name_indexemb,
            device = "cpu",
            verbose = False,
            num_points = None
    ):

        super().__init__(eval_method, None, device, verbose)
        self.num_points = num_points
        self.class_name_indexemb = class_name_indexemb
        
        self._loaded_eval_data = False
        self._loaded_ht_data = False        
    
    def _init_axioms(self, axioms):
        axioms = axioms[:].cpu().detach().numpy().tolist()
        axioms = [tuple(x) for x in axioms]
        if self.num_points is None:
            return axioms
        else:
            return random.sample(axioms, self.num_points)

    def _init_axioms_to_filter(self, axioms):
        return axioms



    def get_predictions(self, samples=None, save = True):
        logging.info("Computing prediction on %s", str(self.device))
    
#        if samples is None:
#            logging.info("No data points specified. Proceeding to compute predictions on test set")
#            model.load_state_dict(th.load( self.model_filepath))
#            model = model.to(self.device)
#            _, _, test_nf3, _ = self.test_nfs

#            eval_data = test_nf3

 #       else:
 #           eval_data = samples

        test_model = TestModuleIntersection(self.eval_method).to(self.device)

        preds = np.zeros((len(samples), len(self.class_name_indexemb)), dtype=np.float32)
            
        test_dataset = TestDatasetIntersection(samples, self.class_name_indexemb, self.left_side_dict)
    
        bs = 8
        test_dl = DataLoader(test_dataset, batch_size = bs)

        
        for idxs, batch in tqdm(test_dl):

            #idxs = []
            #for l,r in zip(idx_l, idx_r):
                #l = l.detach().item()
                #r = r.detach().item()
                #idxs.append(self.left_side_dict[(l,r)])
                
            res = test_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            preds[idxs,:] = res

        

#        if save:
#            with open(self.predictions_file, "wb") as f:
#                pkl.dump(preds, f)

        return preds

    
    def compute_axiom_rank(self, axiom, predictions):
        
        l, r, d = axiom

        lr = self.left_side_dict[(l,r)]

#        data = th.tensor([[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]).to(self.device)

        res = predictions[lr,:]
#        res = self.eval_method(data).squeeze().cpu().detach().numpy()
        
        #self.testing_predictions[c_sc_idx, :] = res                                                                                
        index = rankdata(res, method='average')
        first = res[np.where(index == 1)[0]]
        last  = res[np.where(index == len(index))[0]]
        rank = index[d]

        #print(rank, res[d], first, last)

        findex = rankdata((res), method='average')
        frank = findex[d]

        return rank, frank, len(self.class_name_indexemb)


    def __call__(self, axioms):
        self.axioms = self._init_axioms(axioms)
        self.left_side_dict = {v[:-1]:k for k,v in enumerate(self.axioms)}
        predictions = self.get_predictions(self.axioms)
        tops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        ftops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        mean_rank = 0
        fmean_rank = 0
        ranks = {}
        franks = {}
        
        n = 0
        for axiom in tqdm(self.axioms, total = len(self.axioms)):
            rank, frank, worst_rank = self.compute_axiom_rank(axiom, predictions)
            
            if rank is None:
                continue
             
            n = n+1
            for top in tops:
                if rank <= top:
                    tops[top] += 1
                     
            mean_rank += rank

            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            if self._compute_filtered_metrics:
                for ftop in ftops:
                    if frank <= ftop:
                        ftops[ftop] += 1

                if rank not in franks:
                    franks[rank] = 0
                franks[rank] += 1

                fmean_rank += frank

        tops = {k: v/n for k, v in tops.items()}
        ftops = {k: v/n for k, v in ftops.items()}

        mean_rank, fmean_rank = mean_rank/n, fmean_rank/n

        rank_auc = compute_rank_roc(ranks, worst_rank)
        frank_auc = compute_rank_roc(franks, worst_rank)

        self._metrics = {f"hits@{k}": tops[k] for k in tops}
        self._metrics["mean_rank"] = mean_rank
        self._metrics["rank_auc"] = rank_auc
        self._fmetrics = {f"hits@{k}": ftops[k] for k in ftops}
        self._fmetrics["mean_rank"] = fmean_rank
        self._fmetrics["rank_auc"] = frank_auc

        return

    def get_metrics(self):
        return self._metrics
    

class CatEmbeddingsSubsumptionAsIntersectionEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            class_name_indexemb,
            entailment,
            device = "cpu",
            verbose = False
    ):

        eval_method = None
        super().__init__(eval_method, None, device, verbose)

        self.class_name_indexemb = class_name_indexemb
        self.entailment = entailment
                        
        self._loaded_eval_data = False
        self._loaded_ht_data = False        

        self.bs = 8
        self.top = "http://www.w3.org/2002/07/owl#Thing"
        self.bottom = "http://www.w3.org/2002/07/owl#Nothing"

        self.top_idx = th.zeros(len(class_name_indexemb)*self.bs).to(self.device).long()
        self.top_idx += self.class_name_indexemb[self.top]

        self.bottom_idx = th.zeros(len(class_name_indexemb)*self.bs).to(self.device).long()
        self.bottom_idx += self.class_name_indexemb[self.bottom]

    def _init_axioms(self, axioms):
        return axioms
                                                     

    def _init_axioms_to_filter(self, axioms):
        return axioms



    def get_predictions(self, samples=None, save = True):
        logging.info("Computing prediction on %s", str(self.device))
    
#        if samples is None:
#            logging.info("No data points specified. Proceeding to compute predictions on test set")
#            model.load_state_dict(th.load( self.model_filepath))
#            model = model.to(self.device)
#            _, _, test_nf3, _ = self.test_nfs

#            eval_data = test_nf3

 #       else:
 #           eval_data = samples

        test_model = TestModuleSubsumptionAsIntersection(self.entailment, self.embeddings, self.bottom_idx, self.top_idx).to(self.device)

        preds = np.zeros((len(samples), len(self.embeddings)), dtype=np.float32)
            
        test_dataset = TestDatasetSubsumptionAsIntersection(samples, self.class_name_indexemb, self.left_side_dict)
    
        
        test_dl = DataLoader(test_dataset, batch_size = self.bs)

        
        for idxs, batch in tqdm(test_dl):

            #idxs = []
            #for l,r in zip(idx_l, idx_r):
                #l = l.detach().item()
                #r = r.detach().item()
                #idxs.append(self.left_side_dict[(l,r)])
                
            res = test_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            preds[idxs,:] = res

        

#        if save:
#            with open(self.predictions_file, "wb") as f:
#                pkl.dump(preds, f)

        return preds

    
    def compute_axiom_rank(self, axiom, predictions):
        
        l, r = axiom

        lr = self.left_side_dict[(l,r)]

#        data = th.tensor([[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]).to(self.device)

        res = predictions[lr,:]
#        res = self.eval_method(data).squeeze().cpu().detach().numpy()
        
        #self.testing_predictions[c_sc_idx, :] = res                                                                                
        index = rankdata(res, method='average')
        first = res[np.where(index == 1)[0]]
        last  = res[np.where(index == len(index))[0]]
        rank = index[self.class_name_indexemb[self.bottom]]

        #print(rank, res[d], first, last)

        findex = rankdata((res), method='average')
        frank = findex[self.class_name_indexemb[self.bottom]]

        return rank, frank, len(self.embeddings)


    def __call__(self, axioms, embeddings, init_axioms = False):
        self.embeddings = embeddings
        self.left_side_dict = {v:k for k,v in enumerate(axioms)}
        predictions = self.get_predictions(axioms)
        tops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        ftops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        mean_rank = 0
        fmean_rank = 0
        ranks = {}
        franks = {}
        
        n = 0
        for axiom in tqdm(axioms):
            rank, frank, worst_rank = self.compute_axiom_rank(axiom, predictions)
            
            if rank is None:
                continue
             
            n = n+1
            for top in tops:
                if rank <= top:
                    tops[top] += 1
                     
            mean_rank += rank

            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            if self._compute_filtered_metrics:
                for ftop in ftops:
                    if frank <= ftop:
                        ftops[ftop] += 1

                if rank not in franks:
                    franks[rank] = 0
                franks[rank] += 1

                fmean_rank += frank

        tops = {k: v/n for k, v in tops.items()}
        ftops = {k: v/n for k, v in ftops.items()}

        mean_rank, fmean_rank = mean_rank/n, fmean_rank/n

        rank_auc = compute_rank_roc(ranks, worst_rank)
        frank_auc = compute_rank_roc(franks, worst_rank)

        self._metrics = {f"hits@{k}": tops[k] for k in tops}
        self._metrics["mean_rank"] = mean_rank
        self._metrics["rank_auc"] = rank_auc
        self._fmetrics = {f"hits@{k}": ftops[k] for k in ftops}
        self._fmetrics["mean_rank"] = fmean_rank
        self._fmetrics["rank_auc"] = frank_auc

        return

class CatEmbeddingsSubsumptionEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            class_name_indexemb,
            entailment,
            device = "cpu",
            verbose = False
    ):

        eval_method = None
        super().__init__(eval_method, None, device, verbose)

        self.class_name_indexemb = class_name_indexemb
        self.class_indexemb_indexsc = {v:k for k, v in enumerate(class_name_indexemb.values())}
        self.entailment = entailment
                        
        self._loaded_eval_data = False
        self._loaded_ht_data = False        

        self.bs = 8
                                

    def _init_axioms(self, axioms):
        return axioms
                                                     

    def _init_axioms_to_filter(self, axioms):
        return axioms



    def get_predictions(self, samples=None, save = True):
        logging.info("Computing prediction on %s", str(self.device))
    
#        if samples is None:
#            logging.info("No data points specified. Proceeding to compute predictions on test set")
#            model.load_state_dict(th.load( self.model_filepath))
#            model = model.to(self.device)
#            _, _, test_nf3, _ = self.test_nfs

#            eval_data = test_nf3

 #       else:
 #           eval_data = samples

        test_model = TestModuleSubsumption(self.entailment, self.embeddings).to(self.device)

        preds = np.zeros((len(self.embeddings), len(self.embeddings)), dtype=np.float32)
            
        test_dataset = TestDatasetSubsumption(samples, self.class_name_indexemb, self.class_indexemb_indexsc)
    
        
        test_dl = DataLoader(test_dataset, batch_size = self.bs)

        
        for idxs, batch in tqdm(test_dl):

            #idxs = []
            #for l,r in zip(idx_l, idx_r):
                #l = l.detach().item()
                #r = r.detach().item()
                #idxs.append(self.left_side_dict[(l,r)])
                
            res = test_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            preds[idxs,:] = res

        

#        if save:
#            with open(self.predictions_file, "wb") as f:
#                pkl.dump(preds, f)

        return preds

    
    def compute_axiom_rank(self, axiom, predictions):
        
        l, r = axiom

        l_idx = self.class_indexemb_indexsc[l]
        

#        data = th.tensor([[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]).to(self.device)

        res = predictions[l,:]
#        res = self.eval_method(data).squeeze().cpu().detach().numpy()
        
        #self.testing_predictions[c_sc_idx, :] = res                                                                                
        index = rankdata(res, method='average')
        first = res[np.where(index == 1)[0]]
        last  = res[np.where(index == len(index))[0]]
        rank = index[self.class_indexemb_indexsc[r]]

        #print(rank, res[d], first, last)

        findex = rankdata((res), method='average')
        frank = findex[self.class_indexemb_indexsc[r]]

        return rank, frank, len(self.embeddings)


    def __call__(self, axioms, embeddings, init_axioms = False):
        self.embeddings = embeddings
        
        predictions = self.get_predictions(axioms)
        tops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        ftops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        mean_rank = 0
        fmean_rank = 0
        ranks = {}
        franks = {}
        
        n = 0
        for axiom in tqdm(axioms):
            rank, frank, worst_rank = self.compute_axiom_rank(axiom, predictions)
            
            if rank is None:
                continue
             
            n = n+1
            for top in tops:
                if rank <= top:
                    tops[top] += 1
                     
            mean_rank += rank

            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            if self._compute_filtered_metrics:
                for ftop in ftops:
                    if frank <= ftop:
                        ftops[ftop] += 1

                if rank not in franks:
                    franks[rank] = 0
                franks[rank] += 1

                fmean_rank += frank

        tops = {k: v/n for k, v in tops.items()}
        ftops = {k: v/n for k, v in ftops.items()}

        mean_rank, fmean_rank = mean_rank/n, fmean_rank/n

        rank_auc = compute_rank_roc(ranks, worst_rank)
        frank_auc = compute_rank_roc(franks, worst_rank)

        self._metrics = {f"hits@{k}": tops[k] for k in tops}
        self._metrics["mean_rank"] = mean_rank
        self._metrics["rank_auc"] = rank_auc
        self._fmetrics = {f"hits@{k}": ftops[k] for k in ftops}
        self._fmetrics["mean_rank"] = fmean_rank
        self._fmetrics["rank_auc"] = frank_auc

        return




class TestModulePPI(nn.Module):
    def __init__(self, method):
        super().__init__()

        self.method = method        

    def forward(self, x):
        bs, num_prots, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        x, *_ = self.method(x)

        x = x.reshape(bs, num_prots)

        return x



class TestDatasetPPI(IterableDataset):
    def __init__(self, data, class_name_indexemb, head_indexemb_indexsc, tail_indexemb_indexsc, r):
        super().__init__()
        self.data = data
        self.class_name_indexemb = class_name_indexemb
        self.head_indexemb_indexsc = head_indexemb_indexsc
        self.len_data = len(data)

        self.predata = np.array([[0, r, x] for x in tail_indexemb_indexsc])
        
    def get_data(self):
        for edge in self.data:
            c, r, d = edge.astuple()
            c, d = self.class_name_indexemb[c], self.class_name_indexemb[d]
            c_emb = c #.detach().item()
            c = self.head_indexemb_indexsc[c]
            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_emb
            
            tensor = new_array
            yield c, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data


    
class TestModuleIntersection(nn.Module):
    def __init__(self, method):
        super().__init__() 
        self.method = method

    def forward(self, x):
        bs, num_axioms, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        c_left = x[:,0]
        c_right = x[:,1]
        d = x[:,2]

        scores = self.method(c_left, c_right, d)
        x = scores.reshape(bs, num_axioms)

        return x



class TestDatasetIntersection(IterableDataset):
    def __init__(self, data, class_name_indexemb, indices_emb_indexsc):
        super().__init__()
        self.data = data
        self.len_data = len(data)
        self.class_name_indexemb = class_name_indexemb
        self.indices_emb_indexsc = indices_emb_indexsc
        self.predata = np.array([[-1,-1, x] for x in list(class_name_indexemb.values())])
        
    def get_data(self):
        for axiom in self.data:
            c_left, c_right, d = axiom

            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_left
            new_array[:,1] = c_right
            
            tensor = new_array
            idx = self.indices_emb_indexsc[(c_left, c_right)]
            yield idx, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data



class TestModuleSubsumptionAsIntersection(nn.Module):
    def __init__(self, entailment, embeddings, bottom, top):
        super().__init__()

        self.bottom = bottom
        self.top = top
        self.entailment = entailment
        embeddings = list(embeddings.values())
        self.embeddings = nn.Embedding(len(embeddings), len(embeddings[0]))

    def forward(self, x):
        bs, num_axioms, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        c_left = x[:,0]
        c_right = x[:,1]
        d = x[:,2]
                                                                                        
        scores = self.entailment(c_left, c_right, d, self.embeddings(self.bottom), self.embeddings(self.top))
        
        x = scores.reshape(bs, num_axioms)

        return x



class TestDatasetSubsumptionAsIntersection(IterableDataset):
    def __init__(self, data, class_name_indexemb, indices_emb_indexsc):
        super().__init__()
        self.data = data
        self.len_data = len(data)
        self.class_name_indexemb = class_name_indexemb
        self.indices_emb_indexsc = indices_emb_indexsc
        self.predata = np.array([[-1,-1, x] for x in list(class_name_indexemb.values())])
        
    def get_data(self):
        for axiom in self.data:
            c_left, c_right = axiom
#            c_left, c_right = self.class_name_indexemb[c_left], self.class_name_indexemb[c_right]
            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_left
            new_array[:,1] = c_right
            
            tensor = new_array
            idx = self.indices_emb_indexsc[(c_left, c_right)]
            yield idx, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data




class TestModuleSubsumption(nn.Module):
    def __init__(self, entailment, embeddings):
        super().__init__()

        self.entailment = entailment
        embeddings = list(embeddings.values())
        self.embeddings = nn.Embedding(len(embeddings), len(embeddings[0]))

    def forward(self, x):
        bs, num_axioms, ents = x.shape
        assert 2 == ents
        x = x.reshape(-1, ents)

        c_left = x[:,0]
        c_right = x[:,1]
        
                                                                                        
        scores = self.entailment(c_left, c_right)
        
        x = scores.reshape(bs, num_axioms)

        return x



class TestDatasetSubsumption(IterableDataset):
    def __init__(self, data, class_name_indexemb, indices_emb_indexsc):
        super().__init__()
        self.data = data
        self.len_data = len(data)
        self.class_name_indexemb = class_name_indexemb
        self.indices_emb_indexsc = indices_emb_indexsc
        self.predata = np.array([[-1, x] for x in list(class_name_indexemb.values())])
        
    def get_data(self):
        for axiom in self.data:
            c_left, c_right = axiom
#            c_left, c_right = self.class_name_indexemb[c_left], self.class_name_indexemb[c_right]
            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_left
                        
            tensor = new_array
            idx = self.indices_emb_indexsc[c_left]
            yield idx, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data


    
