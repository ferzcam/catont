from mowl.nn.elmodule import ELModule
from .cat_net import Product, Coproduct, Negation, Existential, EntailmentHomSet
import models.lossesEL as L
import torch as th
import torch.nn as nn
import math
from .cat_net import norm
ACT = nn.Identity()

class CatELModule(ELModule):
    def __init__(self, num_classes, num_rels, hom_set_size, embedding_size, dropout = 0, depth = 1, activation = None):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_obj = num_classes 
        self.hom_set_size = hom_set_size
        self.dropout = dropout
        self.act = nn.Identity if activation is None else activation
        self.depth = depth
        
        self.norm_objects =  nn.LayerNorm(self.embedding_size)
        self.norm_relations = nn.LayerNorm(self.embedding_size)

        self.embed = nn.Embedding(self.num_obj, embedding_size)
        nn.init.uniform_(self.embed.weight, a=-1, b=1)
        self.embed.weight.data /= th.linalg.norm(self.embed.weight.data,axis=1).reshape(-1,1)

        self.embed_rel = nn.Embedding(num_rels, embedding_size)
        nn.init.uniform_(self.embed_rel.weight, a=-1, b=1)
        self.embed_rel.weight.data /= th.linalg.norm(self.embed_rel.weight.data,axis=1).reshape(-1,1)

        self.entailment_net = EntailmentHomSet(
            self.embedding_size,
            self.norm_objects,
            hom_set_size = self.hom_set_size,
            depth = self.depth,
            dropout = self.dropout)
        
        self.coprod_net = Coproduct(
            self.embedding_size,
            self.entailment_net,
            self.norm_objects,
            dropout = self.dropout)
        
        self.prod_net = Product(
            self.embedding_size,
            self.entailment_net,
            self.coprod_net,
            self.norm_objects,
            dropout = self.dropout)

        self.ex_net = Existential(
            self.embedding_size,
            self.prod_net,
            self.norm_objects,
            self.norm_relations,
            dropout = self.dropout)
        
        self.negation_net = Negation(
            self.embedding_size,
            self.entailment_net,
            self.prod_net,
            self.coprod_net,
            self.norm_objects,
            dropout = self.dropout)

        self.dummy_param = nn.Parameter(th.empty(0))
        
        
        # Embedding network for the ontology ojects
        self.net_object = nn.Sequential(
            self.embed,
#            nn.Linear(embedding_size, embedding_size),
#            self.norm_objects,
#            ACT,

        )

        # Embedding network for the ontology relations
        self.net_rel = nn.Sequential(
            self.embed_rel,
#            nn.Linear(embedding_size, embedding_size),
#            self.norm_relations,
#            ACT

        )
        
    def class_reg(self, x):
        
        res = th.abs((th.linalg.norm(x, axis=1) - 1)) #force embedding vector to have size less than 1
#        res = th.zeros(res.shape, device = res.device)
        return res

    def intersection_equivalence(self, a, b, c):
        a = self.net_object(a)
        b = self.net_object(b)
        intersection, *_ = self.prod_net(a, b)
        c = self.net_object(c)
        return norm(intersection,c)
    
    
    def gci0_loss(self, data, neg = False, indices= None):
        device = self.dummy_param.device
        return L.gci1_loss(data, self.entailment_net, self.net_object, self.class_reg, neg = neg)

    def gci1_loss(self, data, neg = False, indices= None):
        device = self.dummy_param.device
        return L.gci4_loss(data, self.entailment_net, self.prod_net, self.net_object, self.class_reg, neg = neg)

    def gci2_loss(self, data, neg = False, indices= None):
        device = self.dummy_param.device
        return L.gci2_loss(data, self.entailment_net, self.ex_net, self.net_object, self.net_rel, self.class_reg, neg=neg)

    def gci3_loss(self, data, neg = False, indices= None):
        device = self.dummy_param.device
        return L.gci3_loss(data, self.entailment_net, self.ex_net, self.net_object, self.net_rel, self.class_reg, neg = neg)

