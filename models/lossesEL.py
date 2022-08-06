import torch as th
import torch.nn as nn
import numpy as np
import random

def gci1_loss(objects, exponential_net,  embed_objects, class_reg, neg = False):

    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])
            
    loss = exponential_net(antecedents, consequents)
    reg_loss = class_reg(antecedents) + class_reg(consequents)
    assert loss.shape == reg_loss.shape, f"{loss.shape}, {reg_loss.shape}"
    return loss + reg_loss

def gci2_loss(objects, exp_net, slicing_net, embed_objects, embed_rels, class_reg,  neg = False ):
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    sliced_prod, prod_loss = slicing_net(relations, consequents, antecedents)
    exp_loss = exp_net(antecedents, sliced_prod)
    reg_loss = class_reg(antecedents) + class_reg(consequents) + class_reg(sliced_prod)
    return (exp_loss + prod_loss) + reg_loss

def gci3_loss(objects, exp_net, slicing_net, embed_objects, embed_rels, class_reg, neg = False):

    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])
                            
    sliced_prod, prod_loss = slicing_net(relations, antecedents, consequents)
    exp_loss = exp_net(sliced_prod, consequents)

    reg_loss = class_reg(antecedents) + class_reg(consequents) + class_reg(sliced_prod)
    return (prod_loss + exp_loss) + reg_loss
    
def gci4_loss(objects, exp_net, prod_net, embed_objects, class_reg, neg = False):

    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    prod, prod_loss = prod_net(antecedents_left, antecedents_right)
    exp_loss = exp_net(prod, consequents)
    reg_loss = class_reg(antecedents_left) + class_reg(antecedents_right) + class_reg(consequents)
    return prod_loss + exp_loss
