import torch as th
import torch.nn as nn
import numpy as np
import random


def negation_loss(objects, negation_net, embed_objects, class_reg, neg = False):
    objs = embed_objects(objects[:, 0])
    bottom = embed_objects(objects[:, 1])
    top = embed_objects(objects[:, 2])

    _, neg_loss, nb_ents = negation_net(objs, bottom, top)
    reg_loss = class_reg(objs)
    reg_loss += class_reg(bottom)
    reg_loss += class_reg(top)

    assert neg_loss.shape == reg_loss.shape, print(f"{neg_loss.shape}, {reg_loss.shape}")
    return neg_loss+reg_loss, None, nb_ents

def gci1_loss(objects, exponential_net,  embed_objects, class_reg, neg = False,  num_objects = None, neg = False, indices = None):

    antecedents = embed_objects(objects[:, 0])
    consequents = embed_objects(objects[:, 1])
    nb_ents = 1

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        neg_loss_1 = 0
        #neg_loss_1 = exponential_net(consequents, antecedents)
        neg_loss_2 = exponential_net(antecedents, embed_negs, neg=neg, indices = indices)

        reg_loss = class_reg(antecedents) + class_reg(embed_negs)

        loss = neg_loss_2
        assert loss.shape == reg_loss.shape, print(f"{loss.shape}, {reg_loss.shape}")
        return loss +reg_loss, nb_ents
                                        
    else:
        loss, indices = exponential_net(antecedents, consequents, get_indices = True)
        reg_loss = class_reg(antecedents) + class_reg(consequents)
        assert loss.shape == reg_loss.shape, f"{loss.shape}, {reg_loss.shape}"
        return loss + reg_loss, indices, nb_ents

def gci2_loss(objects, exp_net, slicing_net, embed_objects, embed_rels, class_reg,  neg = False, num_objects = None, device = "cpu", indices = None, idx_for_negs = None):
    
    antecedents = embed_objects(objects[:, 0])
    relations = embed_rels(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(idx_for_negs, size = len(objects))).to(device)
        #negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        consequents  = embed_objects(negs)

    sliced_prod, prod_loss, nb_ents_slicing = slicing_net(relations, consequents, antecedents)

    if neg:
        exp_loss = exp_net(antecedents, sliced_prod, neg=neg, indices = indices)
    else:
        exp_loss, indices = exp_net(antecedents, sliced_prod, neg=neg, get_indices = True)
        
    reg_loss = class_reg(antecedents) + class_reg(consequents) + class_reg(sliced_prod)
    return (exp_loss + prod_loss) + reg_loss, indices, nb_ents_slicing + 1

def gci3_loss(objects, exp_net, slicing_net, embed_objects, embed_rels, class_reg, neg = False, num_objects = None, device = "cpu", indices = None):

    relations = embed_rels(objects[:, 0])
    antecedents = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        consequents = embed_negs

    sliced_prod, prod_loss, nb_ents_slicing = slicing_net(relations, antecedents, consequents)

    if neg:
        exp_loss = exp_net(sliced_prod, consequents,neg=neg, indices = indices)
    else:
        exp_loss, indices = exp_net(sliced_prod, consequents,neg=neg, get_indices = True)

    reg_loss = class_reg(antecedents) + class_reg(consequents) + class_reg(sliced_prod)
    return (prod_loss + exp_loss) + reg_loss, indices, nb_ents_slicing + 1


    
def gci4_loss(objects, exp_net, prod_net, embed_objects, class_reg, neg = False, num_objects = None, device = "cpu", indices = None):

    antecedents_left = embed_objects(objects[:, 0])
    antecedents_right = embed_objects(objects[:, 1])
    consequents = embed_objects(objects[:, 2])

    prod, prod_loss, nb_ents_prod = prod_net(antecedents_left, antecedents_right)
    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        consequents_neg = embed_negs
        exp_loss_neg = exp_net(prod, consequents_neg, neg=neg, indices = indices)
        neg_loss_1 = prod_loss + exp_loss_neg
        
        # negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        # embed_negs = embed_objects(negs)
        # prod_neg_1, prod_loss_1 = prod_net(antecedents_left, embed_negs)
        # exp_loss = exp_net(prod_neg_1, consequents)
        # neg_loss_2 = prod_loss_1 + exp_loss

        # negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        # embed_negs = embed_objects(negs)
        # prod_neg_2, prod_loss_2 = prod_net(embed_negs, antecedents_right)
        # exp_loss = exp_net(prod_neg_2, consequents)
        # neg_loss_3 = prod_loss_2 + exp_loss

        reg_loss = class_reg(antecedents_left) + class_reg(consequents_right) + class_reg(consequents_neg)
        return neg_loss_1+reg_loss, nb_ents_prod + 1
        #return (neg_loss_2 + neg_loss_3)/2
        #return (neg_loss_1 + neg_loss_2 + neg_loss_3)/3
        
    else:
        exp_loss, indices = exp_net(prod, consequents,neg=neg, get_indices = True)
        reg_loss = class_reg(antecedents_left) + class_reg(consequents_right) + class_reg(consequents)
        return prod_loss + exp_loss + neg_loss, indices, nb_ents_prod + 1

    
#    prod = (antecedents_left + antecedents_right)/2
        
    

#    print(f"prod: {th.mean(prod_loss)} \t exp: {th.mean(exp_loss)}")
#    return prod_loss + exp_loss

def gci5_loss(objects, exp_net, prod_net, embed_objects, class_reg, neg = False, num_objects = None, device = "cpu", indices = None):

    antecedents = embed_objects(objects[:, 0])
    consequents_left = embed_objects(objects[:, 1])
    consequents_right = embed_objects(objects[:, 2])

    prod, prod_loss, nb_ents_prod = prod_net(consequents_left, consequents_right)
    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        antecedents_neg = embed_negs
        exp_loss_neg = exp_net(antecedents_neg, prod,neg=neg, indices = indices)
        neg_loss_1 = prod_loss + exp_loss_neg
        
        # negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        # embed_negs = embed_objects(negs)
        # prod_neg_1, prod_loss_1 = prod_net(consequents_left, embed_negs)
        # exp_loss = exp_net(antecedents, prod_neg_1)
        # neg_loss_2 = prod_loss_1 + exp_loss

        # negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        # embed_negs = embed_objects(negs)
        # prod_neg_2, prod_loss_2 = prod_net(embed_negs, consequents_right)
        # exp_loss = exp_net(antecedents, prod_neg_2)
        # neg_loss_3 = prod_loss_2 + exp_loss

        reg_loss = class_reg(antecedents_neg) + class_reg(consequents_left) + class_reg(consequents_right)
        return neg_loss_1, nb_ents_prod + 1
        #return (neg_loss_2 + neg_loss_3)/2
        #return (neg_loss_1 + neg_loss_2 + neg_loss_3)/3
        
    else:
        exp_loss, indices = exp_net(antecedents, prod,neg=neg, get_indices = True)
        reg_loss = class_reg(antecedents) + class_reg(consequents_left) + class_reg(consequents_right)
        return (prod_loss + exp_loss) + reg_loss, indices, nb_ents_prod + 1

    
#    prod = (consequents_left + consequents_right)/2
        
    

#    print(f"prod: {th.mean(prod_loss)} \t exp: {th.mean(exp_loss)}")
#    return prod_loss + exp_loss


def gci6_loss(objects, exp_net, slicing_net, prod_net, embed_objects, embed_rels, class_reg, neg = False, num_objects = None, device = "cpu", indices = None):

    antecedents_left = embed_objects(objects[:,0])
    relations = embed_rels(objects[:, 1])
    fillers = embed_objects(objects[:, 2])
    consequents = embed_objects(objects[:, 3])

    if neg:
        negs = th.tensor(np.random.choice(num_objects, size = len(objects))).to(device)
        embed_negs = embed_objects(negs)
        consequents = embed_negs

    sliced_prod, sliced_prod_loss, nb_ents_slicing = slicing_net(relations, fillers, antecedents_left, consequents)
    prod, prod_loss, nb_ents_prod = prod_net(antecedents_left, sliced_prod)

    if neg:
        exp_loss = exp_net(prod, consequents,neg=neg, indices = indices)
    else:
        exp_loss, indices = exp_net(prod, consequents,neg=neg, get_indices = True)

    reg_loss = class_reg(antecedents_left) + class_reg(fillers) + class_reg(consequents) + class_reg(sliced_prod)
    return (prod_loss + sliced_prod_loss + exp_loss) + reg_loss, indices, nb_ents_slicing + nb_ents_prod + 1

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
