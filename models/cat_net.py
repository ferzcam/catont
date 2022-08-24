import torch as th
import torch.nn as nn
import random

HID_ACT = nn.ReLU()
FINAL_ACT = nn.Identity()



def norm(a,b, dim = 1):
    n = a.shape[1]

    assert a.shape == b.shape, f"Shape A: {a.shape}, Shape B: {b.shape}"
    sqe = (a-b)**2
    mse = th.sum(sqe, dim =1)/n
    return mse
        
def norm_(a, b, dim = 1, neg = False):
    x = a*b
    x = th.relu(th.sum(x, dim = dim))
    x = th.sigmoid(x)
    return 1 - x

def rand_tensor(shape, device):
    x = th.rand(shape).to(device)
    x = (x*2) - 1
    return x

def assert_shapes(shapes):
    assert len(set(shapes)) == 1


class ObjectGenerator1(nn.Module):
    def __init__(self, embedding_size, dropout, norm_layer):
        super().__init__()

        self.transform = nn.Sequential(

            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            HID_ACT,
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            #norm_layer,
            FINAL_ACT
        )

    def forward(self, obj):
        x = self.transform(obj)
        #x = x+mean
        return x

class ObjectGenerator2(nn.Module):
    def __init__(self, embedding_size, dropout, norm_layer):
        super().__init__()

        self.transform = nn.Sequential(

            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            HID_ACT,
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),

            #norm_layer,
            FINAL_ACT
        )

    def forward(self, left,right):
        #mean = (left+right)/2

        x = th.cat([left,right], dim = 1)
        x = self.transform(x)
        #x = x+mean
        return x




    
class Negation(nn.Module):
    """Representation of entailments for negation
    """

    def __init__(self, embedding_size, entailment_net, product_net, coproduct_net, norm_layer,dropout = 0):
        super().__init__()
        self.embedding_size = embedding_size
        self.entailment_net = entailment_net
        self.product_net = product_net
        self.coproduct_net = coproduct_net
        self.dropout = dropout

        self.neg = ObjectGenerator1(self.embedding_size, self.dropout, norm_layer)
        self.extra_obj = ObjectGenerator1(self.embedding_size, self.dropout, norm_layer)

        
    def forward(self, x, bottom, top):
        neg_x = self.neg(x)
        loss = 0

        # C and not C ----> bottom
        prod, prod_loss = self.product_net(x, neg_x, coproduct = False)
        loss += prod_loss
        loss += self.entailment_net(prod, bottom)
        
        # top ----> C or not C
        coprod, coprod_loss = self.coproduct_net(x, neg_x)
        loss += coprod_loss
        loss += self.entailment_net(top, coprod)
        
        # double negation
        neg_neg_x = self.neg(neg_x)
        loss += norm(neg_neg_x, x)        
        return neg_x, loss
    
class Product(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size, entailment_net, coproduct_net, norm_layer, dropout = 0):
        super().__init__()
        
        self.coproduct_net = coproduct_net
        self.prod = ObjectGenerator2(embedding_size, dropout, norm_layer)
        self.up = ObjectGenerator2(embedding_size, dropout, norm_layer)

        self.pi1 = entailment_net
        self.pi2 = entailment_net
        self.m   = entailment_net
        self.p1  = entailment_net
        self.p2  = entailment_net
        self.ent = entailment_net

    def forward(self, left, right, coproduct=True):

        product = self.prod(left, right)
        up = self.up(left, right)
        
        loss = 0
        #Diagram losses
        loss += self.p1(up, left)
        loss += self.m(up, product)
        loss += self.p2(up, right)
        loss += self.pi1(product, left)
        loss += self.pi2(product, right)

        #Distributivity over conjunction: C and (D or E) entails (C and D) or (C and E)
        #extra_obj = rand_tensor(product.shape, product.device)
        #right_or_extra, roe_loss = self.coproduct_net(right, extra_obj)
        #antecedent = self.prod(th.cat([left, right_or_extra], dim = 1))
        #antecedent = self.prod(left, right_or_extra)
        
        #left_and_extra = self.prod(th.cat([left, extra_obj], dim = 1))
        #left_and_extra = self.prod(left, extra_obj)
        #consequent, cons_loss = self.coproduct_net(product, left_and_extra)

        #loss += roe_loss
        #loss += cons_loss
        #loss += self.ent(antecedent, consequent)

        # (A and B) entails (A or B)
        coprod, coprod_loss = self.coproduct_net(left, right)
        if coproduct:
            loss += coprod_loss
            
        loss += self.ent(product, coprod)
        return product, loss



    
class Coproduct(nn.Module):
    """Representation of the categorical diagram of the product. 
    """

    def __init__(self, embedding_size, entailment_net, norm_layer, dropout = 0):
        super().__init__()
        
        self.coprod = ObjectGenerator2(embedding_size, dropout, norm_layer)
        self.down = ObjectGenerator2(embedding_size, dropout, norm_layer)

        self.iota1 = entailment_net
        self.iota2 = entailment_net
        self.m   = entailment_net
        self.i1  = entailment_net
        self.i2  = entailment_net
        

    def forward(self, left, right, up=None):

        coproduct = self.coprod(left, right)
        down = self.down(left, right)
        
        loss = 0
        loss += self.i1(left, down)
        loss += self.m(coproduct, down)
        loss += self.i2(right, down)
        loss += self.iota1(left, coproduct)
        loss += self.iota2(right, coproduct)
        
        return coproduct, loss

class MorphismBlock(nn.Module):

    def __init__(self, embedding_size, dropout):
        super().__init__()

        self.fc = nn.Linear(embedding_size, embedding_size)
        self.bn = nn.LayerNorm(embedding_size)
        self.act = HID_ACT
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        skip = x
        
        x = self.fc(x)
        x = self.bn(x)
        x = x + skip
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class EntailmentHomSet(nn.Module):
    def __init__(self, embedding_size, norm_layer, hom_set_size = 1, depth = 1, dropout = 0):
        super().__init__()
        embedding_size = embedding_size
        self.hom_set = nn.ModuleList()
        
        for i in range(hom_set_size):
            morphism = nn.Sequential()

            for j in range(depth-1):
                morphism.append(MorphismBlock(embedding_size, dropout))

            morphism.append(nn.Linear(embedding_size, embedding_size))
            #morphism.append(norm_layer)
            
            morphism.append(FINAL_ACT)
            self.hom_set.append(morphism)

        self.hom_set.append(nn.Identity())
        
    def forward(self, antecedent, consequent):

        best_loss = float("inf")
        losses = list()
        
        for i, morphism in enumerate(self.hom_set):
            
            if i == len(self.hom_set) - 1:                
                estim_cons = morphism(antecedent)
            else:
                residual = morphism(antecedent)
                estim_cons = antecedent + residual
            
            loss = norm(estim_cons, consequent)

            losses.append(loss)
            #            mean_loss = th.mean(loss)
            
            #            if mean_loss < best_loss:
            #                chosen_loss = loss
            #                best_loss = mean_loss
            
            #        losses = losses.transpose(0,1)

        losses = th.vstack(losses)
        losses = losses.transpose(0,1)
        
                                         
            
        losses = th.min(losses, dim=1)
        losses_values = losses.values
        losses_indices = losses.indices
        return losses_values



class Existential(nn.Module):
    
    def __init__(self, embedding_size, prod_net, norm_layer_obj, norm_layer_rel, dropout = 0):
        super().__init__()

        self.prod_net = prod_net

        self.bn1 = nn.LayerNorm(2*embedding_size)
        self.bn2 = nn.LayerNorm(embedding_size)
        self.bn3 = nn.LayerNorm(embedding_size)
        
        self.slicing_filler = nn.Sequential(
            nn.Linear(2*embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            HID_ACT,
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),

            #norm_layer_obj,

            FINAL_ACT
        )

        self.slicing_relation = nn.Sequential(
            nn.Linear(3*embedding_size, 2*embedding_size),
            nn.LayerNorm(2*embedding_size),
            HID_ACT,
            nn.Dropout(dropout),
            
            nn.Linear(2*embedding_size, embedding_size),

            #norm_layer_rel,


            FINAL_ACT
            
        )
        
    def forward(self, relation, filler, *outers):
        outer = sum(outers)/len(outers)
        
        x = th.cat([outer, filler, relation], dim =1)
        sliced_relation = self.slicing_relation(x)
        x = th.cat([outer, filler], dim =1)
        sliced_filler = self.slicing_filler(x)

        prod, prod_loss = self.prod_net(sliced_relation, sliced_filler)
        return prod, prod_loss

        
