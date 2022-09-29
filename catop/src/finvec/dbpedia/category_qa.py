"""This module contains the Category class customized for query-answering."""
import torch as th
import torch.nn as nn
from src.finvec.category import CodomainFunctor, DomainFunctor, Existential, ExistentialSlicer, Negation, RelationExistential, Universal
import src.finvec.dbpedia.losses_qa as L
from src.morphism import TransformationMorphism


class Category(nn.Module):
    """This class represents a category and its opposite."""

    def __init__(self,
                 embedding_size,
                 individuals_dict,
                 classes_dict,
                 relations_dict,
                 individual_to_concept_dict,
                 bot_idx=None,
                 num_negatives=1):
        super().__init__()

        self.num_negatives = num_negatives
        self.bot_idx = bot_idx

        n_individuals = len(individuals_dict)
        n_classes = len(classes_dict) + 2  # bot, top
        n_obj_props = len(relations_dict)
        
        self.individual_embeddings = nn.Embedding(n_individuals, embedding_size)
        th.nn.init.xavier_uniform_(self.individual_embeddings.weight.data)

        self.concept_embeddings = nn.Embedding(n_classes, embedding_size)
        th.nn.init.xavier_uniform_(self.concept_embeddings.weight.data)

        self.obj_prop_embeddings = nn.Embedding(n_obj_props, embedding_size)
        th.nn.init.xavier_uniform_(self.obj_prop_embeddings.weight.data)

        self.individual_to_concept_dict = individual_to_concept_dict

        # TODO: remove the ones that are not used
        # morphisms are linear transformations that take a vector and return a vector. They are
        # used to emulate category morphisms as entailments
        
        self.morphisms = nn.ModuleDict({
            "projection": TransformationMorphism(embedding_size),
            "injection": TransformationMorphism(embedding_size),
            "instantiation": TransformationMorphism(embedding_size),
            "projection_op": TransformationMorphism(embedding_size),
            "injection_op": TransformationMorphism(embedding_size),
            "instantiation_op": TransformationMorphism(embedding_size),
            "unslicing": TransformationMorphism(embedding_size),
            "unslicing_op": TransformationMorphism(embedding_size),
            "domain": TransformationMorphism(embedding_size),
            "domain_op": TransformationMorphism(embedding_size),
            "codomain": TransformationMorphism(embedding_size),
            "codomain_op": TransformationMorphism(embedding_size),
        })

        # TODO: remove the ones that are not used
        # nets are neural networks that take one or more vectors and return a vector. They are
        # used to generate objects that do not exist in the category such as negations, products,
        # coproducts, etc.

        self.nets = nn.ModuleDict({
            "codomain_functor": CodomainFunctor(embedding_size),
            "coproduct_generator": Coproduct(), 
           "domain_functor": DomainFunctor(embedding_size),
            "existential_generator": Existential(embedding_size),
            "existential_slicer": ExistentialSlicer(embedding_size),
            "instance_functor": InstanceFunctor(embedding_size),
            "negation_generator": Negation(embedding_size),
            "product_generator": Product(), 
            "relation_existential_generator": RelationExistential(embedding_size),
            "universal_generator": Universal(embedding_size),




        })

    def get_embeddings(self):
        """Return the embeddings."""
        return self.individual_embeddings.weight.data,\
            self.concept_embeddings.weight.data, self.obj_prop_embeddings.weight.data

    #def subsumption_loss(self, data):
    #    cat_loss = L.subsumption_loss(data, self.morphisms, self.concept_embeddings, nets=self.nets,
    #                                  bot_idx=self.bot_idx)
    #    return cat_loss

    def q_1p_loss(self, data):
        """Compute the loss for 1P queries."""
        cat_loss = L.q_1p_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def q_2p_loss(self, data):
        """Compute the loss for 2P queries."""
        cat_loss = L.q_2p_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def q_3p_loss(self, data):
        """Compute the loss for 3P queries."""
        cat_loss = L.q_3p_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def q_2i_loss(self, data):
        """Compute the loss for 2I queries."""
        cat_loss = L.q_2i_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def q_3i_loss(self, data):
        """Compute the loss for 3I queries."""
        cat_loss = L.q_3i_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def q_2in_loss(self, data):
        """Compute the loss for 2IN queries."""
        cat_loss = L.q_2in_loss(data, self.morphisms, self.individual_embeddings,
                                self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                                nets=self.nets,
                                bot_idx=self.bot_idx)
        return cat_loss

    def q_3in_loss(self, data):
        """Compute the loss for 3IN queries."""
        cat_loss = L.q_3in_loss(data, self.morphisms, self.individual_embeddings,
                                self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                                nets=self.nets,
                                bot_idx=self.bot_idx)
        return cat_loss

    def q_2u_loss(self, data):
        """Compute the loss for 2U queries."""
        cat_loss = L.q_2u_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def q_pi_loss(self, data):
        """Compute the loss for PI queries."""
        cat_loss = L.q_pi_loss(data, self.morphisms, self.individual_embeddings,
                               self.concept_embeddings, self.obj_prop_embeddings, self.individual_to_concept_dict,
                               nets=self.nets,
                               bot_idx=self.bot_idx)
        return cat_loss

    def get_loss_fn(self, query_type):
        """Return the loss function for a given GCI name."""

        loss_fn = {
            "subsumption": self.subsumption_loss,
            "1p": self.q_1p_loss,
            "2p": self.q_2p_loss,
            "3p": self.q_3p_loss,
            "2i": self.q_2i_loss,
            "3i": self.q_3i_loss,
            "2in": self.q_2in_loss,
            "3in": self.q_3in_loss,
            "2u": self.q_2u_loss,
        }[query_type]

        return loss_fn[query_type]

    def compute_loss(self, logits):
        pos = logits[:, 0].unsqueeze(dim=-1)
        neg = logits[:, 1:]
        return - th.nn.functional.logsigmoid(pos - neg).mean()

    def query_entity_loss(self, x):
        # 1p e
        x_1p = th.index_select(x, 0, (x[:, 0, 0] == 11).nonzero().squeeze(-1))
        if len(x_1p):
            anchor_1p = x_1p[:, :, -3]
            relation_1p = x_1p[:, :, -2]
            answer_1p = x_1p[:, :, -1]

            # Positive samples
            anchor_1p_pos = anchor_1p[:, 0]
            relation_1p_pos = relation_1p[:, 0]
            answer_1p_pos = answer_1p[:, 0]

            pos_logits = self.q_1p_loss(th.stack([anchor_1p_pos, relation_1p_pos,
                                                  answer_1p_pos], dim=-1))
            pos_logits = pos_logits.unsqueeze(dim=-1)
            # Negative samples
            anchor_1p_neg = anchor_1p[:, 1:]
            relation_1p_neg = relation_1p[:, 1:]
            answer_1p_neg = answer_1p[:, 1:]

            neg_data = th.stack([anchor_1p_neg, relation_1p_neg, answer_1p_neg], dim=-1)
            neg_data = neg_data.reshape(-1, 3)

            neg_logits = self.q_1p_loss(neg_data).reshape(-1, self.num_negatives)

            loss_1p = - th.nn.functional.logsigmoid(pos_logits - neg_logits).mean()
        else:
            loss_1p = 0

        # 2p e
        x_2p = th.index_select(x, 0, (x[:, 0, 0] == 21).nonzero().squeeze(-1))
        if len(x_2p):
            anchor_2p = x_2p[:, :, -4]
            relation_2p_1 = x_2p[:, :, -3]
            relation_2p_2 = x_2p[:, :, -2]
            answer_2p = x_2p[:, :, -1]

            # Positive samples
            anchor_2p_pos = anchor_2p[:, 0]
            relation_2p_1_pos = relation_2p_1[:, 0]
            relation_2p_2_pos = relation_2p_2[:, 0]
            answer_2p_pos = answer_2p[:, 0]

            pos_data = th.stack([anchor_2p_pos, relation_2p_1_pos, relation_2p_2_pos,
                                 answer_2p_pos], dim=-1)
            pos_logits = self.q_2p_loss(pos_data).unsqueeze(dim=-1)

            # Negative samples
            anchor_2p_neg = anchor_2p[:, 1:]
            relation_2p_1_neg = relation_2p_1[:, 1:]
            relation_2p_2_neg = relation_2p_2[:, 1:]
            answer_2p_neg = answer_2p[:, 1:]

            neg_data = th.stack([anchor_2p_neg, relation_2p_1_neg, relation_2p_2_neg,
                                 answer_2p_neg], dim=-1)

            neg_data = neg_data.reshape(-1, 4)

            neg_logits = self.q_2p_loss(neg_data).reshape(-1, self.num_negatives)

            loss_2p = - th.nn.functional.logsigmoid(pos_logits - neg_logits).mean()
        else:
            loss_2p = 0

        # 3p e
        x_3p = th.index_select(x, 0, (x[:, 0, 0] == 31).nonzero().squeeze(-1))
        if len(x_3p):
            anchor_3p = x_3p[:, :, -5]
            relation_3p_1 = x_3p[:, :, -4]
            relation_3p_2 = x_3p[:, :, -3]
            relation_3p_3 = x_3p[:, :, -2]
            answer_3p = x_3p[:, :, -1]

            # Positive samples
            anchor_3p_pos = anchor_3p[:, 0]
            relation_3p_1_pos = relation_3p_1[:, 0]
            relation_3p_2_pos = relation_3p_2[:, 0]
            relation_3p_3_pos = relation_3p_3[:, 0]
            answer_3p_pos = answer_3p[:, 0]

            pos_data = th.stack([anchor_3p_pos, relation_3p_1_pos, relation_3p_2_pos,
                                 relation_3p_3_pos, answer_3p_pos], dim=-1)
            pos_logits = self.q_3p_loss(pos_data).unsqueeze(dim=-1)

            # Negative samples
            anchor_3p_neg = anchor_3p[:, 1:]
            relation_3p_1_neg = relation_3p_1[:, 1:]
            relation_3p_2_neg = relation_3p_2[:, 1:]
            relation_3p_3_neg = relation_3p_3[:, 1:]
            answer_3p_neg = answer_3p[:, 1:]

            neg_data = th.stack([anchor_3p_neg, relation_3p_1_neg, relation_3p_2_neg,
                                 relation_3p_3_neg, answer_3p_neg], dim=-1)
            neg_data = neg_data.reshape(-1, 5)

            neg_logits = self.q_3p_loss(neg_data).reshape(-1, self.num_negatives)

            loss_3p = - th.nn.functional.logsigmoid(pos_logits - neg_logits).mean()
        else:
            loss_3p = 0

        # 2i e
        x_2i = th.index_select(x, 0, (x[:, 0, 0] == 41).nonzero().squeeze(-1))
        if len(x_2i):
            anchor_2i_1 = x_2i[:, :, -5]
            relation_2i_1 = x_2i[:, :, -4]
            anchor_2i_2 = x_2i[:, :, -3]
            relation_2i_2 = x_2i[:, :, -2]
            answer_2i = x_2i[:, :, -1]

            # Positive samples
            anchor_2i_1_pos = anchor_2i_1[:, 0]
            relation_2i_1_pos = relation_2i_1[:, 0]
            anchor_2i_2_pos = anchor_2i_2[:, 0]
            relation_2i_2_pos = relation_2i_2[:, 0]
            answer_2i_pos = answer_2i[:, 0]

            pos_logits = self.q_2i_loss(th.stack([anchor_2i_1_pos, relation_2i_1_pos,
                                                  anchor_2i_2_pos, relation_2i_2_pos,
                                                  answer_2i_pos], dim=-1)).unsqueeze(dim=-1)

            # Negative samples
            anchor_2i_1_neg = anchor_2i_1[:, 1:]
            relation_2i_1_neg = relation_2i_1[:, 1:]
            anchor_2i_2_neg = anchor_2i_2[:, 1:]
            relation_2i_2_neg = relation_2i_2[:, 1:]
            answer_2i_neg = answer_2i[:, 1:]

            neg_data = th.stack([anchor_2i_1_neg, relation_2i_1_neg, anchor_2i_2_neg,
                                 relation_2i_2_neg, answer_2i_neg], dim=-1)
            neg_data = neg_data.reshape(-1, 5)

            neg_logits = self.q_2i_loss(neg_data).reshape(-1, self.num_negatives)

            loss_2i = - th.nn.functional.logsigmoid(pos_logits - neg_logits).mean()

        else:
            loss_2i = 0

        # 3i e
        x_3i = th.index_select(x, 0, (x[:, 0, 0] == 51).nonzero().squeeze(-1))
        if len(x_3i):
            anchor_3i_1 = x_3i[:, :, -7]
            relation_3i_1 = x_3i[:, :, -6]
            anchor_3i_2 = x_3i[:, :, -5]
            relation_3i_2 = x_3i[:, :, -4]
            anchor_3i_3 = x_3i[:, :, -3]
            relation_3i_3 = x_3i[:, :, -2]
            answer_3i = x_3i[:, :, -1]

            # Positive samples
            anchor_3i_1_pos = anchor_3i_1[:, 0]
            relation_3i_1_pos = relation_3i_1[:, 0]
            anchor_3i_2_pos = anchor_3i_2[:, 0]
            relation_3i_2_pos = relation_3i_2[:, 0]
            anchor_3i_3_pos = anchor_3i_3[:, 0]
            relation_3i_3_pos = relation_3i_3[:, 0]
            answer_3i_pos = answer_3i[:, 0]

            pos_logits = self.q_3i_loss(th.stack([anchor_3i_1_pos, relation_3i_1_pos,
                                                  anchor_3i_2_pos, relation_3i_2_pos,
                                                  anchor_3i_3_pos, relation_3i_3_pos,
                                                  answer_3i_pos], dim=-1)).unsqueeze(dim=-1)

            # Negative samples
            anchor_3i_1_neg = anchor_3i_1[:, 1:]
            relation_3i_1_neg = relation_3i_1[:, 1:]
            anchor_3i_2_neg = anchor_3i_2[:, 1:]
            relation_3i_2_neg = relation_3i_2[:, 1:]
            anchor_3i_3_neg = anchor_3i_3[:, 1:]
            relation_3i_3_neg = relation_3i_3[:, 1:]
            answer_3i_neg = answer_3i[:, 1:]

            neg_data = th.stack([anchor_3i_1_neg, relation_3i_1_neg, anchor_3i_2_neg,
                                 relation_3i_2_neg, anchor_3i_3_neg, relation_3i_3_neg,
                                 answer_3i_neg], dim=-1)
            neg_data = neg_data.reshape(-1, 7)

            neg_logits = self.q_3i_loss(neg_data).reshape(-1, self.num_negatives)

            loss_3i = - th.nn.functional.logsigmoid(pos_logits - neg_logits).mean()
        else:
            loss_3i = 0

        return [loss_1p, loss_2p, loss_3p, loss_2i, loss_3i]

    def query_concept_loss(self, x):
        # 1p c
        x_1p = th.index_select(x, 0, (x[:, 0, 0] == 12).nonzero().squeeze(-1))
        if len(x_1p):
            anchor_1p = x_1p[:, :, -3]
            relation_1p = x_1p[:, :, -2]
            answer_1p = self.c_embedding(x_1p[:, :, -1])
            logits_1p = ((anchor_1p + relation_1p) * answer_1p).sum(dim=-1)
            loss_1p = self.compute_loss(logits_1p)
        else:
            loss_1p = 0

        # 2p c
        x_2p = th.index_select(x, 0, (x[:, 0, 0] == 22).nonzero().squeeze(-1))
        if len(x_2p):
            anchor_2p = x_2p[:, :, -4]
            relation_2p_1 = x_2p[:, :, -3]
            relation_2p_2 = x_2p[:, :, -2]
            answer_2p = self.c_embedding(x_2p[:, :, -1])
            logits_2p = ((anchor_2p + relation_2p_1 + relation_2p_2) * answer_2p).sum(dim=-1)
            loss_2p = self.compute_loss(logits_2p)
        else:
            loss_2p = 0

        # 3p c
        x_3p = th.index_select(x, 0, (x[:, 0, 0] == 32).nonzero().squeeze(-1))
        if len(x_3p):
            anchor_3p = x_3p[:, :, -5]
            relation_3p_1 = x_3p[:, :, -4]
            relation_3p_2 = x_3p[:, :, -3]
            relation_3p_3 = x_3p[:, :, -2]
            answer_3p = self.c_embedding(x_3p[:, :, -1])
            logits_3p = ((anchor_3p + relation_3p_1 + relation_3p_2 + relation_3p_3) * answer_3p).sum(dim=-1)
            loss_3p = self.compute_loss(logits_3p)
        else:
            loss_3p = 0

        # 2i c
        x_2i = th.index_select(x, 0, (x[:, 0, 0] == 42).nonzero().squeeze(-1))
        if len(x_2i):
            anchor_2i_1 = x_2i[:, :, -5]
            relation_2i_1 = x_2i[:, :, -4]
            anchor_2i_2 = x_2i[:, :, -3]
            relation_2i_2 = x_2i[:, :, -2]
            answer_2i = self.c_embedding(x_2i[:, :, -1])

            fs_2i_1 = th.sigmoid(th.matmul((anchor_2i_1 + relation_2i_1), self.individual_embeddings.weight.data.t()))
            fs_2i_2 = th.sigmoid(th.matmul((anchor_2i_2 + relation_2i_2), self.individual_embeddings.weight.data.t()))
            fs_2i_q = fs_2i_1 * fs_2i_2
            fs_2i_c = th.sigmoid(th.matmul(answer_2i, self.individual_embeddings.weight.data.t()))
            logits_2i = - self.js_div(fs_2i_q, fs_2i_c)
            loss_2i = self.compute_loss(logits_2i)
        else:
            loss_2i = 0

        # 3i c
        x_3i = th.index_select(x, 0, (x[:, 0, 0] == 52).nonzero().squeeze(-1))
        if len(x_3i):
            anchor_3i_1 = x_3i[:, :, -7]
            relation_3i_1 = x_3i[:, :, -6]
            anchor_3i_2 = x_3i[:, :, -5]
            relation_3i_2 = x_3i[:, :, -4]
            anchor_3i_3 = x_3i[:, :, -3]
            relation_3i_3 = x_3i[:, :, -2]
            answer_3i = self.c_embedding(x_3i[:, :, -1])

            fs_3i_1 = th.sigmoid(th.matmul((anchor_3i_1 + relation_3i_1), self.individual_embeddings.weight.data.t()))
            fs_3i_2 = th.sigmoid(th.matmul((anchor_3i_2 + relation_3i_2), self.individual_embeddings.weight.data.t()))
            fs_3i_3 = th.sigmoid(th.matmul((anchor_3i_3 + relation_3i_3), self.individual_embeddings.weight.data.t()))
            fs_3i_q = fs_3i_1 * fs_3i_2 * fs_3i_3
            fs_3i_c = th.sigmoid(th.matmul(answer_3i, self.individual_embeddings.weight.data.t()))
            logits_3i = - self.js_div(fs_3i_q, fs_3i_c)
            loss_3i = self.compute_loss(logits_3i)
        else:
            loss_3i = 0

        return [loss_1p, loss_2p, loss_3p, loss_2i, loss_3i]

    def query_entity_logit(self, x, query_type):
        # 1p e
        if query_type == '1p':
            x_1p = th.index_select(x, 0, (x[:, 0, 0] == 11).nonzero().squeeze(-1))
            anchor_1p = x_1p[:, :, -3]
            relation_1p = x_1p[:, :, -2]
            answer_1p = x_1p[:, :, -1]

            data = th.stack([anchor_1p, relation_1p, answer_1p], dim=-1).squeeze()
            logits = self.q_1p_loss(data)
        
        # 2p e
        elif query_type == '2p':
            x_2p = th.index_select(x, 0, (x[:, 0, 0] == 21).nonzero().squeeze(-1))
            anchor_2p = x_2p[:, :, -4]
            relation_2p_1 = x_2p[:, :, -3]
            relation_2p_2 = x_2p[:, :, -2]
            answer_2p = x_2p[:, :, -1]

            data = th.stack([anchor_2p, relation_2p_1, relation_2p_2, answer_2p], dim=-1).squeeze()
            logits = self.q_2p_loss(data)

        # 3p e
        elif query_type == '3p':
            x_3p = th.index_select(x, 0, (x[:, 0, 0] == 31).nonzero().squeeze(-1))
            anchor_3p = x_3p[:, :, -5]
            relation_3p_1 = x_3p[:, :, -4]
            relation_3p_2 = x_3p[:, :, -3]
            relation_3p_3 = x_3p[:, :, -2]
            answer_3p = x_3p[:, :, -1]

            data = th.stack([anchor_3p, relation_3p_1, relation_3p_2, 
                             relation_3p_3, answer_3p], dim=-1).squeeze()
            logits = self.q_3p_loss(data)

        # 2i e
        elif query_type == '2i':
            x_2i = th.index_select(x, 0, (x[:, 0, 0] == 41).nonzero().squeeze(-1))
            anchor_2i_1 = x_2i[:, :, -5]
            relation_2i_1 = x_2i[:, :, -4]
            anchor_2i_2 = x_2i[:, :, -3]
            relation_2i_2 = x_2i[:, :, -2]
            answer_2i = x_2i[:, :, -1]

            data = th.stack([anchor_2i_1, relation_2i_1, anchor_2i_2, 
                             relation_2i_2, answer_2i], dim=-1).squeeze()

            logits = self.q_2i_loss(data)
            
        # 3i e
        elif query_type == '3i':
            x_3i = th.index_select(x, 0, (x[:, 0, 0] == 51).nonzero().squeeze(-1))
            anchor_3i_1 = x_3i[:, :, -7]
            relation_3i_1 = x_3i[:, :, -6]
            anchor_3i_2 = x_3i[:, :, -5]
            relation_3i_2 = x_3i[:, :, -4]
            anchor_3i_3 = x_3i[:, :, -3]
            relation_3i_3 = x_3i[:, :, -2]
            answer_3i = x_3i[:, :, -1]

            data = th.stack([anchor_3i_1, relation_3i_1, anchor_3i_2,
                             relation_3i_2, anchor_3i_3, relation_3i_3, 
                             answer_3i], dim=-1).squeeze()
    
            logits = self.q_3i_loss(data)
        
        # pi e
        elif query_type == 'pi':
            x_pi = th.index_select(x, 0, (x[:, 0, 0] == 61).nonzero().squeeze(-1))
            anchor_pi_1 = x_pi[:, :, -6]
            relation_pi_11 = x_pi[:, :, -5]
            relation_pi_12 = x_pi[:, :, -4]
            anchor_pi_2 = x_pi[:, :, -3]
            relation_pi_2 = x_pi[:, :, -2]
            answer_pi = x_pi[:, :, -1]

            data = th.stack([anchor_pi_1, relation_pi_11, relation_pi_12, 
                             anchor_pi_2, relation_pi_2, answer_pi], dim=-1).squeeze()

            logits = self.q_pi_loss(data)
        
        # ip e
        elif query_type == 'ip':
            x_ip = th.index_select(x, 0, (x[:, 0, 0] == 71).nonzero().squeeze(-1))
            anchor_ip_1 = x_ip[:, :, -6]
            relation_ip_1 = x_ip[:, :, -5]
            anchor_ip_2 = x_ip[:, :, -4]
            relation_ip_2 = x_ip[:, :, -3]
            relation_ip_3 = x_ip[:, :, -2]
            answer_ip = x_ip[:, :, -1]

            query_ip_1 = (anchor_ip_1 + relation_ip_1 + relation_ip_3).unsqueeze(dim=1)
            query_ip_2 = (anchor_ip_2 + relation_ip_2 + relation_ip_3).unsqueeze(dim=1)
            query_ip = th.cat([query_ip_1, query_ip_2], dim=1)

            mid_ip = th.nn.functional.relu(self.fc_1(query_ip))
            attention = th.nn.functional.softmax(self.fc_2(mid_ip), dim=1)
            query_emb_ip = th.sum(attention * query_ip, dim=1)
            logits = - th.norm(query_emb_ip.unsqueeze(dim=1) - answer_ip, p=1, dim=-1)
        
        # 2u e
        elif query_type == '2u':
            x_2u = th.index_select(x, 0, (x[:, 0, 0] == 81).nonzero().squeeze(-1))
            anchor_2u_1 = x_2u[:, :, -5]
            relation_2u_1 = x_2u[:, :, -4]
            anchor_2u_2 = x_2u[:, :, -3]
            relation_2u_2 = x_2u[:, :, -2]
            answer_2u = x_2u[:, :, -1]

            logits_2u_1 = - th.norm(anchor_2u_1 + relation_2u_1 - answer_2u, p=1, dim=-1).unsqueeze(dim=-1)
            logits_2u_2 = - th.norm(anchor_2u_2 + relation_2u_2 - answer_2u, p=1, dim=-1).unsqueeze(dim=-1)
            logits = th.cat([logits_2u_1, logits_2u_2], dim=-1).max(dim=-1)[0]

        # up e
        elif query_type == 'up':
            x_up = th.index_select(x, 0, (x[:, 0, 0] == 91).nonzero().squeeze(-1))
            anchor_up_1 = x_up[:, :, -6]
            relation_up_1 = x_up[:, :, -5]
            anchor_up_2 = x_up[:, :, -4]
            relation_up_2 = x_up[:, :, -3]
            relation_up_3 = x_up[:, :, -2]
            answer_up = x_up[:, :, -1]

            logits_up_1 = - th.norm(anchor_up_1 + relation_up_1 + relation_up_3 - answer_up, p=1, dim=-1).unsqueeze(dim=-1)
            logits_up_2 = - th.norm(anchor_up_2 + relation_up_2 + relation_up_3 - answer_up, p=1, dim=-1).unsqueeze(dim=-1)
            logits = th.cat([logits_up_1, logits_up_2], dim=-1).max(dim=-1)[0]
        
        return logits

    def predict(self, x, query_type, answer_type):
        if answer_type == 'e':
            logits = self.query_entity_logit(x, query_type)
        elif answer_type == 'c':
            logits = self.query_concept_logit(x, query_type)
        return logits

    def forward(self, data):
        """Forward pass."""

        qe_losses = self.query_entity_loss(data)

        return qe_losses, None, None, None


class Product(nn.Module):
    """This class represents the product generator.
    The product is generated by averaging the inverse projections of the projected classes
    in the opposite category.
    """
    def __init__(self):
        super().__init__()


    def forward(self, morphisms, *operands):
        """Forward pass."""
        morphism = morphisms["projection_op"]
        product = 0
        for op in operands:
            product += morphism(op)
        return product / len(operands)


class Coproduct(nn.Module):
    """This class represents the coproduct generator.
    The coproduct is generated by averaging the injections of the two classes in the original
    category.
    """
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(th.zeros(1))

    def forward(self, morphisms, *operands):
        """Forward pass."""
        morphism = morphisms["injection"]
        coproduct = 0
        for op in operands:
            coproduct += morphism(op)
        return coproduct / len(operands)


class Existential(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(2 * embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a, b):
        return self.fc(th.cat((a, b), dim=1))


class RelationExistential(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(2 * embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a, b):
        return self.fc(th.cat((a, b), dim=1))


class Universal(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(2 * embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a, b):
        return self.fc(th.cat((a, b), dim=1))


class Negation(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a):
        return self.fc(a)


class ExistentialSlicer(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(2 * embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a, b):
        return self.fc(th.cat((a, b), dim=1))


class CodomainFunctor(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a):
        return self.fc(a)


class DomainFunctor(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, a):
        return self.fc(a)

class InstanceFunctor(nn.Module):
    def __init__(self, embedding_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.fc = nn.Linear(2*embedding_size, embedding_size)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, individual, concept):
        return self.fc(th.cat((individual, concept), dim=1))