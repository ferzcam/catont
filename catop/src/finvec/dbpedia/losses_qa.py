from genericpath import exists
import torch.nn as nn
import torch as th

# mse_loss = nn.MSELoss(reduction="none")

def cosine_similarity(a, b):
    """Compute the cosine similarity between two tensors
    """
    a_norm = th.norm(a, p=2, dim=1)
    b_norm = th.norm(b, p=2, dim=1)
    dot_product = th.sum(a * b, dim=1, keepdim=True)
    return th.sigmoid(dot_product)

def compute_product_loss(operands, product, morphisms):
    """Compute the loss for the product of two objects
        B and C
    """

    # Compute projection loss for all the operands
    projection_loss = 0
    projection_loss_op = 0
    for operand in operands:
        projected_operand = morphisms["projection"](product)  # B and C --> B
        # projection_loss += th.mean(mse_loss(projected_operand, operand), dim=1)
        projection_loss += th.mean(cosine_similarity(projected_operand, operand), dim=1)
        
        projected_operand_op = morphisms["projection_op"](operand)  # B and C --> C
        projection_loss_op += th.mean(cosine_similarity(projected_operand_op, product), dim=1)

    # TODO: Compute coproduct loss

    loss = projection_loss + projection_loss_op
    return loss


def compute_coproduct_loss(operands, coproduct, morphisms):
    """Compute the loss for the coproduct of two objects
        B or C
    """
    # Compute injection loss for all the operands

    injection_loss = 0
    injection_loss_op = 0
    for operand in operands:
        injected_coproduct = morphisms["injection"](operand)  # B --> B or C
        injection_loss += th.mean(cosine_similarity(injected_coproduct, coproduct), dim=1)

        injected_coproduct_op = morphisms["injection_op"](coproduct)  # B or C --> C
        injection_loss_op += th.mean(cosine_similarity(injected_coproduct_op, operand), dim=1)

    loss = injection_loss + injection_loss_op
    return loss


def compute_class_instantiation_loss(individual, concept, morphisms, nets=None):
    """Compute the loss for the instantiation of a class
    """

    instantiation_morphism = morphisms["instantiation_morphism"]
    concept_instance = instantiation_morphism(individual)
    loss = th.mean(cosine_similarity(concept_instance, concept), dim=1)
    return loss


def compute_relation_instantiation_loss(domain_individual, codomain_individual, relation,
                                        morphisms, nets=None):
    """Compute the loss for the instantiation of a relation
    """

    relation_instantiation_morphism = morphisms["relation_instantiation_morphism"]
    relation_instance = relation_instantiation_morphism(domain_individual, codomain_individual)
    loss = th.mean(cosine_similarity(relation_instance, relation), dim=1)
    return loss


def compute_negation_loss(a, not_a, bot, morphisms, nets=None):
    """Compute the loss for the negation of an object
    """
    negation_generator = nets['negation_generator']
    not_a = negation_generator(a)

    product_generator = nets['product_generator']
    a_and_not_a = product_generator(morphisms, *[a, not_a])

    # a and not a --> bot
    assert a_and_not_a.shape == bot.shape, f"Shapes of a_and_not_a and bot_tensor are not the \
same: {a_and_not_a.shape} != {bot.shape}"

    empty_intersection = morphisms["projection"](a_and_not_a)
    negation_loss = th.mean(cosine_similarity(empty_intersection, bot), dim=1)

    return negation_loss


def subsumption_loss(tensor_data, morphisms, class_embeddings, nets=None, bot_idx=None):
    """Compute the loss for subsumption axioms. A subClassOf B
    Process it as: not A or B
    """
    # Unpack data
    a_idx, b_idx = tensor_data[:, 0], tensor_data[:, 1]
    a = class_embeddings(a_idx)
    not_a = nets["negation_generator"](a)
    b = class_embeddings(b_idx)
    not_a_or_b = nets["coproduct_generator"](morphisms, *[not_a, b])

    bot_idx = th.tensor([bot_idx for _ in range(a.shape[0])], dtype=th.long, device=a.device)
    bot = class_embeddings(bot_idx)

    negation_loss = compute_negation_loss(a, not_a, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_a, b], not_a_or_b, morphisms)

    loss = negation_loss + coproduct_loss
    return loss


# TODO: Check if needed to add existential losses

def q_1p_loss(data, morphisms, individual_embeddings, concept_embeddings, relation_embeedings, individual_to_concept,
              nets=None, bot_idx=None):
    """Compute the loss for 1p queries

    'ind' is an individual
    """
    ind1_idx, rel_idx, ind2_idx = data[:, 0], data[:, 1], data[:, 2]
    ind1 = individual_embeddings(ind1_idx)
    rel = relation_embeedings(rel_idx)
    ind2 = individual_embeddings(ind2_idx)

    concept_ind1 = concept_embeddings(individual_to_concept[ind1_idx])
    concept_ind2 = concept_embeddings(individual_to_concept[ind2_idx])

    c1 = nets["instance_functor"](ind1, concept_ind1)
    c2 = nets["instance_functor"](ind2, concept_ind2)

    abox_loss = 0

    bot_idx = concept_embeddings.weight.data.shape[0] - 1
    bot_idx = th.tensor([bot_idx for _ in range(c1.shape[0])], dtype=th.long, device=c1.device)
    bot = concept_embeddings(bot_idx)

    not_c1 = nets["negation_generator"](c1)
    existential_generator = nets['existential_generator']
    exists_r_c2 = existential_generator(rel, c2)

    coproduct_generator = nets['coproduct_generator']

    not_c1_or_exists_r_c2 = coproduct_generator(morphisms, *[not_c1, exists_r_c2])

    negation_loss = compute_negation_loss(c1, not_c1, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_c1, exists_r_c2], not_c1_or_exists_r_c2,
                                            morphisms)

    tbox_loss = negation_loss + coproduct_loss
    return abox_loss + tbox_loss


def q_2p_loss(data, morphisms, individual_embeddings, concept_embeddings, relation_embeedings, individual_to_concept,
              nets=None, bot_idx=None):

    """Compute the loss for 2p queries"""

    ind1_idx, rel1_idx, rel2_idx, ind2_idx = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    ind1 = individual_embeddings(ind1_idx)
    rel1 = relation_embeedings(rel1_idx)
    rel2 = relation_embeedings(rel2_idx)
    ind2 = individual_embeddings(ind2_idx)

    concept_ind1 = concept_embeddings(individual_to_concept[ind1_idx])
    concept_ind2 = concept_embeddings(individual_to_concept[ind2_idx])

    c1 = nets["instance_functor"](ind1, concept_ind1)
    c2 = nets["instance_functor"](ind2, concept_ind2)

    abox_loss = 0
    # Class instantiation
    # ind1_class = class_embeddings(types_dict[ind1_idx])
    # ind2_class = class_embeddings(types_dict[ind2_idx])

    # c1 = nets["class_instantiation_generator"](morphisms, ind1)
    # c2 = nets["class_instantiation_generator"](morphisms, ind2)

    # abox_loss = mse_loss(c1, ind1_class)
    # abox_loss += mse_loss(c2, ind2_class)

    # TBox axiom C1 subclassof exists R1. (exists R2.C2)
    # Process it as: not C1 or exists R1. (exists R2.C2)

    bot_idx = concept_embeddings.weight.data.shape[0] - 1
    bot_idx = th.tensor([bot_idx for _ in range(c1.shape[0])], dtype=th.long, device=c1.device)
    bot = concept_embeddings(bot_idx)

    not_c1 = nets["negation_generator"](c1)
    existential_generator = nets['existential_generator']
    exists_r2_c2 = existential_generator(rel2, c2)
    exists_r1_exists_r2_c2 = existential_generator(rel1, exists_r2_c2)

    coproduct_generator = nets['coproduct_generator']
    not_c1_or_exists_r1_exists_r2_c2 = coproduct_generator(morphisms,
                                                           *[not_c1, exists_r1_exists_r2_c2])

    negation_loss = compute_negation_loss(c1, not_c1, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_c1, exists_r1_exists_r2_c2],
                                            not_c1_or_exists_r1_exists_r2_c2, morphisms)

    tbox_loss = negation_loss + coproduct_loss
    return abox_loss + tbox_loss


def q_3p_loss(data, morphisms, individual_embeddings, concept_embeddings, relation_embeedings, individual_to_concept,
              nets=None, bot_idx=None):

    """Compute the loss for 3p queries"""

    ind1_idx, rel1_idx, rel2_idx, rel3_idx, ind2_idx = data[:, 0], data[:, 1], data[:, 2], \
        data[:, 3], data[:, 4]
    ind1 = individual_embeddings(ind1_idx)
    rel1 = relation_embeedings(rel1_idx)
    rel2 = relation_embeedings(rel2_idx)
    rel3 = relation_embeedings(rel3_idx)
    ind2 = individual_embeddings(ind2_idx)

    concept_ind1 = concept_embeddings(individual_to_concept[ind1_idx])
    concept_ind2 = concept_embeddings(individual_to_concept[ind2_idx])

    c1 = nets["instance_functor"](ind1, concept_ind1)
    c2 = nets["instance_functor"](ind2, concept_ind2)
    
    abox_loss = 0
    # Class instantiation
    # ind1_class = class_embeddings(types_dict[ind1_idx])
    # ind2_class = class_embeddings(types_dict[ind2_idx])

    # c1 = nets["class_instantiation_generator"](morphisms, ind1)
    # c2 = nets["class_instantiation_generator"](morphisms, ind2)

    # abox_loss = mse_loss(c1, ind1_class)
    # abox_loss += mse_loss(c2, ind2_class)

    # TBox axiom C1 subclassof exists R1. (exists R2. (exists R3.C2))
    # Process it as: not C1 or exists R1. (exists R2. (exists R3.C2))

    bot_idx = concept_embeddings.weight.data.shape[0] - 1
    bot_idx = th.tensor([bot_idx for _ in range(c1.shape[0])], dtype=th.long, device=c1.device)
    bot = concept_embeddings(bot_idx)

    not_c1 = nets["negation_generator"](c1)
    existential_generator = nets['existential_generator']
    exists_r3_c2 = existential_generator(rel3, c2)
    exists_r2_exists_r3_c2 = existential_generator(rel2, exists_r3_c2)
    exists_r1_exists_r2_exists_r3_c2 = existential_generator(rel1, exists_r2_exists_r3_c2)

    coproduct_generator = nets['coproduct_generator']

    # long_name = not_c1_or_exists_r1_exists_r2_exists_r3_c2
    long_name = coproduct_generator(morphisms, *[not_c1, exists_r1_exists_r2_exists_r3_c2])

    negation_loss = compute_negation_loss(c1, not_c1, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_c1, exists_r1_exists_r2_exists_r3_c2],
                                            long_name, morphisms)

    tbox_loss = negation_loss + coproduct_loss
    return abox_loss + tbox_loss


def q_2i_loss(data, morphisms, individual_embeddings, concept_embeddings, relation_embeddings, individual_to_concept,
              nets=None, bot_idx=None):

    """Compute the loss for 2i queries"""

    ind1_idx, rel1_idx, ind2_idx, rel2_idx, ind3_idx = data[:, 0], data[:, 1], data[:, 2], \
        data[:, 3], data[:, 4]

    ind1 = individual_embeddings(ind1_idx)
    rel1 = relation_embeddings(rel1_idx)
    ind2 = individual_embeddings(ind2_idx)
    rel2 = relation_embeddings(rel2_idx)
    ind3 = individual_embeddings(ind3_idx)

    concept_ind1 = concept_embeddings(individual_to_concept[ind1_idx])
    concept_ind2 = concept_embeddings(individual_to_concept[ind2_idx])
    concept_ind3 = concept_embeddings(individual_to_concept[ind3_idx])

    c1 = nets["instance_functor"](ind1, concept_ind1)
    c2 = nets["instance_functor"](ind2, concept_ind2)
    c3 = nets["instance_functor"](ind3, concept_ind3)
    
    bot_idx = concept_embeddings.weight.data.shape[0] - 1
    bot_idx = th.tensor([bot_idx for _ in range(c1.shape[0])], dtype=th.long, device=c1.device)
    bot = concept_embeddings(bot_idx)

    # First part
    not_c1 = nets["negation_generator"](c1)
    existential_generator = nets['existential_generator']
    exists_r1_c3 = existential_generator(rel1, c3)
    coproduct_generator = nets['coproduct_generator']
    not_c1_or_exists_r1_c3 = coproduct_generator(morphisms, *[not_c1, exists_r1_c3])

    negation_loss = compute_negation_loss(c1, not_c1, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_c1, exists_r1_c3], not_c1_or_exists_r1_c3,
                                            morphisms)

    # Second part
    not_c2 = nets["negation_generator"](c2)
    exists_r2_c3 = existential_generator(rel2, c3)
    not_c2_or_exists_r2_c3 = coproduct_generator(morphisms, *[not_c2, exists_r2_c3])

    negation_loss += compute_negation_loss(c2, not_c2, bot, morphisms, nets=nets)
    coproduct_loss += compute_coproduct_loss([not_c2, exists_r2_c3], not_c2_or_exists_r2_c3,
                                             morphisms)

    # Intersection
    intersection_generator = nets['product_generator']
    intersection = intersection_generator(morphisms, *[not_c1_or_exists_r1_c3,
                                                       not_c2_or_exists_r2_c3])
    intersection_loss = compute_product_loss([not_c1_or_exists_r1_c3,
                                              not_c2_or_exists_r2_c3], intersection,
                                             morphisms)

    tbox_loss = negation_loss + coproduct_loss + intersection_loss
    return tbox_loss


def q_3i_loss(data, morphisms, individual_embeddings, concept_embeddings, relation_embeddings, individual_to_concept,
              nets=None, bot_idx=None):

    """Compute the loss for 3i queries"""

    ind1_idx, rel1_idx, ind2_idx, rel2_idx, ind3_idx, rel3_idx, ind4_idx = \
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]

    ind1 = individual_embeddings(ind1_idx)
    rel1 = relation_embeddings(rel1_idx)
    ind2 = individual_embeddings(ind2_idx)
    rel2 = relation_embeddings(rel2_idx)
    ind3 = individual_embeddings(ind3_idx)
    rel3 = relation_embeddings(rel3_idx)
    ind4 = individual_embeddings(ind4_idx)

    concept_ind1 = concept_embeddings(individual_to_concept[ind1_idx])
    concept_ind2 = concept_embeddings(individual_to_concept[ind2_idx])
    concept_ind3 = concept_embeddings(individual_to_concept[ind3_idx])
    concept_ind4 = concept_embeddings(individual_to_concept[ind4_idx])

    c1 = nets["instance_functor"](ind1, concept_ind1)
    c2 = nets["instance_functor"](ind2, concept_ind2)
    c3 = nets["instance_functor"](ind3, concept_ind3)
    c4 = nets["instance_functor"](ind4, concept_ind4)
    
    bot_idx = concept_embeddings.weight.data.shape[0] - 1
    bot_idx = th.tensor([bot_idx for _ in range(c1.shape[0])], dtype=th.long, device=c1.device)
    bot = concept_embeddings(bot_idx)

    # First part
    not_c1 = nets["negation_generator"](c1)
    existential_generator = nets['existential_generator']
    exists_r1_c4 = existential_generator(rel1, c4)
    coproduct_generator = nets['coproduct_generator']
    not_c1_or_exists_r1_c4 = coproduct_generator(morphisms, *[not_c1, exists_r1_c4])

    negation_loss = compute_negation_loss(c1, not_c1, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_c1, exists_r1_c4], not_c1_or_exists_r1_c4,
                                            morphisms)

    # Second part
    not_c2 = nets["negation_generator"](c2)
    exists_r2_c4 = existential_generator(rel2, c4)
    not_c2_or_exists_r2_c4 = coproduct_generator(morphisms, *[not_c2, exists_r2_c4])

    negation_loss += compute_negation_loss(c2, not_c2, bot, morphisms, nets=nets)
    coproduct_loss += compute_coproduct_loss([not_c2, exists_r2_c4], not_c2_or_exists_r2_c4,
                                             morphisms)

    # Third part
    not_c3 = nets["negation_generator"](c3)
    exists_r3_c4 = existential_generator(rel3, c4)
    not_c3_or_exists_r3_c4 = coproduct_generator(morphisms, *[not_c3, exists_r3_c4])

    negation_loss += compute_negation_loss(c3, not_c3, bot, morphisms, nets=nets)
    coproduct_loss += compute_coproduct_loss([not_c3, exists_r3_c4], not_c3_or_exists_r3_c4,
                                             morphisms)

    # Intersection
    intersection_generator = nets['product_generator']
    intersection = intersection_generator(morphisms, *[not_c1_or_exists_r1_c4,
                                                       not_c2_or_exists_r2_c4,
                                                       not_c3_or_exists_r3_c4])
    intersection_loss = compute_product_loss([not_c1_or_exists_r1_c4,
                                              not_c2_or_exists_r2_c4,
                                              not_c3_or_exists_r3_c4], intersection,
                                             morphisms)

    tbox_loss = negation_loss + coproduct_loss + intersection_loss
    return tbox_loss


def q_pi_loss(data, morphisms, individual_embeddings, concept_embeddings, relation_embeddings, individual_to_concept,
              nets=None, bot_idx=None):

    """Compute the loss for pi queries"""
    
    ind1_idx, rel11_idx, rel12_idx, ind2_idx, rel2_idx, ind3_idx = \
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

    ind1 = individual_embeddings(ind1_idx)
    rel11 = relation_embeddings(rel11_idx)
    rel12 = relation_embeddings(rel12_idx)
    ind2 = individual_embeddings(ind2_idx)
    rel2 = relation_embeddings(rel2_idx)
    ind3 = individual_embeddings(ind3_idx)

    concept_ind1 = concept_embeddings(individual_to_concept[ind1_idx])
    concept_ind2 = concept_embeddings(individual_to_concept[ind2_idx])
    concept_ind3 = concept_embeddings(individual_to_concept[ind3_idx])

    c1 = nets["instance_functor"](ind1, concept_ind1)
    c2 = nets["instance_functor"](ind2, concept_ind2)
    c3 = nets["instance_functor"](ind3, concept_ind3)

    bot_idx = concept_embeddings.weight.data.shape[0] - 1
    bot_idx = th.tensor([bot_idx for _ in range(c1.shape[0])], dtype=th.long, device=c1.device)
    bot = concept_embeddings(bot_idx)

    # First part
    not_c1 = nets["negation_generator"](c1)
    existential_generator = nets['existential_generator']
    exists_r12_c3 = existential_generator(rel12, c3)
    exists_r11_exists_r12_c3 = existential_generator(rel11, exists_r12_c3)
    coproduct_generator = nets['coproduct_generator']
    not_c1_or_exists_r11_exists_r12_c3 = coproduct_generator(morphisms, *[not_c1, exists_r11_exists_r12_c3])

    negation_loss = compute_negation_loss(c1, not_c1, bot, morphisms, nets=nets)
    coproduct_loss = compute_coproduct_loss([not_c1, exists_r11_exists_r12_c3], not_c1_or_exists_r11_exists_r12_c3,
                                            morphisms)

    # Second part
    not_c2 = nets["negation_generator"](c2)
    exists_r2_c3 = existential_generator(rel2, c3)
    not_c2_or_exists_r2_c3 = coproduct_generator(morphisms, *[not_c2, exists_r2_c3])

    negation_loss += compute_negation_loss(c2, not_c2, bot, morphisms, nets=nets)
    coproduct_loss += compute_coproduct_loss([not_c2, exists_r2_c3], not_c2_or_exists_r2_c3,
                                             morphisms)

    # Intersection
    intersection_generator = nets['product_generator']
    intersection = intersection_generator(morphisms, *[not_c1_or_exists_r11_exists_r12_c3,
                                                       not_c2_or_exists_r2_c3])
    intersection_loss = compute_product_loss([not_c1_or_exists_r11_exists_r12_c3, not_c2_or_exists_r2_c3],
                                             intersection, morphisms)

    q_pi_loss = negation_loss + coproduct_loss + intersection_loss
    return q_pi_loss

