from org.semanticweb.owlapi.model import OWLClass, OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLObjectProperty, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLObjectUnionOf,\
    OWLObjectExactCardinality, OWLDataHasValue, OWLDataSomeValuesFrom, OWLObjectMinCardinality, \
    OWLObjectHasSelf


def is_owl_class(entity):
    """Check if entity is an OWLClass"""
    return isinstance(entity, OWLClass)

def is_owl_object_intersection_of(entity):
    """Check if entity is an OWLObjectIntersectionOf"""
    return isinstance(entity, OWLObjectIntersectionOf)

def is_owl_object_union_of(entity):
    """Check if entity is an OWLObjectUnionOf"""
    return isinstance(entity, OWLObjectUnionOf)

def is_owl_object_some_values_from(entity):
    """Check if entity is an OWLObjectSomeValuesFrom"""
    return isinstance(entity, OWLObjectSomeValuesFrom)

def is_owl_object_all_values_from(entity):
    """Check if entity is an OWLObjectAllValuesFrom"""
    return isinstance(entity, OWLObjectAllValuesFrom)

def is_owl_object_complement_of(entity):
    """Check if entity is an OWLObjectComplementOf"""
    return isinstance(entity, OWLObjectComplementOf)
