import sys

import mowl
mowl.init_jvm("10g")

from mowl.datasets import PathDataset
from mowl.projection.edge import Edge
from mowl.owlapi import OWLAPIAdapter, OWLClass
from mowl.owlapi.defaults import BOT, TOP
from org.semanticweb.owlapi.model import OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, \
    OWLObjectProperty, OWLObjectAllValuesFrom, OWLObjectComplementOf, OWLObjectUnionOf,\
    OWLObjectExactCardinality, OWLDataHasValue, OWLDataSomeValuesFrom, OWLObjectMinCardinality, \
    OWLObjectHasSelf, OWLNamedIndividual, OWLClassAssertionAxiom, OWLObjectPropertyAssertionAxiom

from org.semanticweb.owlapi.model.parameters import Imports
from tqdm import tqdm

from utils import IGNORED_AXIOM_TYPES
import owl_axiom as ax
from tqdm import tqdm

class CategoricalProjector():

    def __init__(self, bidirectional_taxonomy = False):
        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.adapter = OWLAPIAdapter()
        self.ont_manager = self.adapter.owl_manager
        self.data_factory = self.adapter.data_factory
        self.intersection_virtual_nodes_count = 0
        self.union_virtual_nodes_count = 0
        self.existential_virtual_nodes_count = 0
        self.universal_virtual_nodes_count = 0
        self.negation_virtual_nodes_count = 0

        self.class_assertion_ignored = 0
        self.object_property_assertion_ignored = 0
        
    def get_intersection_virtual_node(self):
        """Return a new virtual node for an intersection concept description."""
        self.intersection_virtual_nodes_count += 1
        return "http://mowl/intersection_{}".format(self.intersection_virtual_nodes_count)

    def get_existential_virtual_node(self):
        """Return a new virtual node for an existential concept description."""
        self.existential_virtual_nodes_count += 1
        return "http://mowl/existential_{}".format(self.existential_virtual_nodes_count)

    def get_universal_virtual_node(self):
        """Return a new virtual node for an universal concept description."""
        self.universal_virtual_nodes_count += 1
        return "http://mowl/universal_{}".format(self.universal_virtual_nodes_count)

    def get_negation_virtual_node(self):
        """Return a new virtual node for a negation concept description."""
        self.negation_virtual_nodes_count += 1
        return "http://mowl/negation_{}".format(self.negation_virtual_nodes_count)

    def get_union_virtual_node(self):
        """Return a new virtual node for a union concept description."""
        self.union_virtual_nodes_count += 1
        return "http://mowl/union_{}".format(self.union_virtual_nodes_count)

    def project(self, ontology):
        """Project an ontology into a graph using categorical diagrams."""

        all_axioms = ontology.getAxioms(True)
        graph = []
        for axiom in tqdm(all_axioms, total = len(all_axioms)):
            graph += self._process_axiom(axiom)


        print("intersection_virtual_nodes_count: {}".format(self.intersection_virtual_nodes_count))
        print("union_virtual_nodes_count: {}".format(self.union_virtual_nodes_count))
        print("existential_virtual_nodes_count: {}".format(self.existential_virtual_nodes_count))
        print("universal_virtual_nodes_count: {}".format(self.universal_virtual_nodes_count))
        print("negation_virtual_nodes_count: {}".format(self.negation_virtual_nodes_count))
        print("class_assertion_ignored: {}".format(self.class_assertion_ignored))
        print("object_property_assertion_ignored: {}".format(self.object_property_assertion_ignored))
        return graph



    def _process_axiom(self, axiom):
        """Process an OWLClass and return a list of edges."""

        axiom_type = axiom.getAxiomType().getName()
        if axiom_type == "SubClassOf":
            sub_class = axiom.getSubClass()
            super_class = axiom.getSuperClass()
            return self._process_subclassof(sub_class, super_class)
        elif axiom_type == "EquivalentClasses":
            return self._process_equivalentclasses(axiom)
        elif axiom_type == "DisjointClasses":
            return self._process_disjointness(axiom)
        elif axiom_type == "ClassAssertion":
            return self._process_class_assertion(axiom)
        elif axiom_type == "ObjectPropertyAssertion":
            return self._process_object_property_assertion(axiom)
        elif axiom_type in IGNORED_AXIOM_TYPES:
            #Ignore these types of axioms
            return []
        else:
            if axiom_type == "ClassAssertion":
                self.class_assertion_ignored += 1
            elif axiom_type == "ObjectPropertyAssertion":
                self.object_property_assertion_ignored += 1
            print(f"process_axiom: Unknown axiom type: {axiom_type}")
            return []


    def _process_equivalentclasses(self, axiom):
        """Process an EquivalentClasses axiom and return a list of edges."""

        edges = []
        exprs = axiom.getClassExpressionsAsList()
        head_class = exprs[0]
        for expr in exprs:
            if isinstance(expr, OWLClass):
                head_class = expr
        if head_class is None:
            print("No OWLClass found in EquivalentClasses axiom.")
            return edges

        for expr in exprs:
            if expr == head_class:
                continue
            edges += self._process_subclassof(head_class, expr)
            edges += self._process_subclassof(expr, head_class)
        if edges == []:
            print(f"No edges found for EquivalentClasses axiom: {axiom}")
        return edges


    def _process_subclassof(self, sub_class, super_class):
        """Process a SubClassOf axiom and return a list of edges."""

        sub_edges = []
        super_edges = []

        ignored_explicit1 = ignored_explicit2 = False
        
        if not isinstance(sub_class, OWLClass):
            virtual_node1, sub_edges, ignored_explicit1 = self._process_expression_and_get_virtual_node(sub_class)
            sub_class = virtual_node1

        if not isinstance(super_class, OWLClass):
            virtual_node2, super_edges, ignored_explicit2 = self._process_expression_and_get_virtual_node(super_class)
            super_class = virtual_node2

        if ignored_explicit1 or ignored_explicit2:
            return []
        
        edges = self._subclassMorphism(sub_class, super_class)
        edges += sub_edges
        edges += super_edges

        
        if edges == []:
            print(f"No edges found for SubClassOf axiom: {sub_class} {super_class}")

        return edges

    def _process_expression_and_get_virtual_node(self, expression):

        if ax.is_owl_class(expression):
            raise ValueError("Expression is an OWLClass. It should be processed directly.")

        explicitely_ignored = False
        edges = []
        virtual_node = None
                                
        if ax.is_owl_object_intersection_of(expression):
            virtual_node = self.get_intersection_virtual_node()
            edges = self._process_intersectionof(expression, virtual_node)

        elif ax.is_owl_object_union_of(expression):
            virtual_node = self.get_union_virtual_node()
            edges += self._process_unionof(expression, virtual_node)

        elif ax.is_owl_object_some_values_from(expression):
            virtual_node = self.get_existential_virtual_node()
            edges += self._process_somevaluesfrom(expression, virtual_node)

        elif ax.is_owl_object_all_values_from(expression):
            virtual_node = self.get_universal_virtual_node()
            edges += self._process_allvaluesfrom(expression, virtual_node)

        elif ax.is_owl_object_complement_of(expression):
            virtual_node = self.get_negation_virtual_node()
            edges += self._process_negationof(expression, virtual_node)

        elif isinstance(expression, (OWLObjectExactCardinality, OWLObjectMinCardinality, OWLObjectHasSelf)):
            explicitely_ignored = True
            #Ignore cardinality restrictions for now
            pass
        else:
            print("process_subclassof: Unknown super class type: {}".format(expression))
        
        if edges == [] and not explicitely_ignored:
            print(f"No edges found for expression: {expression}")
        return virtual_node, edges, explicitely_ignored

    def _process_subclassof_complex_subclass(self, sub_class, super_class):
        """Process a SubClassOf axiom with a complex subclass and return a list of edges."""

        edges = []
        if isinstance(sub_class, OWLObjectIntersectionOf):
            virtual_node = self.get_intersection_virtual_node()

            virtual_node_class = self.adapter.create_class(virtual_node)
            edges += self._process_subclassof(virtual_node_class, super_class)
            edges += self._process_intersectionof(sub_class, virtual_node)
        elif isinstance(sub_class, OWLObjectSomeValuesFrom):
            virtual_node = self.get_existential_virtual_node()

            virtual_node_class = self.adapter.create_class(virtual_node)
            edges += self._process_subclassof(virtual_node_class, super_class)
            edges += self._process_somevaluesfrom(sub_class, virtual_node)
        else:
            print("process_subclassof_complex_subclass: Unknown sub class type: {}".format(sub_class))

        if edges == []:
            print(f"No edges found for SubClassOf axiom: {sub_class} {super_class}")
        return edges

    def _process_subclassof_inverted(self, sub_class, super_class: OWLClass):
        """Process a SubClassOf axiom and return a list of edges."""

        edges = []

        if isinstance(sub_class, OWLClass):
            edges +=[]# self._subclassMorphism(sub_class, super_class)
        elif isinstance(sub_class, OWLObjectIntersectionOf):
            virtual_node = self.get_intersection_virtual_node()
            edges += self._subclassMorphism(virtual_node, super_class)
            edges += self._process_intersectionof(sub_class, virtual_node)
        elif isinstance(sub_class, OWLObjectSomeValuesFrom):
            virtual_node = self.get_existential_virtual_node()
            edges += self._subclassMorphism(virtual_node, super_class)
            edges += self._process_somevaluesfrom(sub_class, virtual_node)
        elif isinstance(sub_class, OWLObjectUnionOf):
            virtual_node = self.get_union_virtual_node()
            edges += self._subclassMorphism(virtual_node, super_class)
            edges += self._process_unionof(sub_class, virtual_node)
        else:
            print(f"process_subclassof_inverted: Unknown sub class type: {sub_class}. Type: {type(sub_class)}")
        
        if edges == []:
            print(f"No edges found for SubClassOf axiom: {sub_class} {super_class}")
        return edges

    def _process_disjointness(self, axiom):
        """Process a disjointness axiom"""
        edges = []
        exprs = axiom.getClassExpressionsAsList()
        head_class = exprs[0]
        for expr in exprs:
            if isinstance(expr, OWLClass):
                head_class = expr
        if head_class is None:
            print("No OWLClass found in EquivalentClasses axiom.")
            return edges

        virtual_node = self.get_intersection_virtual_node()
        for expr in exprs:
            if expr == head_class:
                continue
            edges += self._process_disjoint_pairwise(head_class, expr, virtual_node)
        
        if edges == []:
            print(f"No edges found for DisjointClasses axiom: {axiom}")
        return edges

    def _process_disjoint_pairwise(self, class1, class2, virtual_node):
        """Process a disjointness axiom between two classes"""
        edges = []

        edges += self._subclassMorphism(virtual_node, BOT)

        if isinstance(class1, OWLClass):
            edges.append(self._projectionMorphism(virtual_node, class1))
        elif isinstance(class1, OWLObjectSomeValuesFrom):
            virtual_node1 = self.get_existential_virtual_node()
            edges.append(self._projectionMorphism(virtual_node, virtual_node1))
            edges += self._process_somevaluesfrom(class1, virtual_node1)
        else:
            print("process_disjoint_pairwise: Unknown class1 type: {}".format(class1))
            
            
        if isinstance(class2, OWLClass):
            edges.append(self._projectionMorphism(virtual_node, class2))
        elif isinstance(class2, OWLObjectSomeValuesFrom):
            virtual_node2 = self.get_existential_virtual_node()
            edges.append(self._projectionMorphism(virtual_node, virtual_node2))
            edges += self._process_somevaluesfrom(class2, virtual_node2)
        elif isinstance(class2, OWLObjectUnionOf):
            virtual_node2 = self.get_union_virtual_node()
            edges.append(self._projectionMorphism(virtual_node, virtual_node2))
            edges += self._process_unionof(class2, virtual_node2)
        else:
            print("process_disjoint_pairwise: Unknown class2 type: {}".format(class2))

        if edges == []:
            print(f"No edges found for DisjointClasses axiom: {class1} {class2}")

        return edges

    ######## Process OWLClassExpression ########

    def _process_intersectionof(self, intersection_of: OWLObjectIntersectionOf, virtual_node):
        """Process an OWLObjectIntersectionOf and return a list of edges."""
        explicitely_ignored = False
        edges = []
        for expr in intersection_of.getOperandsAsList():
            if isinstance(expr, OWLClass):
                edges.append(self._projectionMorphism(virtual_node, expr))
            elif isinstance(expr, OWLObjectSomeValuesFrom):
                virtual_node2 = self.get_existential_virtual_node()
                edges.append(self._projectionMorphism(virtual_node, virtual_node2))
                edges += self._process_somevaluesfrom(expr, virtual_node2)
            elif isinstance(expr, OWLObjectAllValuesFrom):
                virtual_node2 = self.get_universal_virtual_node()
                edges.append(self._projectionMorphism(virtual_node, virtual_node2))
                edges += self._process_allvaluesfrom(expr, virtual_node2)
            elif isinstance(expr, OWLObjectComplementOf):
                virtual_node2 = self.get_negation_virtual_node()
                edges.append(self._projectionMorphism(virtual_node, virtual_node2))
                edges += self._process_negationof(expr, virtual_node2)
            elif isinstance(expr, OWLObjectIntersectionOf):
                virtual_node2 = self.get_intersection_virtual_node()
                edges.append(self._projectionMorphism(virtual_node, virtual_node2))
                edges += self._process_intersectionof(expr, virtual_node2)
            elif isinstance(expr, OWLObjectUnionOf):
                virtual_node2 = self.get_union_virtual_node()
                edges.append(self._projectionMorphism(virtual_node, virtual_node2))
                edges += self._process_unionof(expr, virtual_node2)
            elif isinstance(expr, (OWLDataHasValue, OWLDataSomeValuesFrom, OWLObjectExactCardinality, OWLObjectMinCardinality)):
                #Ignore this type for now
                explicitely_ignored = True
            else:
                print("process_intersectionof: Unknown expression type: {}".format(expr))

        if edges == [] and not explicitely_ignored:
            print(f"No edges found for IntersectionOf axiom: {intersection_of}")
        return edges

    def _process_unionof(self, union_of: OWLObjectUnionOf, virtual_node):
        """Process an OWLObjectUnionOf and return a list of edges."""

        edges = []
        for expr in union_of.getOperandsAsList():
            if isinstance(expr, OWLClass):
                edges.append(self._injectionMorphism(expr, virtual_node))
            elif isinstance(expr, OWLObjectSomeValuesFrom):
                virtual_node2 = self.get_existential_virtual_node()
                edges.append(self._injectionMorphism(virtual_node2, virtual_node))
                edges += self._process_somevaluesfrom(expr, virtual_node2)
            elif isinstance(expr, OWLObjectAllValuesFrom):
                virtual_node2 = self.get_universal_virtual_node()
                edges.append(self._injectionMorphism(virtual_node2, virtual_node))
                edges += self._process_allvaluesfrom(expr, virtual_node2)
            elif isinstance(expr, OWLObjectIntersectionOf):
                virtual_node2 = self.get_intersection_virtual_node()
                edges.append(self._injectionMorphism(virtual_node2, virtual_node))
                edges += self._process_intersectionof(expr, virtual_node2)
            else:
                print("process_unionof: Unknown expression type: {}".format(expr))
        
        if edges == []:
            print(f"No edges found for UnionOf axiom: {union_of}")
        return edges

    def _process_somevaluesfrom(self, some_values_from: OWLObjectSomeValuesFrom, virtual_node):
        edges = []
        property_ = some_values_from.getProperty()
        filler = some_values_from.getFiller()

        edges.append(self._projectionMorphism(virtual_node, property_))
        if isinstance(filler, OWLClass):
            edges.append(self._projectionMorphism(virtual_node, filler))
        elif isinstance(filler, OWLObjectIntersectionOf):
            inter_virtual_node = self.get_intersection_virtual_node()
            edges.append(self._projectionMorphism(virtual_node, inter_virtual_node))
            edges += self._process_intersectionof(filler, inter_virtual_node)
        elif isinstance(filler, OWLObjectUnionOf):
            union_virtual_node = self.get_union_virtual_node()
            edges.append(self._projectionMorphism(virtual_node, union_virtual_node))
            edges += self._process_unionof(filler, union_virtual_node)
        elif isinstance(filler, OWLObjectComplementOf):
            neg_virtual_node = self.get_negation_virtual_node()
            edges += self._process_negationof(filler, neg_virtual_node)
        elif isinstance(filler, OWLObjectSomeValuesFrom):
            exist_virtual_node = self.get_existential_virtual_node()
            edges += self._process_somevaluesfrom(filler, exist_virtual_node)
        else:
            print("process_somevaluesfrom: Unknown filler type: {}".format(filler))
        
        if edges == []:
            print(f"No edges found for SomeValuesFrom axiom: {some_values_from}")   
        return edges

    def _process_allvaluesfrom(self, all_values_from: OWLObjectAllValuesFrom, virtual_node):
        edges = []
        property_ = all_values_from.getProperty()
        filler = all_values_from.getFiller()
        
        if isinstance(filler, OWLClass):
            edges.append(self._negatedInjectionMorphism(property_, virtual_node))
            edges.append(self._injectionMorphism(filler, virtual_node))
            edges.append(self._implicationMorphism(property_, filler))
        elif isinstance(filler, OWLObjectUnionOf):
            union_virtual_node = self.get_union_virtual_node()
            edges.append(self._negatedInjectionMorphism(property_, virtual_node))
            edges.append(self._injectionMorphism(union_virtual_node, virtual_node))
            edges += self._process_unionof(filler, union_virtual_node)
            edges.append(self._implicationMorphism(property_, union_virtual_node))
            
        else:
            print(f"process_allvaluesfrom: Unknown filler type: {filler}")

        return edges
    
    def _process_negationof(self, complement_of: OWLObjectComplementOf, virtual_node):
        edges = []
        expr = complement_of.getOperand()
        if isinstance(expr, OWLClass):
            virtual_node_int = self.get_intersection_virtual_node() 
            edges.append(self._projectionMorphism(virtual_node_int, expr))
            edges.append(self._projectionMorphism(virtual_node_int, virtual_node))
            edges += self._subclassMorphism(virtual_node_int, BOT)
 
            virtual_node_union = self.get_union_virtual_node()
            edges += self._subclassMorphism(TOP, virtual_node_union)
            edges.append(self._injectionMorphism(expr, virtual_node_union))
            edges.append(self._injectionMorphism(virtual_node, virtual_node_union))
        elif isinstance(expr, OWLObjectSomeValuesFrom):
            virtual_node_int = self.get_intersection_virtual_node()
            virtual_node_union = self.get_union_virtual_node()
            virtual_node_exist = self.get_existential_virtual_node()
            #TODO: assert virtual node exist already exists
            edges.append(self._projectionMorphism(virtual_node_int, virtual_node_exist))
            edges.append(self._projectionMorphism(virtual_node_int, virtual_node))
            edges += self._subclassMorphism(virtual_node_int, BOT)
            edges += self._subclassMorphism(TOP, virtual_node_union)
            edges.append(self._injectionMorphism(virtual_node_exist, virtual_node_union))
            edges.append(self._injectionMorphism(virtual_node, virtual_node_union))
        else:
            print(f"process_negationof: Unknown expression type: {expr}")
        return edges    
#    def negationMorphism(go_class: OWLClassExpression):
#        raise NotImplementedError()

#        val go_class_OWLClass = go_class.asInstanceOf[OWLClass]
#        new Triple(s"Not_${goClassToStr(go_class_OWLClass)}", "negate", go_class_OWLClass)

 #   def injectionMorphism(src: OWLClassExpression, dst: OWLClassExpression, rel: Option[String] = None
 #       val src_OWLClass = src.asInstanceOf[OWLClass]
 #       val dst_OWLClass = dst.asInstanceOf[OWLClass]
 #       rel match {
 #           case Some(r) => new Triple(src_OWLClass, "injects_" + r, dst_OWLClass)
 #           case None => new Triple(src_OWLClass, "injects", dst_OWLClass)
 #       }
 #   }

 ##### ASSERTION AXIOMS #####

    def _process_class_assertion(self, class_assertion: OWLClassAssertionAxiom):
        edges = []
        individual = class_assertion.getIndividual()
        cls = class_assertion.getClassExpression()
        if isinstance(cls, OWLClass):
            edges.append(self._membershipMorphism(individual, cls))
        else:
            print(f"process_class_assertion: Unknown class expression type: {cls}")
        return edges


    def _process_object_property_assertion(self, object_property_assertion: OWLObjectPropertyAssertionAxiom):
        edges = []
        subject = object_property_assertion.getSubject()
        property_ = object_property_assertion.getProperty()
        obj = object_property_assertion.getObject()

        if not isinstance(subject, OWLNamedIndividual):
            print(f"process_object_property_assertion: Unknown subject type: {subject}")
        if not isinstance(obj, OWLNamedIndividual):
            print(f"process_object_property_assertion: Unknown object type: {obj}")

        if not isinstance(property_, OWLObjectProperty):
            print(f"process_object_property_assertion: Unknown property type: {property_}")
        edges.append(self._propertyAssertionMorphism(subject, property_, obj))
        return edges
####################### MORPHISMS ###########################
    def _projectionMorphism(self, src, dst):
        if isinstance(src, OWLClass):
            src = str(src.toStringID())
        elif isinstance(src, OWLObjectProperty):
            src = str(src.toString())[1:-1]
        else:
            if not isinstance(src, str):
                raise TypeError(f"Unknown source type: {src}. Type: {type(src)}")

        if isinstance(dst, OWLClass):
            dst = str(dst.toStringID())
        elif isinstance(dst, OWLObjectProperty):
            dst = str(dst.toString())[1:-1] 
        else:
            if not isinstance(dst, str):
                raise TypeError(f"Unknown type for dst: {dst}")


        src = check_entity_format(src)
        dst = check_entity_format(dst)
        return Edge(src, "http://projects", dst)

    def _membershipMorphism(self, src, dst):
        if isinstance(src, OWLNamedIndividual):
            src = str(src.toStringID())
        else:
            if not isinstance(src, str):
                raise TypeError(f"Unknown source type: {src}. Type: {type(src)}")

        if isinstance(dst, OWLClass):
            dst = str(dst.toStringID())
        else:
            if not isinstance(dst, str):
                raise TypeError(f"Unknown type for dst: {dst}")

        src = check_entity_format(src)
        dst = check_entity_format(dst)

        return Edge(src, "http://member", dst)

    def _propertyAssertionMorphism(self, src, property_, dst):
        if isinstance(src, OWLNamedIndividual):
            src = str(src.toStringID())
        else:
            if not isinstance(src, str):
                raise TypeError(f"Unknown source type: {src}. Type: {type(src)}")

        if isinstance(dst, OWLNamedIndividual):
            dst = str(dst.toStringID())
        else:
            if not isinstance(dst, str):
                raise TypeError(f"Unknown type for dst: {dst}")

        if isinstance(property_, OWLObjectProperty):
            property_ = str(property_.toString())[1:-1]
        else:
            if not isinstance(property_, str):
                raise TypeError(f"Unknown type for property_: {property_}")

        src = check_entity_format(src)
        dst = check_entity_format(dst)
        property_ = check_entity_format(property_)
        return Edge(src, property_, dst)

    def _subclassMorphism(self, src, dst):
        if isinstance(src, OWLClass):
            src = str(src.toStringID())
        elif isinstance(src, OWLObjectProperty):
            src = str(src.toString())[1:-1]
        else:
            if not isinstance(src, str):
                print(type(src))

        if isinstance(dst, OWLClass):
            dst = str(dst.toStringID())
            
        elif isinstance(dst, OWLObjectProperty):
            dst = str(dst.toString())[1:-1]
        else:
            if not isinstance(dst, str):
                raise TypeError(f"Unknown type for dst: {dst}")

        src = check_entity_format(src)
        dst = check_entity_format(dst)
        
        if self.bidirectional_taxonomy:
            return [Edge(src, "http://subclassof", dst), Edge(dst, "http://superclassof", src)]
        else:
            return [Edge(src, "http://subclassof", dst)]

    def _implicationMorphism(self, src, dst):

        if isinstance(src, OWLObjectProperty):
            src = str(src.toString())[1:-1]
        else:
            if not isinstance(src, str):
                raise TypeError(f"implicationMorphism: Unknown type for src: {src}. Type: {type(src)}")
        
        if isinstance(dst, OWLClass):
            dst = str(dst.toStringID())
        else:
            if not isinstance(dst, str):
                raise TypeError(f"Unknown type for dst: {dst}")
        
        src = check_entity_format(src)
        dst = check_entity_format(dst)
        return Edge(src, "http://implies", dst)

    def _injectionMorphism(self, src, dst):

        if isinstance(src, OWLClass):
            src = str(src.toStringID())
        else:
            if not isinstance(src, str):
                raise TypeError(f"injectionMorphism: Unknown type for src: {src}: Type: {type(src)}")
        
        if isinstance(dst, OWLClass):
            dst = str(dst.toStringID())
        elif not isinstance(dst, str):
            raise TypeError(f"Unknown type for dst: {dst}. Type: {type(dst)}")
        
        src = check_entity_format(src)
        dst = check_entity_format(dst)
        return Edge(src, "http://injects", dst)

    def _negatedInjectionMorphism(self, src, dst):
        if isinstance(src, OWLObjectProperty):
            src = str(src.toString())
        elif not isinstance(src, str):
            raise TypeError(f"negatedInjectionMorphism: Unknown type for src: {src}. Type: {type(src)}")
        
        if isinstance(dst, OWLObjectProperty):
            dst = str(dst.toString())
        elif not isinstance(dst, str):
            raise TypeError(f"Unknown type for dst: {dst}. Type: {type(dst)}")

        src = check_entity_format(src)
        dst = check_entity_format(dst)
        return Edge(src, "http://negatively_injects", dst)

def check_entity_format(entity):
    """Check if the entity is in the correct format."""
    well_formed = True
    if entity.startswith("<"):
        entity = entity[1:]
        well_formed &= True 
    if entity.endswith(">"):
        entity = entity[:-1]
        well_formed &= True
    if not entity.startswith("http://"):
        well_formed &= False
    if not well_formed:
        raise ValueError(f"Entity {entity} is not well formed.")
    return entity


if __name__ == "__main__":
    if len(sys.argv) == 2:
        ontology_file = sys.argv[1]
        bidirectional_taxonomy = False
    if len(sys.argv) == 3:
        ontology_file = sys.argv[1]
        bidirectional_taxonomy = sys.argv[2]
        
    ds = PathDataset(ontology_file)
    projector = CategoricalProjector(bidirectional_taxonomy=bidirectional_taxonomy)
    graph = projector.project(ds.ontology)

    if bidirectional_taxonomy:
        outfile = ontology_file.replace(".owl", ".cat.projection.bi.edgelist")
    else:
        outfile = ontology_file.replace(".owl", ".cat.projection.edgelist")
        
    print(f"Graph computed. Writing into file: {outfile}")

    with open(outfile, "w") as f:
        for edge in tqdm(graph, total=len(graph)):
            f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")
    print("Done.")
    
