"""Generates a graph from ontology using OWL2VecStar projection rules"""
import sys
from owl2vec_star.lib.Onto_Projection import Reasoner, OntologyProjection

def owl2vecstar_projection(ontology_file):

    projection = OntologyProjection(ontology_file, reasoner=Reasoner.STRUCTURAL, only_taxonomy=False,
                                    bidirectional_taxonomy=True, include_literals=False, avoid_properties=set(),
                                    additional_preferred_labels_annotations=set(),
                                    additional_synonyms_annotations=set(),
                                    memory_reasoner='13351')

    projection.extractProjection()
    output_file = ontology_file.replace('.owl', '.only.graph.projection.ttl')
    projection.saveProjectionGraph(output_file)


if __name__ == '__main__':
    ontology_file = sys.argv[1]
    owl2vecstar_projection(ontology_file)
    print("Done!")
