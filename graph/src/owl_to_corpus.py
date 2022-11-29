import mowl
mowl.init_jvm("10g")
import sys
from mowl.corpus import extract_and_save_axiom_corpus, extract_and_save_annotation_corpus
from mowl.datasets import PathDataset

def generate_corpus(filepath):
    dataset = PathDataset(filepath)
    outfile = filepath.replace('.owl', '.corpus')
    extract_and_save_axiom_corpus(dataset.ontology, outfile)
    extract_and_save_annotation_corpus(dataset.ontology, outfile, "a")


if __name__ == '__main__':
    filepath = sys.argv[1]
    generate_corpus(filepath)
    print('Done')
