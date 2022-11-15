"""Script to transform graphs in ttl format into edgelist format"""
import sys
import rdflib
import tqdm

def ttl_to_edgelist(onto_file):
    g = rdflib.Graph()
    if onto_file.endswith('ttl') or onto_file.endswith('TTL'):
        g.parse(onto_file, format='turtle')
    else:
        raise Exception('File format not supported')

    with open(onto_file.replace('.ttl', '.edgelist'), 'w') as f:
        for s, p, o in tqdm.tqdm(g, total=len(g)):
            if " " in s or " " in o:
                continue
            if "oboInOwl" in p or "annotated" in p:
                continue
            f.write(str(s) + '\t' + str(p) + '\t' + str(o) + '\n')

if __name__ == '__main__':
    filename = sys.argv[1]
    ttl_to_edgelist(filename)
    print("Done")

