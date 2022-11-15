import rdflib
import time
import sys

def ow2rdf(owlfile, rdfout):
    start_time = time.time()
    
    g = rdflib.Graph()
    g.parse (owlfile, format='application/rdf+xml')
    g.serialize(destination=rdfout, format='turtle')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    owlfile = sys.argv[1]
    if not owlfile.endswith(".owl"):
        raise Exception("Input file must be an OWL file")

    rdfout = owlfile.replace(".owl", ".rdf.ttl")
    ow2rdf(owlfile, rdfout)
    
