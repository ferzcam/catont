import pandas as pd
import sys

def analyze_results(root):
    owl2vec_star_file = root + "owl2vecstar_results"
    categorical_file = root + "categorical_results"

    df_owl2vec = pd.read_csv(owl2vec_star_file, header=None)

    df_owl2vec.columns = ["num_walks", "walk_length", "alpha", "w2v_epochs", "window_size", "embedding_\
size", "hits1", "hits5", "hits10", "mrr", "fhits1", "fhits5", "fhits10", "fmrr"]

    df_cat = pd.read_csv(categorical_file, header=None)

    df_cat.columns = ["num_walks", "walk_length", "alpha", "w2v_epochs", "window_size", "embedding_size\
", "hits1", "hits5", "hits10", "mrr", "fhits1", "fhits5", "fhits10", "fmrr"]


    #Get best hits1
    best_owl2vec = df_owl2vec.loc[df_owl2vec['hits1'].idxmax()]
    best_cat = df_cat.loc[df_cat['hits1'].idxmax()]

    print("Best owl2vec hits1: " + str(best_owl2vec['hits1']))
    print("Best cat hits1: " + str(best_cat['hits1']))

    #Get best hits5
    best_owl2vec = df_owl2vec.loc[df_owl2vec['hits5'].idxmax()]
    best_cat = df_cat.loc[df_cat['hits5'].idxmax()]

    print("Best owl2vec hits5: " + str(best_owl2vec['hits5']))
    print("Best cat hits5: " + str(best_cat['hits5']))

    #Get best hits10
    best_owl2vec = df_owl2vec.loc[df_owl2vec['hits10'].idxmax()]
    best_cat = df_cat.loc[df_cat['hits10'].idxmax()]

    print("Best owl2vec hits10: " + str(best_owl2vec['hits10']))
    print("Best cat hits10: " + str(best_cat['hits10']))

    #Get best mrr
    best_owl2vec = df_owl2vec.loc[df_owl2vec['mrr'].idxmax()]
    best_cat = df_cat.loc[df_cat['mrr'].idxmax()]

    print("Best owl2vec mrr: " + str(best_owl2vec['mrr']))
    print("Best cat mrr: " + str(best_cat['mrr']))

    #Get best fhits1
    best_owl2vec = df_owl2vec.loc[df_owl2vec['fhits1'].idxmax()]
    best_cat = df_cat.loc[df_cat['fhits1'].idxmax()]

    print("Best owl2vec fhits1: " + str(best_owl2vec['fhits1']))
    print("Best cat fhits1: " + str(best_cat['fhits1']))

    #Get best fhits5
    best_owl2vec = df_owl2vec.loc[df_owl2vec['fhits5'].idxmax()]
    best_cat = df_cat.loc[df_cat['fhits5'].idxmax()]

    print("Best owl2vec fhits5: " + str(best_owl2vec['fhits5']))
    print("Best cat fhits5: " + str(best_cat['fhits5']))

    #Get best fhits10
    best_owl2vec = df_owl2vec.loc[df_owl2vec['fhits10'].idxmax()]
    best_cat = df_cat.loc[df_cat['fhits10'].idxmax()]

    print("Best owl2vec fhits10: " + str(best_owl2vec['fhits10']))
    print("Best cat fhits10: " + str(best_cat['fhits10']))

    #Get best fmrr
    best_owl2vec = df_owl2vec.loc[df_owl2vec['fmrr'].idxmax()]
    best_cat = df_cat.loc[df_cat['fmrr'].idxmax()]

    print("Best owl2vec fmrr: " + str(best_owl2vec['fmrr']))
    print("Best cat fmrr: " + str(best_cat['fmrr']))
            
if __name__ == "__main__":                                                                              
    case = sys.argv[1]                                                                                  
    root = "../case_studies/"                                                                           
    if case == "go":                                                                                    
        root += "go_subsumption/"                                                                       
    elif case == "foodon":                                                                              
        root += "foodon_subsumption/"                                                                   
    elif case == "helis":                                                                               
        root += "helis_membership/"                                                                     
                                                                                                        
    analyze_results(root)                                                                               
                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                        
                                          
