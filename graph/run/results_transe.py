import pandas as pd
import sys

def analyze_results(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ["epochs_first", "epochs_second", "embedding_size", "lr", "margin", "hits1", "hits5", "hits10", "mrr", "fhits1", "fhits5", "fhits10", "fmrr"]

    hits1 = df["hits1"].max()
    hits5 = df["hits5"].max()
    hits10 = df["hits10"].max()
    mrr = df["mrr"].max()
    fhits1 = df["fhits1"].max()
    fhits5 = df["fhits5"].max()
    fhits10 = df["fhits10"].max()
    fmrr = df["fmrr"].max()

    print("Hits@1\tHits@5\tHits@10\tMRR\tFHits@1\tFHits@5\tFHits@10\tFMRR")
    print(f"{hits1:.4f}\t{hits5:.4f}\t{hits10:.4f}\t{mrr:.4f}\t{fhits1:.4f}\t{fhits5:.4f}\t{fhits10:.4f}\t{fmrr:.4f}")
    
if __name__ == "__main__":
    filepath = sys.argv[1]
    analyze_results(filepath)
