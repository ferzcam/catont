import pandas as pd
import sys

def analyze_results(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ["num_walks", "walk_length", "alpha", "w2v_epochs", "window_size", "embedding_size", "hits1", "hits5", "hits10", "mrr", "fhits1", "fhits5", "fhits10", "fmrr"]

    fmrr = df["fmrr"]
    print(max(fmrr))


if __name__ == "__main__":
    filepath = sys.argv[1]
    analyze_results(filepath)
