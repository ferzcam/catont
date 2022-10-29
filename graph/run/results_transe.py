import pandas as pd
import sys

def analyze_results(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ["epochs_first", "epochs_second", "embedding_size", "lr", "margin", "hits1", "hits5", "hits10", "mrr", "fhits1", "fhits5", "fhits10", "fmrr"]

    fmrr = df["fmrr"]
    print(max(fmrr))


if __name__ == "__main__":
    filepath = sys.argv[1]
    analyze_results(filepath)
