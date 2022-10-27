import sys

def generate_params(outfile):

    with open(outfile, "w") as f:
        for num_walks in [10,20,30]:
            for walk_length in [5,10,15]:
                for alpha in [0.1, 0.2, 0.3]:
                    for epochs_w2v in [15,20,25]:
                        for window_size in [5,7,9]:
                            for embedding_size in [50,100,150]:
                                for epochs_mlp in [1000, 2000, 3000]:
                                    for learning_rate in [0.01, 0.001, 0.0001]:
                                        f.write(f"{num_walks} {walk_length} {alpha} {epochs_w2v} {window_size} {embedding_size} {epochs_mlp} {learning_rate}\n


if __name__ == "__main__":
                
    outfile = sys.argv[1]
    generate_params(outfile)
