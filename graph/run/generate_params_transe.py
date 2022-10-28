import sys

def generate_params(outfile):

    with open(outfile, "w") as f:
        for epochs_first in [20, 40, 60]:
            for embedding_size in [50, 100, 200]:
                for epochs_second in [500, 1000, 2000]:
                    for learning_rate in [0.01, 0.001, 0.0001]:
                        for margin in [1, 2]:
                            f.write(f"{epochs_first} {embedding_size} {epochs_second} {learning_rate} {margin}\n")
                            
def generate_params_only_train(outfile):

    with open(outfile, "w") as f:
        for epochs_first in [20, 40, 60]:
            for embedding_size in [50, 100, 200]:
                f.write(f"{epochs_first} {embedding_size} 0 0 0\n")
                            
                                        


if __name__ == "__main__":
                
    outfile = sys.argv[1]
    mode = sys.argv[2]
    if mode == "train":
        generate_params_only_train(outfile)
    else:
        generate_params(outfile)
