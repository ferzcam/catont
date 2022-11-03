import sys

def generate_params_go(outfile):

    with open(outfile, "w") as f:
        for num_walks in [20,30,40]:
            for walk_length in [10,15]:
                for epochs_w2v in [15,30, 50]:
                    for window_size in [9]:
                        for embedding_size in [50,100, 200]:
                            #f.write(f"{num_walks} {walk_length} {epochs_w2v} {window_size} {embedding_size} 0\n")
                            for learning_rate in [0.01, 0.001, 0.0001]:
                                f.write(f"{num_walks} {walk_length} {epochs_w2v} {window_size} {embedding_size} {learning_rate}\n")



def generate_params_helis(outfile):

    with open(outfile, "w") as f:
        for num_walks in [10, 20,30]:
            for walk_length in [10, 20]:
                for epochs_w2v in [15,30]:
                    for window_size in [3, 5]:
                        for embedding_size in [20, 50,100]:
                            #f.write(f"{num_walks} {walk_length} {epochs_w2v} {window_size} {embedding_size} 0\n")
                            for learning_rate in [0.01, 0.001, 0.0001]:
                                f.write(f"{num_walks} {walk_length} {epochs_w2v} {window_size} {embedding_size} {learning_rate}\n")


if __name__ == "__main__":

    case = sys.argv[1]
    outfile = sys.argv[2]

    if case == "go":
        print("Generating params for GO")
        generate_params_go(outfile)
    elif case == "helis":
        print("Generating params for Helis")
        generate_params_helis(outfile)
