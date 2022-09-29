import pickle as pkl


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pkl.load(f)
