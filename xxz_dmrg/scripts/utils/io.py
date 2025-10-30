import pickle
from datetime import datetime

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def timestamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)
