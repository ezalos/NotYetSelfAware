class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = DotDict()

config.learning_rate = 1e-1
config.seed = 466880822

config.cache_folder = "cache/"
config.mlp_dataset_path = "src/NotYetSelfAware/datasets/MLP/data.csv"
# config.mlp_dataset_path = "src/NotYetSelfAware/datasets/MLP/min.csv"
