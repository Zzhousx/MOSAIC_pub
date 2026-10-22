import random
import numpy as np
import torch
import os
import copy
import importlib
import datetime
import argparse

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name: str, model_type: str):
    """
    Automatically select model class based on model name

    Args:
        model_name (str): model name
        model_type (str): rec, fs, es

    Returns:
        Recommender: model class
        Dict: model configuration dict
    """
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['models', model_type, model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    else:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    # dir = os.path.dirname(model_module.__file__)
    # conf = dict()
    # fname = os.path.join(os.path.dirname(dir), 'basemodel', 'basemodel.yaml')
    # conf.update(parser_yaml(fname))
    # for name in ['all', model_file_name]:
    #     fname = os.path.join(dir, 'config', name+'.yaml')
    #     if os.path.isfile(fname):
    #         conf = deep_update(conf, parser_yaml(fname))
    return model_class


class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True



def print_time(message):
    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S '), message)

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
