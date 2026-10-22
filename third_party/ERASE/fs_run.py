# Derived from ERASE; restricted to the four paper ranking methods.
import numpy as np
import torch,os,argparse,yaml,time
import utils.utils as utils
from utils.fs_trainer import modeltrainer
from utils.datasets import read_dataset
from models.basemodel import BaseModel

def main(args):
    if args.seed != 0:
        utils.seed_everything(args.seed)
    features, label, train_dataloader, val_dataloader, test_dataloader, unique_values = read_dataset(args.dataset, args.data_path, batch_size=args.batch_size, shuffle=args.dataset_shuffle, num_workers=args.num_workers, machine_learning_method=False)
    print(features)
    print(unique_values)
    model = BaseModel(args, args.model, args.fs, args.es, unique_values, features)
    model.fs.mode = 'train'
    trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=False)
    trainer.fit(train_dataloader, val_dataloader)
    auc = trainer.test(test_dataloader, ['auc', 'logloss'])

    if hasattr(model.fs, 'save_selection'):
        model_path = 'checkpoints/' + args.model + '_' + args.fs + '_' + args.es + '_' + args.dataset + '_' + args.timestr + '/'
        res = model.fs.save_selection(k=args.k)
        if isinstance(res, np.ndarray):
            feature_rank = res
        else:
            feature_rank = res(val_dataloader, model, torch.device(args.device))
        utils.print_time('feature rank:')
        utils.print_time(feature_rank)
        np.save(model_path + 'feature_rank.npy', feature_rank)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avazu', choices=['avazu','criteo','movielens-1m'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'])
    parser.add_argument('--fs', type=str, default='autofield', choices=['autofield','lpfs','sfs','shark'])
    parser.add_argument('--es', type=str, default='no_selection', choices=['no_selection'])
    parser.add_argument('--seed', type=int, default=0, help='random seed, 0 represents not setting the random seed')
    parser.add_argument('--device',type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu, cuda')
    parser.add_argument('--data_path', type=str, default='data/', help='data path') # ~/autodl-tmp/ or data/
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--dataset_shuffle', type=bool, default=True, help='whether to shuffle the dataset')
    parser.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension')
    parser.add_argument('--train_or_search', type=utils.str2bool, default=True, help='whether to train or search')
    parser.add_argument('--retrain', type=utils.str2bool, default=False, choices=[False], help='uniform evaluation is handled by MOSAIC')
    parser.add_argument('--k', type=int, default=0, help='top k features')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='epoch')
    parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=32, help='num_workers')

    args = parser.parse_args()

    # k
    if args.k == 0:
        if args.dataset == 'avazu':
            args.k = 22
        elif args.dataset == 'criteo':
            args.k = 39

    with open('models/config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    args.__dict__.update(data)

    # print args
    for key in args.__dict__:
        if key not in ['fs_config', 'es_config', 'rec_config']:
            print(key, ':', args.__dict__[key])
        else:
            print(key, ':')
            for key2 in args.__dict__[key]:
                if key2 in [args.model, args.fs, args.es]:
                    print('\t', key2, ':', args.__dict__[key][key2])

    args.timestr = str(time.time())

    main(args)
