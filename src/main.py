#!/usr/bin/env python

from __future__ import division

import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # ['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased','albert-base-v2','albert-large-v2','microsoft/deberta-large','microsoft/deberta-base', 'funnel-transformer/medium']
    # ['hfl/chinese-macbert-base','hfl/chinese-macbert-large']
    parser = argparse.ArgumentParser()
    # Basic args_
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'test'])
    parser.add_argument("-data_path", default='data', type=str)
    parser.add_argument("-model_path", default='models', type=str)
    parser.add_argument("-result_path", default='results', type=str)
    parser.add_argument('-task', default='qqp', type=str, choices=['qqp', 'mrpc', 'medical'])
    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument('-loss_type', default=1, type=int)  # For ablation

    # Batch sizes
    parser.add_argument("-batch_size", default=16, type=int)
    parser.add_argument("-test_batch_size", default=64, type=int)

    # Model args
    parser.add_argument("-baseline", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-model", default="", type=str)
    parser.add_argument("-num_labels", default=2, type=int)

    # Training process args
    parser.add_argument("-save_checkpoint_steps", default=2000, type=int)
    parser.add_argument("-accum_count", default=4, type=int)
    parser.add_argument("-report_every", default=5, type=int)
    parser.add_argument("-train_steps", default=50000, type=int)

    # Optim args
    parser.add_argument("-lr", default=2e-05, type=float)
    parser.add_argument("-warmup", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-warmup_steps", default=1000, type=int)
    parser.add_argument("-weight_decay", default=0.01, type=float)
    parser.add_argument("-max_grad_norm", default=1.0, type=float)

    # Utility args
    parser.add_argument("-test_from", default='', type=str)
    parser.add_argument("-train_from", default='', type=str)
    parser.add_argument("-debug", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    model_name = args.model.replace('/', '-')
    args.model_path = args.model_path + '/' + args.task + '/' + model_name
    args.result_path = args.result_path + '/' + args.task + '/' + model_name + '.txt'
    args.data_path = args.data_path + '/' + args.task + '/' + model_name + '.save'

    from train import train
    from test import test

    if (args.mode == 'train'):
        train(args)
    elif (args.mode == 'test'):
        test(args)
    else:
        print("Undefined mode! Please check input.")
