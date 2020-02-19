# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys

import argparse
import json
import logging
import pickle as pic
import time

import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
import sklearn.metrics

import torch
import torch.nn as nn

class MemoryTracer(object):
    def __init__(self, device):
        self.device = device

    def start(self):
        self.initial_memory = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_max_memory_allocated(self.device)

    def end(self):
        factor = 2**20

        self.used   = (torch.cuda.memory_allocated(self.device) - self.initial_memory) // factor
        self.peaked = (torch.cuda.max_memory_allocated(self.device) - self.initial_memory) // factor

def eval_model(args, model, eval_data, output_pred=False):
    logger = logging.getLogger("main")

    device = next(model.parameters()).device

    s_eval_batch = args.bs // args.grad_accum_step

    test_X, test_y = eval_data

    model.eval()

    pred_y = []
    with torch.no_grad():
        time_start_eval = time.time()
        for i_batch, i_start in enumerate(range(0, len(test_X), s_eval_batch)):
            i_end = min(len(test_X), i_start + s_eval_batch)

            batch_X = test_X[i_start:i_end]

            _pred_y = model(batch_X)

            pred_y += np.argmax(_pred_y.tolist(), axis=1).tolist()
        time_end_eval = time.time()
        logger.info("eval_time\t{}\t{}".format(
            time_end_eval - time_start_eval, len(test_X)
        ))

    if output_pred:
        return pred_y
    else:
        accu = sklearn.metrics.accuracy_score(test_y, pred_y)

        return accu

# `args` requires bs, epoch, lr, lr_decay_factor, lr_decay_freq,
# warmup_step, options
def train_model(args, model, optimizer, train_data, eval_data):
    logger = logging.getLogger("main")

    device = next(model.parameters()).device

    train_X, train_y = train_data

    best_eval = None
    best_eval_model = None

    counter_warmup = 0
    finish_warmup = False
    counter_decay = 0

    s_batch = args.bs // args.grad_accum_step
    for i_epoch in range(args.epoch):
        logger.info("start_epoch\t{}".format(i_epoch))

        if args.gpu >= 0:
            mem_trace = MemoryTracer(args.gpu)
            mem_trace.start()

        # Train
        model.train()

        train_X, train_y = shuffle(train_X, train_y)
        if args.n_epoch_data > 0:
            train_X = train_X[:args.n_epoch_data]
            train_y = train_y[:args.n_epoch_data]

        time_epoch_start = time.time()
        time_ignore = 0.0
        lst_loss = []
        for i_batch, i_start in enumerate(range(0, len(train_X), s_batch)):
            sys.stdout.write("{}/{}\r".format(i_batch, len(train_X)//s_batch))
            sys.stdout.flush()

            # Initialize gradient accumulation.
            if i_batch % args.grad_accum_step == 0:
                optimizer.zero_grad()
                n_loss = 0

            # Scheduling learning rate.
            if not finish_warmup:
                new_lr = args.lr * (counter_warmup+1) / args.warmup_step
                counter_warmup += 1
                if counter_warmup == args.warmup_step:
                    logger.info("finish warmup steps")
                    finish_warmup = True
            else:
                if i_batch == 0:
                    counter_decay += 1
                    if counter_decay % args.lr_decay_freq == 0:
                        logger.info("decay learning rate")
                new_lr = args.lr / args.lr_decay_factor ** (counter_decay // args.lr_decay_freq)

            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

            # Computing prediction & loss
            i_end = min(len(train_X), i_start + s_batch)

            batch_X = train_X[i_start:i_end]
            batch_y = train_y[i_start:i_end]

            score_y = model(batch_X)
            dummy_index = torch.LongTensor(list(range(len(batch_y)))).to(device)
            batch_y = torch.LongTensor(batch_y).to(device)
            #print(torch.max(score_y), torch.min(score_y))
            nll = -nn.functional.log_softmax(score_y, dim=1)[dummy_index, batch_y]

            n_loss += nll.size(0)
            loss = torch.sum(nll)

            loss.backward()

            # Update with accumulated gradients.
            if i_batch % args.grad_accum_step == args.grad_accum_step - 1:
                #print("----------------------")
                for param_name, param in model.named_parameters():
                    if param.grad is not None:
                        #print(param_name, torch.min(param.grad), torch.max(param.grad))
                        param.grad.div_(n_loss)
                optimizer.step()

            lst_loss.append(loss.item())
        time_epoch_end = time.time()

        # Check memory usage
        if args.gpu >= 0:
            mem_trace.end()
            logger.info("train_memory_usage\t{}\t{}\t(MB)".format(
                mem_trace.used, mem_trace.peaked
            ))

        print("Train loss: {}".format(np.mean(lst_loss)))

        train_accu = eval_model(args, model, train_data)
        print("Train accuracy: {}".format(train_accu))

        accu = eval_model(args, model, eval_data)
        print("Test accuracy: {}".format(accu))

        logger.info("train_epoch_result\t{}\t{}\t{}\t{}".format(
            i_epoch, np.mean(lst_loss), train_accu, accu
        ))
        logger.info("train_epoch_time\t{}\t{}".format(
            time_epoch_end - time_epoch_start - time_ignore, len(train_X)
        ))

        if (best_eval is None) or (best_eval < accu):
            logger.info("new best model")
            best_eval_model = pic.loads(pic.dumps(model))
            best_eval = accu

    return best_eval_model, best_eval

def construct_optimizer(args, model):
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.decay)

def setup_logger(args):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s\t%(message)s")

    handler = logging.FileHandler("logs/{}.log".format(args.suffix))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# `args` requires test_ratio, test_seed, mode, eval_seed, suffix, valid_ratio
# in addition to options for train_model(), eval_model(), construct_model()
def train_eval(args, construct_model, log_fn=None):
    setup_logger(args)

    logger = logging.getLogger("main")
    logger.info("arg_info\t{}".format(args))

    ## Load dataset.
    train_data, test_data = load_dataset(args)
    train_X, train_y = train_data
    test_X, test_y = test_data
    logger.info("finish loading data")

    ## Training
    if args.mode == "k-fold":
        fold_results = []

        kf = KFold(n_splits=args.n_fold, shuffle=True, random_state=args.eval_seed)
        folds = kf.split(train_X)

        for i_fold, (train_inds, eval_inds) in enumerate(folds):
            logger.info(f"Fold::{i_fold}")

            fold_train_X = [train_X[_] for _ in train_inds]
            fold_train_y = [train_y[_] for _ in train_inds]
            fold_eval_X = [train_X[_] for _ in eval_inds]
            fold_eval_y = [train_y[_] for _ in eval_inds]

            logger.info("creating model")

            model = construct_model(args)

            device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)
            model.to(device)

            optimizer = construct_optimizer(args, model)

            logger.info("start training")
            _, best_eval = train_model(args, model, optimizer,
                (fold_train_X, fold_train_y), (fold_eval_X, fold_eval_y))

            fold_results.append(best_eval)

        logger.info("Fold metrics\t{}".format(fold_results))
        logger.info("Mean metric\t{}".format(np.mean(fold_results)))

        if log_fn is not None:
            with open(log_fn, "a") as h:
                h.write("{}\t{}\t{}\t{}\n".format(args.suffix, np.mean(fold_results), str(args), fold_results))
    elif args.mode == "test":
        train_X, dev_X, train_y, dev_y = train_test_split(
            train_X, train_y, test_size=args.valid_ratio, random_state=args.eval_seed
        )

        logger.info("creating model")

        if args.gpu >= 0:
            mem_trace = MemoryTracer(args.gpu)
            mem_trace.start()

        model = construct_model(args)

        device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)
        model.to(device)

        optimizer = construct_optimizer(args, model)

        if args.gpu >= 0:
            mem_trace.end()
            logger.info("model_memory_usage\t{}\t{}\t(MB)".format(
                mem_trace.used, mem_trace.peaked
            ))

        logger.info("start training")
        best_model, best_eval = train_model(args, model, optimizer,
            (train_X, train_y), (dev_X, dev_y))

        # Save models.
        torch.save(best_model.state_dict(), "models/{}.mdl".format(args.suffix))

        logger.info("evaluating on test data")
        test_eval = eval_model(args, best_model, (test_X, test_y))

        logger.info("Valid metric\t{}".format(best_eval))
        logger.info("Test metric\t{}".format(test_eval))

        if log_fn is not None:
            with open(log_fn, "a") as h:
                h.write("{}\t{}\t{}\n".format(args.suffix, test_eval, str(args)))
    elif args.mode == "valid":
        train_X, dev_X, train_y, dev_y = train_test_split(
            train_X, train_y, test_size=args.valid_ratio, random_state=args.eval_seed
        )

        logger.info("creating model")

        if args.gpu >= 0:
            mem_trace = MemoryTracer(args.gpu)
            mem_trace.start()

        model = construct_model(args)

        device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)
        model.to(device)

        optimizer = construct_optimizer(args, model)

        if args.gpu >= 0:
            mem_trace.end()
            logger.info("model_memory_usage\t{}\t{}\t(MB)".format(
                mem_trace.used, mem_trace.peaked
            ))

        logger.info("start training")
        best_model, best_eval = train_model(args, model, optimizer,
            (train_X, train_y), (dev_X, dev_y))

        logger.info("Valid metric\t{}".format(best_eval))

        if log_fn is not None:
            with open(log_fn, "a") as h:
                h.write("{}\t{}\t{}\n".format(args.suffix, best_eval, str(args)))

def load_dataset(args):
    with open("{}/train.{}".format(args.dataset, args.alg)) as h:
        train_X = json.load(h)
        if args.alg == "lzd":
            train_X = [[(_[1], _[2]) for _ in _seq] for _seq in train_X]
    with open("{}/test.{}".format(args.dataset, args.alg)) as h:
        test_X = json.load(h)
        if args.alg == "lzd":
            test_X = [[(_[1], _[2]) for _ in _seq] for _seq in test_X]

    with open("{}/train.label".format(args.dataset)) as h:
        train_y = json.load(h)
    with open("{}/test.label".format(args.dataset)) as h:
        test_y = json.load(h)

    return (train_X, train_y), (test_X, test_y)

def common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=-1,
        help="ID of GPU device used in the experiment. CPU is used if it is -1.")
    parser.add_argument("--suffix", type=str, default="tmp",
        help="Name of the experiment. It is used to distinguish log files.")

    # Dataset configurations
    dataset_group = parser.add_argument_group("Dataset configurations")
    dataset_group.add_argument("--dataset", type=str, help="Dataset directory. (e.g. \"data/dna/H3\")")
    dataset_group.add_argument("--n-char", type=int, default=5,
        help="Number of character types of the dataset.")
    dataset_group.add_argument("--n-cls", type=int, default=2,
        help="Number of class categories of the dataset.")
    dataset_group.add_argument("--alg", type=str, choices=["lzd", "repair", "uncomp"],
        default="repair", help="Type of data compression (choose 'uncomp' for baseline models).")

    # Evaluation
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--mode", type=str, choices=["k-fold", "test", "valid"],
        help="[valid]: Train on training set and select best model on development set. [test]: Train on training set, select best model on development set, and evaluates on test set.")
    eval_group.add_argument("--eval-seed", type=int, default=4321,
        help="Seed value used during splitting develpment set.")
    eval_group.add_argument("--n-fold", type=int, default=5,
        help="Number of folds. Only used in k-fold mode. default: 5")
    eval_group.add_argument("--valid-ratio", type=int, default=0.2,
        help="Ratio of development data which are held-out from training data. default: 0.2")
    eval_group.add_argument("--trained-model-fn", type=str, default=None)
    eval_group.add_argument("--result-fn", type=str, default="out.json")

    # (Common) Training hyperparameters
    hyperparam_group = parser.add_argument_group("Training hyperparameters")
    hyperparam_group.add_argument("--bs", type=int, default=10, help="Batch size. default: 10")
    hyperparam_group.add_argument("--grad-accum-step", type=int, default=1,
        help="Number of steps to accumulate gradient. default: 1 (step)")
    hyperparam_group.add_argument("--n-epoch-data", type=int, default=-1,
        help="Number of maximum training data per one epoch. defalt: Unlimited")
    hyperparam_group.add_argument("--optimizer", type=str, choices=["Adam", "SGD"],
        default="Adam", help="Type of optimizer. default: Adam")
    hyperparam_group.add_argument("--momentum", type=float, default=0.9,
        help="Momentum (Only used with SGD optimizer.) default: 0.9")
    hyperparam_group.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate. default: 1e-3")
    hyperparam_group.add_argument("--decay", type=float, default=0.0,
        help="Weight decay hyperparameter. default: 0.0")
    hyperparam_group.add_argument("--epoch", type=int, default=50,
        help="Number of training epochs. default: 50 (epochs)")
    hyperparam_group.add_argument("--lr-decay-freq", type=int, default=10,
        help="How many epochs to wait between learning rate decay. default: 10 (epochs)")
    hyperparam_group.add_argument("--lr-decay-factor", type=float, default=2,
        help="Factor of learning rate decay. default: 2")
    hyperparam_group.add_argument("--warmup-step", type=int, default=1000,
        help="Number of warmup steps. default: 1000 (steps)")

    return parser
