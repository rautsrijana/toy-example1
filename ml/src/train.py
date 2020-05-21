from __future__ import print_function, division

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

from tqdm import tqdm
import utils as utils
import data_loader as data_loader
import models as net
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/256x256_testset',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='ml/experiments/',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

# def train_model(model, loss_fn, optimizer, scheduler, dataloaders, num_epochs=10):
def train(model, optimizer, loss_fn, dataloader, metrics, params):
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
                # move to GPU if available
                if params.cuda:
                    train_batch, labels_batch = train_batch.cuda(
                        non_blocking=True), labels_batch.cuda(non_blocking=True)
                # convert to torch Variables
                train_batch, labels_batch = Variable(
                    train_batch), Variable(labels_batch)

                # compute model output and loss
                output_batch = model(train_batch)
                loss = loss_fn(output_batch, labels_batch)

                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                # Evaluate summaries only once in a while
                if i % params.save_summary_steps == 0:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()

                    # compute all metrics on this batch
                    summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                     for metric in metrics}
                    summary_batch['loss'] = loss.item()
                    summ.append(summary_batch)

                # update the average loss
                loss_avg.update(loss.item())

                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                         for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 48
    feature_extract = True
    model_name = "resnet"
    model_ft, input_size = net.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    params = utils.Params("./ml/experiments/resnet/params.json")
    data_dir = 'data/256x256_testset'
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['train']

    train_and_evaluate(model_ft, train_dl, val_dl, optimizer_ft, loss_fn, net.metrics, params,
            args.model_dir, args.restore_file)
