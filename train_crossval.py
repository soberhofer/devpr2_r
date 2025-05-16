import os
import sys
import datetime
import importlib
import logging
import time
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.utils import EarlyStopping
from dataset.dataset_ESC50 import ESC50
import config

#random.seed(42)
# digits for logging
float_fmt = ".3f"

# mean and std of train data for every fold
global_stats = np.array([[-109.731209,41.657070],
                         [-109.567825,41.688847],
                         [-109.380600,41.618454],
                         [-109.435097,41.616096],
                         [-109.384743,41.937431]])
"""global_stats = np.array([[-54.364834, 20.853344],
                         [-54.279022, 20.847532],
                         [-54.18343, 20.80387],
                         [-54.223698, 20.798292],
                         [-54.200905, 20.949806]])"""

logger = logging.getLogger(__name__)
logger_fold = logging.getLogger("fold")

# evaluate model on different testing data 'dataloader'
def test(model, dataloader, criterion, device):
    model.eval()

    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.inference_mode():
        # no gradient computation needed
        for k, x, label in tqdm(dataloader, unit='bat', disable=config.disable_bat_pbar, position=0):
            x = x.float().to(device)
            y_true = label.to(device)

            # the forward pass through the model
            y_prob = model(x)

            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


def train_epoch():
    """
    Schaltet Modell auf Training und Iteriert durch
    :return:
    """
    # switch to training
    model.train()

    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=config.disable_bat_pbar, position=0):
        x = x.float().to(device)
        y_true = label.to(device)

        # the forward pass through the model
        y_prob = model(x)

        # we could also use 'F.one_hot(y_true)' for 'y_true', but this would be slower
        # loss ist ein skalar
        loss = criterion(y_prob, y_true)
        # reset the gradients to zero - avoids accumulation
        # gradient ist so groß wie viele parameter es gibt
        # Gradient im optimizer wird auf 0 gesetzt --> ist standard dass der bei jedem schritt auf 0 gesetzt wird
        # Kann auch vorkommen dass er über mehrere steps weiter verwendet wird --> daher so mühsam gelöst
        optimizer.zero_grad()
        # compute the gradient with backpropagation
        loss.backward()
        losses.append(loss.detach().item())
        # minimize the loss via the gradient - adapts the model parameters
        optimizer.step()
        # kleinere batch --> mehr steps --> mehr veränderung --> mehr random herumspringen
        # welcher opimizer sqg

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier():
    """ Parameter werden berechnet"""
    num_epochs = config.epochs

    loss_stopping = EarlyStopping(patience=config.patience, delta=0.002, verbose=True, float_fmt=float_fmt,
                                  checkpoint_file=os.path.join(experiment, 'best_val_loss.pt'))

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in (range(1, 1 + num_epochs)):
        start_time = time.perf_counter()
        # iterate once over training data
        train_acc, train_loss = train_epoch()

        # validate model
        val_acc, val_loss, _ = test(model, val_loader, criterion=criterion, device=device)
        val_loss_avg = np.mean(val_loss)

        pbar.update()
        # pbar.refresh() syncs output when pbar on stderr
        # pbar.refresh()
        early_stop, improved, msg_checkpoint = loss_stopping(val_loss_avg, model, epoch)
        logger_fold.info(f"    Epoch: {epoch}/{num_epochs}, "+
                         f"Time={time.perf_counter() - start_time:.2f}s, "+
                         f"TrnAcc={train_acc:{float_fmt}}, "+
                         f"ValAcc={val_acc:{float_fmt}}, "+
                         f"TrnLoss={np.mean(train_loss):{float_fmt}}, "+
                         f"ValLoss={val_loss_avg:{float_fmt}}{msg_checkpoint}")
        print(f"  Epoch: {epoch}/{num_epochs}, "+
              f"TrnAcc={train_acc:{float_fmt}}, "+
              f"ValAcc={val_acc:{float_fmt}}, "+
              f"TrnLoss={np.mean(train_loss):{float_fmt}}, "+
              f"ValLoss={val_loss_avg:{float_fmt}}{msg_checkpoint}")
        if not improved:
            print()
        if early_stop:
            logger.info("Early stopping")
            logger_fold.info('Early stopping')
            break

        # advance the optimization scheduler
        scheduler.step()
    # save full model
    torch.save(model.state_dict(), os.path.join(experiment, 'terminal.pt'))


# build model from configuration.
def make_model():
    model_path = config.model_path
    model_constructor = config.model_constructor

    modul = importlib.import_module(model_path)
    globals().update(vars(modul))

    model = eval(model_constructor)

    return model

def setup_logger(folder_path):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger.setLevel(logging.DEBUG)

    # Logfile
    file_handler = logging.FileHandler(os.path.join(folder_path, "logfile.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def main():
    global model, train_loader, val_loader, criterion, optimizer, scheduler, device, experiment

    # Result Path
    runs_path = config.runs_path
    experiment_root = os.path.join(runs_path, str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
    os.makedirs(experiment_root, exist_ok=True)

    setup_logger(experiment_root)

    # LOG config:
    logger.debug('Config Values')
    variables = {k: v for k, v in vars(config).items()
                 if not k.startswith("__") and not callable(v)}
    for name, value in variables.items():
        logger.debug(f"    {name}= {value}")

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_mps:
        device = torch.device("mps")
    elif use_cuda:
        device = torch.device(f"cuda:{config.device_id}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    data_path = config.esc50_path

    pd.options.display.float_format = ('{:,' + float_fmt + '}').format

    # for all folds
    scores = {}
    # expensive!
    #global_stats = get_global_stats(data_path)
    # for spectrograms
    #print("WARNING: Using hardcoded global mean and std. Depends on feature settings!")
    for test_fold in config.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')
        os.makedirs(experiment, exist_ok=True)

        logger_fold.setLevel(logging.DEBUG)
        if logger_fold.hasHandlers():
            logger_fold.handlers.clear()
        file_handler_fold = logging.FileHandler(os.path.join(experiment, 'train.log'))
        file_handler_fold.setLevel(logging.DEBUG)
        file_handler_fold.setFormatter(logging.Formatter('%(message)s'))
        logger_fold.addHandler(file_handler_fold)
        logger_fold.info("Logger_fold setup test")

        # this function assures consistent 'test_folds' setting for train, val, test splits
        # Instanzierugn der Datasets
        # partial == standardfunktion von python --> damit kann man einige parameter einer funktion bereits vorausfüllen --> erzeugt eine neue funktion
        # Es wird für jeden fold eine funktion erstellt damit man danach keinen fehler mehr machen kann
        get_fold_dataset = partial(ESC50, root=data_path, download=True,
                                   test_folds={test_fold}, global_mean_std=global_stats[test_fold - 1],
                                   prob_aug_wave=config.prob_aug_wave, prob_aug_spec=config.prob_aug_spec)
        # Datensatz, ist eine Instanz von ESC50 das die methode get_
        train_set = get_fold_dataset(subset="train", num_aug=config.num_aug)
        logger.info(f'Train folds are {train_set.train_folds} and test fold is {train_set.test_folds}.')
        logger_fold.info(f'Train folds are {train_set.train_folds} and test fold is {train_set.test_folds}.')

        # batch_size, num_workers, persistent_workers ändern um es schneller machen --> multiprocessing
        # prefetchfactor kann auch performance erhöhen
        logger.debug('Start DataLoader')
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.num_workers,
                                                   drop_last=False,
                                                   persistent_workers=config.persistent_workers,
                                                   pin_memory=True,
                                                   )

        val_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="val"),
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.num_workers,
                                                 drop_last=False,
                                                 persistent_workers=config.persistent_workers,
                                                 )
        logger.debug('Finish DataLoader')

        logger.debug('Start instantiate model')
        # instantiate model
        model = make_model()
        # model = nn.DataParallel(model, device_ids=config.device_ids)
        model = model.to(device)
        logger.debug('Finish instantiate model')
        #print('*****')

        # Define a loss function and optimizer
        criterion = nn.CrossEntropyLoss().to(device)

        """optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    momentum=0.9,
                                    weight_decay=config.weight_decay)
        # stattdessen adam ausprobieren

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.step_size,
                                                    gamma=config.gamma)
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.step_size,
                                                    gamma=config.gamma)

        # fit the model using only training and validation data, no testing data allowed here
        logger.debug('Start Training')
        fit_classifier()
        logger.debug('Finished Training')

        # tests
        test_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="test"),
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,  # config.num_workers,
                                                  drop_last=False,
                                                  )

        logger.info(f'\ntest {experiment}')
        test_acc, test_loss, _ = test(model, test_loader, criterion=criterion, device=device)
        scores[test_fold] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
        logger.info(scores[test_fold])
        # print(scores[test_fold].unstack())
        logger.info('')
    scores = pd.concat(scores).unstack([-1])
    logger.info(pd.concat((scores, scores.agg(['mean', 'std']))))

    return experiment_root


if __name__ == "__main__":
    main()
