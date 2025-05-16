import shutil
import sys
import os
import datetime
from pathlib import Path
from functools import partial
import importlib
import logging
import random
import inspect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm import tqdm

import config
from dataset.dataset_ESC50 import ESC50, get_global_stats

logger = logging.getLogger(__name__)

def create_folder(folder_name):
    path_folder = os.path.join(config.folder_path, 'exploration')
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    path_folder = os.path.join(path_folder, folder_name)
    if os.path.exists(path_folder):
        shutil.rmtree(path_folder)
    os.makedirs(path_folder)
    return path_folder

def category_plot_example_spectrogram():
    """
    Plots per Class 20 Example Spectrogram
    """
    path_folder = create_folder(inspect.currentframe().f_code.co_name)
    path_data = config.esc50_path

    esc50_dataset = ESC50(root=path_data, download=True)
    all_classes = sorted(list(set(esc50_dataset.get_metadata(i)['category'] for i in range(len(esc50_dataset)))))

    # Plot for every class 20 examples
    for class_name in all_classes:
        # All Indices of class
        class_indices = [i for i in range(len(esc50_dataset))
                         if esc50_dataset.get_metadata(i)['category'] == class_name]

        # Choose randomly 20
        selected_indices = random.sample(class_indices, min(20, len(class_indices)))

        # Plot
        plt.figure(figsize=(15, 2 * 10))
        for i, idx in enumerate(selected_indices):
            file_name, feat, class_id = esc50_dataset[idx]
            feat = feat.squeeze(0)
            plt.subplot(10, 2, i + 1)
            plt.imshow(feat.numpy(), aspect='auto', origin='lower', cmap='jet')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{file_name} + {feat.shape}")
            plt.xlabel('Time')
            plt.ylabel('Freq')

        plt.tight_layout()
        plt.savefig(os.path.join(path_folder, f"{class_name.replace('/', '_')}.png"))
        plt.close()

def plot_augmentation():
    path_folder = create_folder(inspect.currentframe().f_code.co_name)
    data_path = config.esc50_path

    esc50_dataset = ESC50(root=data_path, download=True)
    all_classes = sorted(list(set(esc50_dataset.get_metadata(i)['category'] for i in range(len(esc50_dataset)))))

def print_global_stats():
    from dataset.dataset_ESC50 import get_global_stats
    import csv

    path_folder = create_folder(inspect.currentframe().f_code.co_name)
    path_data = config.esc50_path
    global_stats = get_global_stats(path_data)

    res = []
    all_data = []
    plt.figure(figsize=(8, 5))
    colors = ['blue', 'red', 'green', 'cyan', 'magenta']
    labels = [f'Fold {i}' for i in range(1, 6)]
    for i in range(1, 6):
        train_set = ESC50(subset="train", test_folds={i}, root=path_data, download=True)
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        #a_flatten = np.array([val for val in a.flatten() if val != 160])
        #res.append((a_flatten.mean(), a_flatten.std()))
        res.append((float(a.mean()), float(a.std())))
        all_data.append(a.flatten().numpy())

    # Gemeinsame Bins berechnen
    data_min = min(d.min() for d in all_data)
    data_max = max(d.max() for d in all_data)
    bins = np.linspace(data_min, data_max, 51)  # 50 bins = 51 edges
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    num_folds = len(all_data)

    plt.figure(figsize=(12, 6))

    # Balken nebeneinander einzeichnen
    for i, (data, color, label) in enumerate(zip(all_data, colors, labels)):
        counts, _ = np.histogram(data, bins=bins)
        # Versatz pro Fold für nebeneinander-Anzeige
        offset = (i - num_folds / 2) * (bin_width / num_folds)
        plt.bar(bin_centers + offset, counts, width=bin_width / num_folds, color=color, label=label, alpha=0.8, align='center', edgecolor='black')

    plt.title('Histogram per Fold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path_folder, 'histogram_grouped.png'))
    plt.show()
    plt.close()

    with open(os.path.join(path_folder, 'global_stats.csv'), 'w') as f:
        writer = csv.writer(f)
        for mean, std in res:
            writer.writerow([round(mean,6), round(std,6)])


if __name__ == "__main__":
    #category_plot_example_spectrogram()
    #plot_augmentation()
    print_global_stats()