#Plot results.csv metrics file generated by train.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plotting function
def plot_results(save_dir='runs/detect/train6', half=False, save_txt=False):
    os.makedirs(save_dir, exist_ok=True)
    results = pd.read_csv(f'{save_dir}/results.csv')  # load results.csv
    results = results.drop_duplicates()  # drop duplicate rows
    results = results[results.epoch > 0]  # skip first row
    results = results.sort_values('epoch')  # sort by epoch

    # Results
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    x = results['epoch']
    y = results['GIoU']
    ax[0, 0].plot(x, y, '.-', label='GIoU', color='tab:orange', lw=2)
    ax[0, 0].set_title('GIoU loss', fontsize=14)

    y = results['Objectness']
    ax[0, 1].plot(x, y, '.-', label='Objectness', color='tab:orange', lw=2)
    ax[0, 1].set_title('Objectness loss', fontsize=14)

    y = results['Classification']
    ax[1, 0].plot(x, y, '.-', label='Classification', color='tab:orange', lw=2)
    ax[1, 0].set_title('Classification loss', fontsize=14)

    y = results['Train_Accuracy']
    ax[1, 1].plot(x, y, '.-', label='Train Acc', color='tab:orange', lw=2)
    ax[1, 1].set_title('Train accuracy', fontsize=14)

    for i in range(2):
        for j in range(2):
            ax[i, j].grid()
            ax[i, j].set_xlabel('Epoch', fontsize=10)
            ax[i, j].legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results.png', dpi=300)

    # Save txt
    if save_txt:
        df = results.copy()
        df['Region Avg IOU'] = results['Region Avg IOU'].map('{:.5f}'.format)
        df['Class'] = results
        df['Obj'] = results
        df['No Obj'] = results
        df['Avg Recall'] = results
        df['count'] = results
        df['LR'] = results
        df.to_csv(f'{save_dir}/results.txt', index=False)
        