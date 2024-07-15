# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:10:18 2024

@author: Mico
"""

import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gdown
import zipfile


def download_dataset(dataset_dir):
    dataset_zip_path = dataset_dir + '.zip'
    url = 'https://drive.google.com/file/d/1eBYiFFiuZ3U5q51ebuP4QWOlP4_nTain/view?usp=sharing'
    gdown.download(url, dataset_zip_path, fuzzy=True)


def unzip_dataset(dataset_dir):
    dataset_zip_path = dataset_dir + '.zip'
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)


def create_dataframe(dataset_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    train_clean = os.path.join(dataset_dir, 'train_cleaned')

    df = {}
    df['image'] = [os.path.join('train', fn) for fn in os.listdir(train_dir)]
    df['target'] = [os.path.join('train_cleaned', fn) for fn in os.listdir(train_clean)]

    df['h'] = []
    df['w'] = []
    for img_fn in tqdm(df['image']):
        im_path = os.path.join(dataset_dir, img_fn)
        im = io.imread(im_path, as_gray=True)
        h, w = im.shape[0:2]
        df['h'].append(h)
        df['w'].append(w)
    df = pd.DataFrame(df)
    df['aspect'] = df['w'] / df['h']

    return df


def split_dataset(df, test_size, stratify_by='aspect'):
    train_idx, val_idx = train_test_split(df.index.values,
                                          test_size=test_size,
                                          stratify=df[stratify_by].values)
    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'test'
    return df


if __name__ == "__main__":
    project_dir = '..'
    data_dir = os.path.join(project_dir, 'data')
    dataset_dir = os.path.join(data_dir, 'denoising-dirty-documents')
    df_path = os.path.join(data_dir, 'train.csv')

    show_figures = True
    test_size = 0.05

    if os.path.exists(dataset_dir):
        print('dataset has already downloaded')
    else:
        download_dataset(dataset_dir)
        unzip_dataset(dataset_dir)

    df = create_dataframe(dataset_dir)
    df = split_dataset(df, test_size, stratify_by='aspect')  # split stratified by aspect ratio
    df.to_csv(df_path, index=False)

    if show_figures:
        plt.scatter(df['h'], df['w'])
        plt.show()

        df['aspect'].hist()
        plt.show()

        print(df.describe())
        print(df.groupby('h').count())

        df_examples = df.groupby('h').first().reset_index()

        for index, row in df_examples.iterrows():
            im_path = os.path.join(dataset_dir, row['image'])
            im_clean_path = os.path.join(dataset_dir, row['target'])
            im = io.imread(im_path, as_gray=True)
            im_clean = io.imread(im_clean_path, as_gray=True)

            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            ax[0].imshow(im, cmap='gray')
            ax[0].axis('off')
            ax[1].imshow(im_clean, cmap='gray')
            ax[1].axis('off')
            plt.show()
