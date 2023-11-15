import os.path as osp
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from stages import LoadFeature
from torch.utils.data import DataLoader, Dataset


class FeatureEarlyFusionDataset(Dataset):

    def __init__(self, df, feature_dir1, feature_dir2):
        self.df = df
        self.feature_dir1 = feature_dir1
        self.feature_dir2 = feature_dir2

    def __len__(self):
        return len(self.df)

    def aggregate_frame_features(
            self, frame_features: np.ndarray) -> np.ndarray:
        """
        frame_features: [N x D]

        Return: [D]
        """
        # max pooling:
        aggregated_feature = np.max(frame_features, axis=0)
   
        return aggregated_feature


    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        vid = item['Id']
        label = item.get('Category', None)
        
        feature_path1 = osp.join(self.feature_dir1, f'{vid}.pkl')
        frame_features1 = np.stack(LoadFeature.load_features(feature_path1))
        feature1 = self.aggregate_frame_features(frame_features1)
        feature1 = torch.as_tensor(feature1, dtype=torch.float).squeeze()

        feature_path2 = osp.join(self.feature_dir2, f'{vid}.csv')
        frame_features2 = np.stack(LoadFeature.load_features(feature_path2))
        feature2 = self.aggregate_frame_features(frame_features2)
        feature2 = torch.as_tensor(feature2, dtype=torch.float).squeeze()

        feature = torch.cat((feature1, feature2), dim=0)
        return feature, label


class FeatureEarlyFusionDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super(FeatureEarlyFusionDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        train_val_df = pd.read_csv(self.hparams.train_val_list_file)
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=self.hparams.test_frac,
            random_state=self.hparams.split_seed)
        self.test_df = pd.read_csv(self.hparams.test_list_file)
        self.feature_dir1 = self.hparams.feature_dir1
        self.feature_dir2 = self.hparams.feature_dir2
        self.batch_size = self.hparams.batch_size

    def setup(self, stage=None):
        self.train_set = FeatureEarlyFusionDataset(self.train_df, self.feature_dir1, self.feature_dir2)
        self.val_set = FeatureEarlyFusionDataset(self.val_df, self.feature_dir1, self.feature_dir2)
        self.test_set = FeatureEarlyFusionDataset(self.test_df, self.feature_dir1, self.feature_dir2)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True, pin_memory=True,
                          num_workers=len(psutil.Process().cpu_affinity()))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=len(psutil.Process().cpu_affinity()))

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=len(psutil.Process().cpu_affinity()))

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--train_val_list_file', default=osp.abspath(osp.join(
                osp.dirname(__file__), '../../data/labels/train_val.csv')))
        parser.add_argument(
            '--test_list_file', default=osp.abspath(osp.join(osp.dirname(
                __file__), '../../data/labels/test_for_students.csv')))
        parser.add_argument('--feature_dir1')
        parser.add_argument('--feature_dir2')
        # parser.add_argument('--when', type=str, default="early")
        parser.add_argument('--test_frac', type=float, default=0.2)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--split_seed', type=int, default=666)
        return parser
