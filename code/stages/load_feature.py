import os.path as osp
import pickle
import pandas as pd
import os

import numpy as np
from pyturbo import Stage


class LoadFeature(Stage):

    """
    Input: video_id
    Output: features [N x D]
    """

    def allocate_resource(self, resources, *, feature_dir, file_suffix='pkl',
                          worker_per_cpu=1):
        self.feature_dir = feature_dir
        self.file_suffix = file_suffix
        return resources.split(len(resources.get('cpu'))) * worker_per_cpu

    @staticmethod
    def load_features(feature_path):
        features = []
        
        # check file extension
        _, file_extension = os.path.splitext(feature_path)

        # Pickle 
        if file_extension == '.pkl':
            with open(feature_path, 'rb') as f:
                while True:
                    try:
                        _, frame_feature = pickle.load(f)
                        features.append(frame_feature)
                    except EOFError:
                        break

        # CSV 
        elif file_extension == '.csv':
                try:
                    # UTF-8 encoding
                    frame_feature = pd.read_csv(feature_path, encoding='utf-8')
                    features.append(frame_feature)
                except UnicodeDecodeError:
                    # othere encoding
                    frame_feature = pd.read_csv(feature_path, encoding='ISO-8859-1')
                    features.append(frame_feature)

        return features

    def process(self, task):
        task.start(self)
        video_id = task.content
        feature_path = osp.join(
            self.feature_dir, f'{video_id}.{self.file_suffix}')
        features = self.load_features(feature_path)
        return task.finish(features)
