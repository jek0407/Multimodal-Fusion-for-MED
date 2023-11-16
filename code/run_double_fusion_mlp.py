import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy

from modules import FeatureEarlyFusionDataModule, FeatureDataModule, MlpClassifier


def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('name')
    parser = FeatureDataModule.add_argparse_args(parser)
    parser = MlpClassifier.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(accelerator='gpu', devices=1,
                        default_root_dir=osp.abspath(
                            osp.join(osp.dirname(__file__), '../data/mlp')))
    args = parser.parse_args(argv)
    return args

# late fusion for classification
def main(args):
    # 1. Load two different unimodal feature descriptors
    # data_module = feature, label
    # Set different feature_dir for each data module
    args_1 = deepcopy(args)
    args_2 = deepcopy(args)
    # Set different feature_dir for each data module
    args_1.feature_dir = 'data/cnn3d'
    args_1.num_features = 512
    args_2.feature_dir1 = 'data/cnn3d'
    args_2.feature_dir2 = 'data/snf'
    args_2.num_features = 767
    # Initialize data modules with different directories
    data_module_1 = FeatureDataModule(args_1)
    data_module_2 = FeatureEarlyFusionDataModule(args_2)

    print("\n\n########## First features. ############")
    model_1 = MlpClassifier(args_1)
    logger = TensorBoardLogger(args.default_root_dir, args.name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)
    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model_1, data_module_1)
    predictions_1 = trainer.predict(datamodule=data_module_1, ckpt_path='best')

    
    print("\n\n########## Second features. ############")
    model_2 = MlpClassifier(args_2)
    logger = TensorBoardLogger(args.default_root_dir, args.name)
    checkpoint_callback_2 = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)
    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,
        callbacks=[checkpoint_callback_2, early_stop_callback])
    
    trainer.fit(model_2, data_module_2)
    predictions_2 = trainer.predict(datamodule=data_module_2, ckpt_path='best')

    # 3. prediction scores from different models then combined to yield a final score.
    if len(predictions_1) != len(predictions_2):
        raise ValueError("The lists of predictions have different lengths")
    # Add corresponding tensors element-wise
    combined_predictions = [p1*0.5 + p2*0.5 for p1, p2 in zip(predictions_1, predictions_2)]
    predictions = [tensor.round().int() for tensor in combined_predictions]
    
    # code for cnn3d test prediction
    # print(f"predictions_1 :\n{predictions_1}")
    # df = data_module_1.test_df.copy()
    # df['Category'] = torch.concat(predictions_1).numpy()
    # prediction_path = osp.join(logger.log_dir, 'test_prediction.csv')
    # df.to_csv(prediction_path, index=False)
    # print('Output file:', prediction_path)    
    
    # print(f"predictions :\n{predictions}")
    df = data_module_1.test_df.copy()
    df['Category'] = torch.concat(predictions).numpy()
    prediction_path = osp.join(logger.log_dir, 'double_fusion_test_prediction.csv')
    df.to_csv(prediction_path, index=False)
    print('Output file:', prediction_path)


if __name__ == '__main__':
    main(parse_args())
