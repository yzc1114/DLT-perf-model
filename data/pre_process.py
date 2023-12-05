import argparse
import os.path
import pathlib

import pandas as pd

dataset_path = str(pathlib.Path(os.getcwd()) / ".." / "datasets")
config_path = str(pathlib.Path(os.getcwd()) / ".." / "configs" / "models")


def split_data(data_set: str):
    """
    将models划分为训练集和测试机
    """
    data_set_path = os.path.join(dataset_path, data_set)
    model_path = os.path.join(data_set_path, 'models')
    train_path = os.path.join(data_set_path, 'train')
    eval_path = os.path.join(data_set_path, 'eval')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    with open(os.path.join(config_path, 'eval.txt'), 'r') as f:
        eval_models = f.readlines()
        for model in eval_models:
            model = model.rstrip('\n')
            os.system('mv ' + os.path.join(model_path, model) + '.*' + ' ' + eval_path)
    with open(os.path.join(config_path, 'train.txt'), 'r') as f:
        train_models = f.readlines()
        for model in train_models:
            model = model.rstrip('\n')
            os.system('mv ' + os.path.join(model_path, model) + '.*' + ' ' + train_path)
    return


def delete_unvalid_data(data_set: str):
    """
    删除无效数据,即space小于-1000
    """
    train_data = os.path.join(dataset_path, data_set, 'train')
    for file in os.listdir(train_data):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(train_data, file))
        if (df['space'] < -1000).any():
            print(os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
    eval_data = os.path.join(dataset_path, data_set, 'eval')
    for file in os.listdir(eval_data):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(eval_data, file))
        if (df['space'] < 0).any():
            print(os.path.join(eval_data, file))
            os.remove(os.path.join(eval_data, file))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_set', type=str, default='RTX2080Ti_CPU100')
    parser.add_argument('--data_set', type=str, default='T4_CPU100')

    args = parser.parse_args()

    # delete_unvalid_data('T4_CPU100')
    # delete_unvalid_data('RTX2080Ti_CPU100')
    # delete_unvalid_data('TEST_CPU100')

    split_data('T4_CPU100')
