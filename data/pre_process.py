import argparse
import os.path
import pandas as pd

dataset_path = '/root/guohao/DLT-perf-model/datasets'
configs_path = '/root/guohao/DLT-perf-model/configs'

def delete_unvalid_data(data_set : str):
    """
    删除无效数据
    """
    train_data= os.path.join(dataset_path, data_set, 'train')
    for file in os.listdir(train_data):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(train_data, file))
        if (df['space']<-1000).any():
            print(os.path.join(train_data, file))
            os.remove(os.path.join(train_data, file))
    eval_data = os.path.join(dataset_path, data_set, 'eval')
    for file in os.listdir(eval_data):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(eval_data, file))
        if (df['space']<0).any():
            print(os.path.join(eval_data, file))
            os.remove(os.path.join(eval_data, file))
    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_set', type=str, default='RTX2080Ti_CPU100')
    parser.add_argument('--data_set', type=str, default='T4_CPU100')

    args = parser.parse_args()

    delete_unvalid_data('T4_CPU100')
    delete_unvalid_data('RTX2080Ti_CPU100')
    delete_unvalid_data('TEST_CPU100')