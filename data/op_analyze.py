import argparse
import os.path
import pandas as pd

dataset_path = '/root/guohao/DLT-perf-model/datasets'
configs_path = '/root/guohao/DLT-perf-model/configs'

def get_frequency(data_set):
    """
    统计每个op出现的频率
    """
    d = {}
    train_data= os.path.join(dataset_path, data_set)
    for dir_path, dir_names, files in os.walk(train_data):
        for file in files:
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(dir_path, file))
            tmp = df['op'].value_counts()
            for op in tmp.index:
                if op in d:
                    d[op] += tmp[op]
                else:
                    d[op] = tmp[op]
    sum = 0
    for op in d:
        sum += d[op]
    for op in d:
        d[op] /= sum
    print(len(d))
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(configs_path,  data_set+'_op_frequency.dict'), 'w') as f:
        f.write(str(d))
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_set', type=str, default='RTX2080Ti_CPU100')
    parser.add_argument('--data_set', type=str, default='T4_CPU100')

    args = parser.parse_args()

    op_frequency = get_frequency(args.data_set)