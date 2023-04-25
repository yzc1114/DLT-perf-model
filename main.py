import argparse
import pathlib

from executor import Coordinator
from config import Config

parser = argparse.ArgumentParser(description='DLT Perf Predicting')

parser.add_argument('--mode',
                    choices=['train', 'eval'],
                    default='train', help='Select mode to run (default: train)')
parser.add_argument('--config-file', type=str, required=True, help='Path to config file')

args = parser.parse_args()

print(f"Mode: {args.mode}")
print(f"Config file: {args.config_file}")

def get_config() -> Config:
    config_dir = pathlib.Path(__file__).parent / "configs" / args.mode
    config_path = config_dir / args.config_file
    return Config(str(config_path))

def main():
    c = get_config()
    if c.train_configs is not None:
        for train_config in c.train_configs:
            Coordinator.train(train_config)
    if c.eval_configs is not None:
        for eval_config in c.eval_configs:
            Coordinator.eval(eval_config)

if __name__ == '__main__':
    main()
