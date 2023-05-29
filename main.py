import argparse
import pathlib

from config import Config
from executor import init_executor
from logger import init_logging
import logging
init_logging()

parser = argparse.ArgumentParser(description='DLT Perf Predicting')

parser.add_argument('--mode',
                    choices=['train', 'eval'],
                    default='train', help='Select mode to run (default: train)')
parser.add_argument('--config-file', type=str, required=True, help='Path to config file')

args = parser.parse_args()

logging.info(f"Mode: {args.mode}")
logging.info(f"Config file: {args.config_file}")


def get_config() -> Config:
    config_dir = pathlib.Path(__file__).parent / "configs" / args.mode
    config_path = config_dir / args.config_file
    return Config(str(config_path))


def main():
    c = get_config()
    if c.train_configs is not None:
        for train_config in c.train_configs:
            executor = init_executor(train_config)
            executor.train()
    if c.eval_configs is not None:
        for eval_config in c.eval_configs:
            executor = init_executor(eval_config)
            executor.evaluate()


if __name__ == '__main__':
    main()
