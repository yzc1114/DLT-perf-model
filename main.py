import argparse
import logging
import pathlib

from config import Config
from executor import get_executor_cls
from logger import init_logging

init_logging()

config_filename = "test.json"

def get_config() -> Config:
    config_path = pathlib.Path(__file__).parent / "configs" / config_filename
    return Config.from_file(str(config_path))


def main():
    c = get_config()
    executor = get_executor_cls(c)
    executor.run()


if __name__ == '__main__':
    main()
