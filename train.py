from docker.docker import Docker
from docker.config import Configuration, base_args
import argparse

def get_args():
    parser = base_args()
    return parser

if __name__ == "__main__":
    args = get_args()
    cfg = Configuration(args, 'train')
    doc = Docker(cfg)
    doc.train()


