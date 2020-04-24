from docker.docker import Docker
from docker.config import Configuration,base_args
import argparse

def get_args():
    parser = base_args()
    parser.add_argument("-visual", action='store_true',
                        help='if you want to visualize your result, choose it.')
    parser.add_argument("-save", action='store_true',
                        help='if you want to save your result, choose it.')
    return parser

if __name__ == "__main__":
    args = get_args()
    cfg = Configuration(args,'test')
    doc = Docker(cfg)
    mean = doc.test(visualize=args.visual,save_result=args.save)

