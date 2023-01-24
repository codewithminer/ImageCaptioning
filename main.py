import argparse
from utils.dataloader import getData
from torchvision import transforms
from utils.build_vocab import Vocabulary # import this, so you can load vocab.pkl file
import pickle
# from RelTR import inference


def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', default=False, help='heeellllpppp')
    return parser

def main():
    pass

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Image Captioning', parents=[get_args_parser()])
    # args = parser.parse_args()
    main()