import argparse
from RelTR import inference

def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', default=False, help='heeellllpppp')
    return parser

def main():
    exec(open('inference.py').read())

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Image Captioning', parents=[get_args_parser()])
    # args = parser.parse_args()
    main()