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
    # exec(open('inference.py').read())
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    
    with open('datasets/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # vocab = load_vocab.vocab('datasets/vocab.pkl')

    data = getData(
        'datasets/images/resized2014/',
        'datasets/annotations/captions_train2014.json',
        'datasets/SG/Detected_Scene_Graph.json',
        vocab,
        transform, 128, False, 2)

    # caption [128, 24] --> [1, 23, 17, ..., 2]
    for i,(image, caption, length, sg) in enumerate(data):
        print(sg[127])
        print(image.shape)
        print(caption.shape)
        break

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Image Captioning', parents=[get_args_parser()])
    # args = parser.parse_args()
    main()