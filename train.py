import argparse
import torch
import torch.nn as nn
import os
from torchvision import transforms
import pickle
from utils.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    #create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrappers
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = DataLoader()

    # Build models
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ckpt/', help='path for saving trained model')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='datasets/images/resized2014', help='direcotory for resized images')
    parser.add_argument('--caption_path', type=str, default='datasets/annotations/caption_train2014.json', help='path for train annotation json file')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # LSTM model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimention of word embedding vectors')
    parser.add_argument('--lstm_hidden_size', type=int, default=512, help='dimention of LSTM hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    main(args)
