import argparse
import torch
import numpy as np
import torch.nn as nn
import os
import pickle
from utils.dataloader import getData
from torchvision import transforms
from utils.build_vocab import Vocabulary # import this, so you can load vocab.pkl file
from utils.sg_dataloader import getSGData
from models.models import EncoderDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from models.Bert import Word2Vector
from utils.dataholder import DataHolder
from utils.CustomLoss import BertDistancesLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    #create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained Resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
 
    # Load Vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Loading Data (images, captions, captions length, scene graph)
    data = getData(
        args.image_dir, args.caption_path, args.sg_path,
        vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)


    gcn_in_channels = 768
    gcn_hidden_channels = 1024
    gcn_out_channels = 256
    gcn_pooling_type = 'mean'
    gcn_edge_in_channels = 768

    resnet_embed_size = 256

    encoder_lstm_embed_size = 256
    encoder_lstm_hidden_size = 512
    encoder_lstm_num_layers = 1

    decoder_lstm_embed_size = 256
    decoder_lstm_hidden_size = 512
    decoder_lstm_vocab_size = len(vocab)
    decoder_lstm_num_layers = 1
    max_seq_length = 20

    model = EncoderDecoder(
                gcn_in_channels, gcn_hidden_channels, gcn_out_channels, gcn_pooling_type, gcn_edge_in_channels,
                resnet_embed_size,
                encoder_lstm_embed_size, encoder_lstm_hidden_size, encoder_lstm_num_layers,
                decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size, decoder_lstm_num_layers, max_seq_length
            ).to(device)
    bert = Word2Vector()
    word2vec = DataHolder().word2vec

    # criterion = nn.CrossEntropyLoss()
    bertLoss = BertDistancesLoss(vocab)
    # params = list(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_step = len(data)
    for epoch in range(args.num_epochs):
        # caption [128, 24] --> [1, 23, 17, ..., 2]
        for i,(images, captions, lengths, SGs) in enumerate(data):
            SG_data =  getSGData(SGs,bert,word2vec)
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)  # with [0] to get the concatenated and padded-removed all of the word in captions.
            outputs =  model(SG_data, images, captions, lengths)
            # loss = criterion(outputs, targets[0])
            loss = bertLoss.loss(outputs, targets, lengths)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'encoder-decoder-{}-{}.ckpt'.format(epoch+1, i+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ckpt/', help='path for saving trained model')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='datasets/images/resized2014/', help='direcotory for resized images')
    parser.add_argument('--caption_path', type=str, default='datasets/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--sg_path', type=str, default='datasets/SG/Detected_Scene_Graph.json', help='path for train scene graph json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
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
