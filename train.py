import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from utils.dataloader import getData
from utils.build_vocab import Vocabulary # import this, Necessary for load vocab.pkl file
from utils.sg_dataloader import getSGData
from utils.dataholder import DataHolder
from utils.CustomLoss import BertDistancesLoss
from models.models import EncoderDecoder
from models.Bert import Bert

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

    # validation_data = getData(
    # args.val_image_dir, args.val_caption_path, args.val_sg_path,
    # vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)


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

    best_val_loss = float("inf")
    early_stopping_counter = 0

    model = EncoderDecoder(
                gcn_in_channels, gcn_hidden_channels, gcn_out_channels, gcn_pooling_type, gcn_edge_in_channels,
                resnet_embed_size,
                encoder_lstm_embed_size, encoder_lstm_hidden_size, encoder_lstm_num_layers,
                decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size, decoder_lstm_num_layers, max_seq_length
            ).to(device)
    bert = Bert()
    word2vec = DataHolder().word2vec

    criterion = nn.CrossEntropyLoss()
    # params = list(model.parameters())
    # bertLoss = BertDistancesLoss(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_step = len(data)
    for epoch in range(args.num_epochs):
        # if early_stopping_counter >= args.early_stopping_patience:
        #     break
        # caption [128, 24] --> [1, 23, 17, ..., 2]
        for i,(images, captions, lengths, SGs) in enumerate(data):
            SG_data =  getSGData(SGs,bert,word2vec)
            images = images.to(device)
            captions = captions.to(device)
            outputs =  model(SG_data, images, captions, lengths)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)  # with [0] to get the concatenated and padded-removed all of the word in captions.
            loss = criterion(outputs, targets[0])
            # loss = bertLoss.loss(outputs, targets, lengths)
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
                    args.model_path, 'encoder-decoder-{}-{}.pth'.format(epoch+1, i+1)))


            # evaluate the model on the validation set and save the best model weights
            # if (i+1) % args.save_step == 0:
            #     val_loss = evaluate_model_on_validation_set(model, validation_data, loss, bert, word2vec)
            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         early_stopping_counter = 0
            #         torch.save(model.state_dict(), os.path.join(args.model_path, 'encoder-decoder-{}-{}.pth'.format(epoch+1, i+1)))
            #     else:
            #         early_stopping_counter += 1
            #     if early_stopping_counter >= args.early_stopping_patience:
            #         print("Early stopping triggered at epoch {}".format(epoch))
            #         break

def evaluate_model_on_validation_set(model, validation_data, criterion, bert, word2vec):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, captions, lengths, SGs in validation_data:
            SG_data = getSGData(SGs, bert, word2vec)
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)
            outputs = model(SG_data, images, captions, lengths)
            loss = criterion(outputs, targets, lengths)
            val_loss += loss.item()
        val_loss /= len(validation_data)
    return val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='ckpt/', help='path for saving trained model')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='datasets/images/resized2014/', help='direcotory for resized images')
    parser.add_argument('--caption_path', type=str, default='datasets/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--sg_path', type=str, default='datasets/SG/Detected_Scene_Graph.json', help='path for train scene graph json file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='if the validation loss has not improved for 10 consecutive epochs, early stopping will be triggered.')

    # Validation set parameters
    parser.add_argument('--val_image_dir', type=str, default='datasets/images/val_resized2014/', help='direcotory for validation resized images')
    parser.add_argument('--val_caption_path', type=str, default='datasets/annotations/captions_val2014.json', help='path for validation annotation json file')
    parser.add_argument('--val_sg_path', type=str, default='datasets/SG/Val_Detected_Scene_Graph.json', help='path for validation scene graph json file')

    # GCN model parameters
    parser.add_argument('--gcn_in_channels', type=int, default=768, help='dimention of node vectors')
    parser.add_argument('--gcn_hidden_channels', type=int, default=1024, help='dimention of GCN hidden layer')
    parser.add_argument('--gcn_out_channels', type=int, default=256, help='dimention of GCN out layer')
    parser.add_argument('--gcn_edge_in_channels', type=int, default=768, help='dimention of edge vectors')
    parser.add_argument('--gcn_pooling_type', type=str, default='mean', help='type of pooling layer')

    # Resnet model parameters
    parser.add_argument('--resnet_embed_size', type=int, default=256, help='dimention of image embedding vectors')

    # Encoder-LSTM model parameters
    parser.add_argument('--encoder_lstm_embed_size', type=int, default=256, help='dimention of gcn and resnet embedding vectors')
    parser.add_argument('--encoder_lstm_hidden_size', type=int, default=512, help='dimention of Encoder-LSTM hidden states')
    parser.add_argument('--encoder_lstm_num_layers', type=int, default=1, help='number of layers in Encoder-LSTM')

    # Decoder-LSTM model parameters
    parser.add_argument('--decoder_lstm_embed_size', type=int, default=256, help='dimention of encoder-lstm embedding vectors')
    parser.add_argument('--decoder_lstm_hidden_size', type=int, default=512, help='dimention of Decoder-LSTM hidden states')
    parser.add_argument('--decoder_lstm_num_layers', type=int, default=1, help='number of layers in Decoder-LSTM')
    parser.add_argument('--max_seq_length', type=int, default=20, help='maximum lenght of sequences')

    # Training loop parametrs
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    main(args)
