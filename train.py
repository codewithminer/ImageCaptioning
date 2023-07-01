import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from utils.dataloader import getData
from utils.build_vocab import Vocabulary, GraphVocabulary # import this, Necessary for load vocab.pkl file
from utils.scene_graph_extractor import extractSceneGraphData
from models.model2 import EncoderDecoder
from utils.tools import save_model
from nltk.translate.bleu_score import corpus_bleu



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alpha_c = 1.

def main(args):
    #create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained Resnet
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
 
    # Load Vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load Graph Vocabulary wrapper
    with open(args.vocab_graph_path, 'rb') as f:
        vocab_graph = pickle.load(f)

    # Loading training data (images, captions, captions length, scene graphs)
    print('Loading training data...')
    training_data = getData(
        args.train_image_dir, args.train_caption_path, args.train_sg_path,
        vocab, True, transform, batch_size=args.batch_size,  shuffle=True, num_workers=args.num_workers)

    # Loading validation data (images, captions, captions length, scene graphs, all_captions)
    print('Loading validation data...')
    validation_data = getData(
        args.val_image_dir, args.val_caption_path, args.val_sg_path,
        vocab, False, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    gcn_in_channels = 256
    gcn_hidden_channels = 512
    gcn_edge_in_channels = 256

    geo_gcn_in_channels = 4
    geo_gcn_hidden_channels = 128
    geo_edge_in_channels = 3

    resnet_embed_size = 2048

    decoder_lstm_embed_size = 512
    decoder_lstm_hidden_size = 512
    decoder_lstm_vocab_size = len(vocab)
    max_seq_length = 20
    
    start_epoch = 0
    load_ckpt = args.load_ckpt
    best_score = 0
    early_stopping = 0

    if (load_ckpt is None):
        model = EncoderDecoder(
            gcn_in_channels, gcn_hidden_channels, gcn_edge_in_channels,
            geo_gcn_in_channels, geo_gcn_hidden_channels, geo_edge_in_channels,
            resnet_embed_size,
            decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size,  max_seq_length
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    else:
        print('Loading pre-trained model {} ...'.format(args.load_ckpt))
        checkpoint = torch.load(args.load_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        early_stopping = checkpoint['stopping_counter']
        model = checkpoint['model']
        model.to(device)
        optimizer = checkpoint['optimizer']
        # best_score = checkpoint['bleu-4']

    criterion = nn.CrossEntropyLoss()

    total_step = len(training_data)
    total_val_step = len(validation_data)

    for epoch in range(start_epoch, args.num_epochs):
        # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 10
        if early_stopping > args.early_stopping_patience:
            break
        if early_stopping > 0 and early_stopping % 8 == 0:
            print("\nDECAYING learning rate.")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
            print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
            
        training(
            training_data, model, criterion, 
            optimizer, epoch, total_step,
            vocab_graph
        )

        score = validation(validation_data, model, criterion, epoch, total_val_step,vocab, vocab_graph)
        print('Epoch [{}], Bleu-4 score [{}]'.format(epoch+1, score))
        is_best = score > best_score
        best_score = max(score, best_score)
        if not is_best:
            early_stopping += 1
            print("\nEpochs since last improvement: %d\n" % (early_stopping,))
        else:
            early_stopping = 0

        save_model(
            model, epoch, optimizer, early_stopping, 
            args.model_path, 'SSRAT-T7-{}.pth'.format(epoch+1), score)

def training(training_data, model, criterion, optimizer, epoch, total_step, vocab_graph):

    model.train()
    total_loss = 0
    for i,(images, captions, lengths, SG, _, _) in enumerate(training_data):

        # Extract Scene Graphs features
        scene_graph = extractSceneGraphData(SG, vocab_graph)

        images = images.to(device)
        captions = captions.to(device)

        outputs =  model(scene_graph, images, captions, lengths)
        captions = captions[:, 1:]
        lengths = [num - 1 for num in lengths]
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]  # with [0] to get the concatenated and padded-removed all of the word in captions.
        
        loss = criterion(outputs[0], targets)

        # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % args.log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch+1, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
        if i % 1000 == 0:
            print('Total Loss: {:.4f}'.format(total_loss/1000))
            total_loss = 0            
    
def validation(validation_data, model, criterion, epoch, total_val_step, vocab, vocab_graph):

    model.eval()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        for i, (images, captions, lengths, SG, allcaptions, _) in enumerate(validation_data):
            # Extract Scene Graphs features
            scene_graph = extractSceneGraphData(SG, vocab_graph)

            images = images.to(device)
            captions = captions.to(device)

            outputs =  model(scene_graph, images, captions, lengths)
            captions = captions[:, 1:]
            lengths = [num - 1 for num in lengths]
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]  # with [0] to get the concatenated and padded-removed all of the word in captions.
            
            # output_copy = outputs.clone()
            # print(outputs)
            padded_output = pad_packed_sequence(outputs, batch_first=True)

            loss = criterion(outputs[0], targets)

            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch+1, args.num_epochs, i, total_val_step, loss.item(), np.exp(loss.item()))) 

            # References
            allcaptions = list(allcaptions)
            for j in range(len(allcaptions)):
                img_caps = allcaptions[j]
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {vocab('<start>'), vocab('<end>'), vocab('<pad>')}],
                        img_caps))  # remove <start>, <end> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(padded_output[0], dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        return bleu4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--model_path', type=str, default='ckpt/', help='path for saving trained model')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_graph_path', type=str, default='datasets/vocab_graph.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='if the validation loss has not improved for 10 consecutive epochs, early stopping will be triggered.')
    parser.add_argument('--load_ckpt', type=str, default='ckpt/SSRAT-T7-1.pth', help='load pre-trained model')


    # Training set parameters
    parser.add_argument('--train_caption_path', type=str, default='datasets/annotations/filtered_annotations.json', help='path for train annotation json file')
    parser.add_argument('--train_image_dir', type=str, default='datasets/images/combined_images/', help='direcotory for resized images')
    parser.add_argument('--train_sg_path', type=str, default='datasets/SG/combined_Detected_Scene_Graphs.json', help='path for train scene graph json file')
    parser.add_argument('--training', type=bool, default=True, help="don't take all captions")

    # Validation set parameters
    parser.add_argument('--val_image_dir', type=str, default='datasets/images/val5000/images/', help='direcotory for validation resized images')
    parser.add_argument('--val_caption_path', type=str, default='datasets/annotations/captions_val50002014.json', help='path for validation annotation json file')
    parser.add_argument('--val_sg_path', type=str, default='datasets/SG/Val5000_Detected_Scene_Graphs.json', help='path for validation scene graph json file')
    parser.add_argument('--validation', type=bool, default=False, help="take all captions")

    # Semantic model parameters
    parser.add_argument('--sem_in_channels', type=int, default=256, help='dimention of node vectors')
    parser.add_argument('--sem_edge_in_channels', type=int, default=256, help='dimention of edge vectors')
    parser.add_argument('--sem_hidden_channels', type=int, default=512, help='dimention of GCN hidden layer')
    parser.add_argument('--sem_out_channels', type=int, default=256, help='dimention of GCN out layer')
    parser.add_argument('--sem_pooling_type', type=str, default='mean', help='type of pooling layer')

    # Resnet model parameters
    parser.add_argument('--resnet_embed_size', type=int, default=2048, help='dimention of image embedding vectors')

    # Geometric  model parameters
    parser.add_argument('--geo_in_channels', type=int, default=4, help='dimention of node vectors')
    parser.add_argument('--geo_edge_in_channels', type=int, default=3, help='dimention of edge vectors')
    parser.add_argument('--geo_hidden_channels', type=int, default=128, help='dimention of GCN hidden layer')

    # Decoder model parameters
    parser.add_argument('--decoder_lstm_embed_size', type=int, default=512, help='dimention of encoder-lstm embedding vectors')
    parser.add_argument('--decoder_lstm__hidden_size', type=int, default=512, help='dimention of Decoder-LSTM hidden states')
    parser.add_argument('--decoder_lstm_num_layers', type=int, default=1, help='number of layers in Decoder-LSTM')
    parser.add_argument('--max_seq_length', type=int, default=20, help='maximum lenght of sequences')

    # Training loop parametrs
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    args = parser.parse_args()
    main(args)