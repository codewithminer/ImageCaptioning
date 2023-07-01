import os
import time
import argparse
import pickle
import torch
from torchvision import transforms
from utils.dataloader import getData
from utils.build_vocab import Vocabulary, GraphVocabulary # import this, Necessary for load vocab.pkl file
from models.model2 import EncoderDecoder
import torch.nn.functional as F
from utils.scene_graph_extractor import extractSceneGraphData
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    data = getData(
        args.val_image_dir, args.val_caption_path, args.val_sg_path,
        vocab, False, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)


    gcn_in_channels = 256
    gcn_hidden_channels = 512
    gcn_edge_in_channels = 256

    geo_gcn_in_channels = 4
    geo_gcn_hidden_channels = 128
    geo_edge_in_channels = 3

    resnet_embed_size = 2048

    decoder_lstm_embed_size = 256
    decoder_lstm_hidden_size = 512
    decoder_lstm_vocab_size = len(vocab)
    max_seq_length = 20

    # references = list()
    hypotheses = list()

    model = EncoderDecoder(
        gcn_in_channels, gcn_hidden_channels, gcn_edge_in_channels,
        geo_gcn_in_channels, geo_gcn_hidden_channels, geo_edge_in_channels,
        resnet_embed_size,
        decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size,  max_seq_length
    ).to(device)
    
    checkpoint = torch.load(args.load_ckpt)
    model = checkpoint['model']
    model.to(device)
    model.eval()

    decoder = model.decoder_lstm
    total_step = len(data)
    captions_json = []
    image_ids = []   
    for i,(images, captions, lengths, SGs, all_captions, img_id) in enumerate(data):
    
        image_ids.append(img_id[0])
        SG_data = extractSceneGraphData(SGs, vocab_graph)

        images = images.to(device)
        # captions = captions.to(device)

        features =  model(SG_data, images, None, None, training=False)
        sampled_ids = []
        # seq = []
        states = None
        inputs = features.unsqueeze(1)  # [1, 1, 256]
        for j in range(decoder.max_seq_length):
            hiddens, states = decoder.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size) [1,1,512]
            outputs = decoder.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)   [1, 9956]
            _, predicted = outputs.max(1)                        # predicted: (batch_size)  [1]
            sampled_ids.append(predicted)
            inputs = decoder.embed(predicted)                       # inputs: (batch_size, embed_size) [1, 256]
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size) [1,1,256]
        sampled_ids = torch.stack(sampled_ids, 1)  
        # print(sampled_ids.tolist())
        seq = sampled_ids.tolist()[0]
        if i % args.log_step == 0:
            print('Step [{}/{}]'
                    .format(i, total_step)) 
            
        # all_captions = all_captions[0]
        # img_captions = list(
        #     map(lambda c: list(filter(lambda w: w not in [vocab('<start>'), vocab('<end>'), vocab('<pad>')], c)), all_captions[0]))
        # references.append(img_captions)

        # Hypotheses
        hypotheses.append(list(filter(lambda w: w not in {vocab('<start>'), vocab('<end>'), vocab('<pad>')}, seq)))

    for i, image_id in enumerate(image_ids):
        cap = {'image_id': image_id, 'caption': ' '.join(ids_to_tokens(hypotheses[i], vocab))}
        captions_json.append(cap)
    with open('captions_val2014_Semantic_T8_e10_results.json', 'w') as f:
        json.dump(captions_json, f)

def ids_to_tokens(ids, vocab):
    return [vocab.idx2word[id] for id in ids]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument('--model_path', type=str, default='ckpt/', help='path for saving trained model')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_graph_path', type=str, default='datasets/vocab_graph.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_step', type=int , default=1000, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='if the validation loss has not improved for 10 consecutive epochs, early stopping will be triggered.')
    parser.add_argument('--load_ckpt', type=str, default='ckpt/Semantic-T8-10.pth', help='load pre-trained model')


    # Training set parameters
    parser.add_argument('--train_caption_path', type=str, default='datasets/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--train_image_dir', type=str, default='datasets/images/train_resized_2014/', help='direcotory for resized images')
    parser.add_argument('--train_sg_path', type=str, default='datasets/SG/Train_Detected_Scene_Graphs.json', help='path for train scene graph json file')
    parser.add_argument('--training', type=bool, default=True, help="don't take all captions")

    # Validation set parameters
    parser.add_argument('--val_image_dir', type=str, default='datasets/images/val_resized_2014/', help='direcotory for validation resized images')
    parser.add_argument('--val_caption_path', type=str, default='datasets/annotations/captions_val2014.json', help='path for validation annotation json file')
    parser.add_argument('--val_sg_path', type=str, default='datasets/SG/Val_Detected_Scene_Graphs.json', help='path for validation scene graph json file')
    parser.add_argument('--validation', type=bool, default=False, help="take all captions")

    # Semantic model parameters
    parser.add_argument('--sem_in_channels', type=int, default=768, help='dimention of node vectors')
    parser.add_argument('--sem_edge_in_channels', type=int, default=768, help='dimention of edge vectors')
    parser.add_argument('--sem_hidden_channels', type=int, default=1024, help='dimention of GCN hidden layer')
    parser.add_argument('--sem_out_channels', type=int, default=256, help='dimention of GCN out layer')
    parser.add_argument('--sem_pooling_type', type=str, default='mean', help='type of pooling layer')

    # Resnet model parameters
    parser.add_argument('--resnet_embed_size', type=int, default=256, help='dimention of image embedding vectors')

    # Geometric  model parameters
    parser.add_argument('--geo_in_channels', type=int, default=16, help='dimention of node vectors')
    parser.add_argument('--geo_edge_in_channels', type=int, default=512, help='dimention of edge vectors')
    parser.add_argument('--geo_hidden_channels', type=int, default=256, help='dimention of GCN hidden layer')
    parser.add_argument('--geo_out_channels', type=int, default=256, help='dimention of GCN out layer')
    parser.add_argument('--geo_pooling_type', type=str, default='mean', help='type of pooling layer')

    # Decoder model parameters
    parser.add_argument('--decoder_encoder_size', type=int, default=256, help='dimention of encoder-lstm embedding vectors')
    parser.add_argument('--decoder_embed_size', type=int, default=512, help='dimention of encoder-lstm embedding vectors')
    parser.add_argument('--decoder_hidden_size', type=int, default=512, help='dimention of Decoder-LSTM hidden states')
    parser.add_argument('--decoder_num_layers', type=int, default=1, help='number of layers in Decoder-LSTM')
    parser.add_argument('--max_seq_length', type=int, default=20, help='maximum lenght of sequences')

    # Training loop parametrs
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    args = parser.parse_args()
    main(args)
