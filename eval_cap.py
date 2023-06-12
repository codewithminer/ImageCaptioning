import os
import json
import argparse
import pickle
import torch
from torchvision import transforms
from utils.dataloader import getData
from utils.build_vocab import Vocabulary, GraphVocabulary # import this, Necessary for load vocab.pkl file
from models.models import EncoderDecoder
import torch.nn.functional as F
from utils.scene_graph_extractor import extractSceneGraphData

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

    references = list()
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

        encoded_visual, encoded_graph, encoded_geo =  model(SG_data, images, None, None, training=False)

        encoded_graph = encoded_graph.to_data_list()
        encoded_geo = encoded_geo.to_data_list()
        
        k = 5
        vocab_size = len(vocab)
        encoded_visual = encoded_visual.repeat(k, 1, 1)
        encoded_graph = encoded_graph * k
        encoded_geo = encoded_geo * k

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[vocab('<start>')]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        # complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        h, c = decoder.init_hidden_state(encoded_visual, encoded_graph, encoded_geo)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embed(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe = decoder.attention(h, encoded_visual, encoded_graph, encoded_geo)  # (s, encoder_dim), (s, num_pixels)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            next_word_inds = next_word_inds.to(torch.long)
            prev_word_inds = prev_word_inds.to(torch.long)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != vocab('<end>')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoded_visual = encoded_visual[prev_word_inds[incomplete_inds]]
            incomplete_inds_list = prev_word_inds[incomplete_inds].tolist()
            encoded_graph = [encoded_graph[i] for i in incomplete_inds_list]
            encoded_geo = [encoded_geo[i] for i in incomplete_inds_list]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
            # print(seqs)
        if i % args.log_step == 0:
            print('Step [{}/{}]'
                    .format(i, total_step)) 
            
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        # all_captions = all_captions[0]
        # img_captions = list(
        #     map(lambda c: list(filter(lambda w: w not in [vocab('<start>'), vocab('<end>'), vocab('<pad>')], c)), all_captions[0]))
        # references.append(img_captions)

        # Hypotheses
        hypotheses.append(list(filter(lambda w: w not in {vocab('<start>'), vocab('<end>'), vocab('<pad>')}, seq)))
        # assert len(references) == len(hypotheses)

        # references_str = [[ids_to_tokens(ref, vocab) for ref in ref_list] for ref_list in references]
        # hypotheses_str = [ids_to_tokens(hypo, vocab) for hypo in hypotheses]

    for i, image_id in enumerate(image_ids):
        cap = {'image_id': image_id, 'caption': ' '.join(ids_to_tokens(hypotheses[i], vocab))}
        captions_json.append(cap)
    with open('captions_val2014_SSRAT_T6_e15_results.json', 'w') as f:
        json.dump(captions_json, f)
    
    # Compute scores for each metric
    # Calculate BLEU-4 scores
    # bleu4 = corpus_bleu(references, hypotheses)
    # print(bleu4)
    # return bleu4

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
    parser.add_argument('--load_ckpt', type=str, default='ckpt/SSRAT-T6-15.pth', help='load pre-trained model')


    # Training set parameters
    parser.add_argument('--train_caption_path', type=str, default='datasets/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--train_image_dir', type=str, default='datasets/images/train_resized_2014/', help='direcotory for resized images')
    parser.add_argument('--train_sg_path', type=str, default='datasets/SG/Train_Detected_Scene_Graphs.json', help='path for train scene graph json file')
    parser.add_argument('--training', type=bool, default=True, help="don't take all captions")

    # Validation set parameters
    parser.add_argument('--val_image_dir', type=str, default='datasets/images/val5000/images/', help='direcotory for validation resized images')
    parser.add_argument('--val_caption_path', type=str, default='datasets/images/val5000/captions_val50002014.json', help='path for validation annotation json file')
    parser.add_argument('--val_sg_path', type=str, default='datasets/images/val5000/val5000_Detected_Scene_Graphs.json', help='path for validation scene graph json file')
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
