import argparse
import pickle 
import numpy as np 
import torch
from torchvision import transforms 
import matplotlib.pyplot as plt
from PIL import Image
from utils.build_vocab import Vocabulary # import this, Necessary for load vocab.pkl file
from utils.sg_dataloader import getSGData
from utils.dataholder import DataHolder
from models.models import EncoderDecoder
from models.Bert import Bert
from RelTR.scenegraph import SceneGraph


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.Resampling.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

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
    model.eval()
    bert = Bert()
    word2vec = DataHolder().word2vec

    # Prepare an image
    img = 'images/COCO_val2014_000000000692.jpg'
    image = load_image(img, transform)
    image_tensor = image.to(device)

    # Load the trained model parameters
    model.load_state_dict(torch.load(args.encoder_decoder_path))

    
    # Generate an caption from the image
    sg = SceneGraph(img)
    SG_data =  getSGData((sg[img],),bert,word2vec)
    feature = model(SG_data,image_tensor,None, None, training=False)
    sampled_ids = model.decoder_lstm.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    image = Image.open(img).convert('RGB')
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False, help='input image for generating caption')
    parser.add_argument('--encoder_decoder_path', type=str, default='ckpt/encoder-decoder-5-3000.pth', help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
