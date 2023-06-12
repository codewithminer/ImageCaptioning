import os
import argparse
import pickle 
import torch
import cv2
import math
from torchvision import transforms
from PIL import Image
from utils.build_vocab import Vocabulary, GraphVocabulary # import this, Necessary for load vocab.pkl file
# from utils.scene_graph_extractor import getSGData
from models.models import EncoderDecoder
from RelTR.scenegraph import SceneGraph
from utils.tools import extractGeometricFeaturesForGCN, extractGeometricFeatureAsEdgeAttr
from utils.scene_graph_extractor import extractSceneGraphData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    # image = image.resize([224, 224], Image.Resampling.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
        # print(image.shape)
    
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
    
    # Load Graph Vocabulary wrapper
    with open(args.vocab_graph_path, 'rb') as f:
        vocab_graph = pickle.load(f)

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

    model = EncoderDecoder(
        gcn_in_channels, gcn_hidden_channels, gcn_edge_in_channels,
        geo_gcn_in_channels, geo_gcn_hidden_channels,geo_edge_in_channels,
        resnet_embed_size,
        decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size,  max_seq_length
    ).to(device)
    

    # Load the trained model parameters
    checkpoint = torch.load(args.encoder_decoder_path)
    model = checkpoint['model']
    model.to(device)
    model.eval()

    images = os.listdir('images')
    for image in images:
        image = os.path.join('images/', image)
        # img2 = cv2.imread(image)
        img = load_image(image, transform)
        image_tensor = img.to(device)
        # Generate an caption from the image
        sg = SceneGraph(image)
        if sg:
            print(sg)
            SG_data = extractSceneGraphData((sg[image],), vocab_graph)
            # batched_graph, feat = extractGeometricFeaturesForGCN(SG_data.bboxes, SG_data.edge_indexes)
            # visualizeGeometricFeatures(img2, feat,SG_data.edge_indexes)
            # feature = model(SG_data, image_tensor,None, None, training=False)
            # sampled_ids = model.decoder_lstm.sample(feature)
            f1, f2, f3 = model(SG_data, image_tensor,None, None, training=False)
            sampled_ids = model.decoder_lstm.caption_image_beam_search(f1, f2, f3, vocab)

            # sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            print (f"{sentence}[{image}]")
        else:
            print("Image no have SceneGraph")
    
def visualizeGeometricFeatures(image, features_list, edge_indexes_list):
    for i, features in enumerate(features_list):
        for j, bbox_feature in enumerate(features):
            center = tuple(map(int, bbox_feature[:2]))
            size = tuple(map(int, bbox_feature[2:4]))
            # angle = int(bbox_feature[4])
            color = (0, 255, 0) if j == 0 else (255, 0, 0)
            cv2.rectangle(image, (center[0]-size[0]//2, center[1]-size[1]//2), (center[0]+size[0]//2, center[1]+size[1]//2), color, 2)
            
            # Draw lines between related objects
            for rel in edge_indexes_list[i]:
                if j == rel[0]:
                    other_center = tuple(map(int, features_list[i][rel[1]][:2]))
                    cv2.line(image, center, other_center, color, 2)
    
    cv2.imshow('Image with Geometric Features', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False, help='input image for generating caption')
    parser.add_argument('--encoder_decoder_path', type=str, default='ckpt/SSRAT-T6-30.pth', help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--vocab_graph_path', type=str, default='datasets/vocab_graph.pkl', help='path for vocabulary wrapper')

    args = parser.parse_args()
    main(args)
