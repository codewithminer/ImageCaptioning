import nltk
import torch
import re
from torch_geometric.data import Data
from torch_geometric.data import Batch

class ExtractSceneGraph:
    def __init__(self,sg, word2id):
        self.sg = sg
        self.nodes = []
        self.edges = []
        self.edgeIndexes = []
        self.bboxes = []
        self.word2id = word2id

        
    def tokenizeNodes(self):
        nodeLabels = []
        edgeLabels = []
        for item in self.sg:
            nodeLabels.append(item['node_labels'])
            edgeLabels.append(item['edge_labels'])
            self.edgeIndexes.append(torch.tensor(item['edges']))
            self.bboxes.append(item['node_bboxes'])

        # tokenize nodes and edges
        for nodes in nodeLabels:
            nodes = [re.sub(r"\s+", "", s) for s in nodes]
            nodes = ' '.join(nodes)
            nodeTokens = nltk.tokenize.word_tokenize(nodes)
            tokens = []
            for token in nodeTokens:
                tokens.append(self.word2id[token])
            
            self.nodes.append(torch.tensor(tokens))

        for edges in edgeLabels:
            edges = [re.sub(r"\s+", "", s) for s in edges]
            edges = ' '.join(edges)
            edgeTokens = nltk.tokenize.word_tokenize(edges)
            tokens = []
            for token in edgeTokens:
                tokens.append(self.word2id[token])
            self.edges.append(torch.tensor(tokens))

    def loadData(self):
        data_list = []
        for ind in range(len(self.nodes)):
            data_list.append(Data(
                x=self.nodes[ind],
                edge_attr=self.edges[ind],
                edge_index=self.edgeIndexes[ind].t().contiguous()
                ))
        batch = Batch.from_data_list(data_list)
        return self.nodes, self.edges, batch.edge_index
        # return batch, self.bboxes, self.edgeIndexes

def extractSG(sg, word2id):
    extract = ExtractSceneGraph(sg,word2id)
    extract.tokenizeNodes()
    return extract.loadData()

def prepareSGVocab():
        CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

        REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                    'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                    'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                    'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                    'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                    'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

        word2id = dict()
        nodes = [re.sub(r"\s+", "", s) for s in CLASSES]
        edges = [re.sub(r"\s+", "", s) for s in REL_CLASSES]
        nodes = ' '.join(nodes)
        edges = ' '.join(edges)
        nodes = nltk.tokenize.word_tokenize(nodes)
        edges = nltk.tokenize.word_tokenize(edges)
        counter = 0
        for node in nodes:
            word2id[node] = counter
            counter += 1
            
        for edge in edges:
            word2id[edge] = counter
            counter += 1

        return word2id