import torch
import re
from torch_geometric.data import Data
from torch_geometric.data import Batch
from utils.build_vocab import GraphVocabulary
import torch
from torch.nn.utils.rnn import pad_sequence

class SceneGraphExtractor:
    def __init__(self, scene_graph, vocab_graph):
        self.sg = scene_graph
        self.node_labels = []
        self.edge_labels = []
        self.edge_indexes = []
        self.node_tokens = []
        self.edge_tokens = []
        # self.adjmat = []
        self.bboxes = []
        self.batch = []
        self.vocab = vocab_graph
        self.padded_nodes = tuple()
        self.padded_edges = tuple()

    def prepare(self):
        # extract nodes, edges and edge_index from scene graph
        inverse_indexes_list = []
        for item in self.sg:
            self.node_labels.append(item['node_labels'])
            self.edge_labels.append(item['edge_labels'])

            unique_edges, inverse_indexes = torch.tensor(item['edges']).unique(dim=0, return_inverse=True)
            inverse_indexes_list.append(inverse_indexes)
            self.edge_indexes.append(unique_edges)
            self.bboxes.append(item['node_bboxes'])

        for i, edges in enumerate(self.edge_labels):
            new_list = [None] * len(set(inverse_indexes_list[i].tolist()))
            for ind, label in enumerate(edges):
                index = inverse_indexes_list[i][ind].item()
                new_list[index] = label
            self.edge_labels[i] = new_list

        # Get tokens of node labels and edge labels
        for nodes in self.node_labels:
            nodes = [re.sub(r"\s+", "", s) for s in nodes]  # remove space of Dichotomous words
            nodes = [self.vocab(node) for node in nodes]
            self.node_tokens.append(nodes)
        self.padded_nodes = pad_sequence([torch.tensor(tokens) for tokens in self.node_tokens], batch_first=True, padding_value=0)

        for edges in self.edge_labels:
            edges = [re.sub(r"\s+", "", s) for s in edges]
            edges = [self.vocab(edge) for edge in edges]
            self.edge_tokens.append(edges)
        self.padded_edges = pad_sequence([torch.tensor(tokens) for tokens in self.edge_tokens], batch_first=True, padding_value=0)

        # get batches for pooling graph vectors
        for i,graph in enumerate(self.sg):
            self.batch.append(torch.tensor([i] * len(graph['node_labels'])))

    def build_adjacencyMatrix(self):
        E = self.edge_indexes
        for edges in E:
            size = len(set([n for e in edges for n in e])) 
            # make an empty adjacency list  
            adjacency = [[0]*size for _ in range(size)]
            # populate the list for each edge
            for sink, source in edges:
                adjacency[sink][source] = 1
            self.adjmat.append(adjacency)

    # def collectData():
    #     return 

    # def loadData(self):
    #     data_list = []
    #     for ind in range(len(self.nodeVec)):
    #         data_list.append(Data(
    #             x=self.nodeVec[ind],
    #             edge_attr=self.edgeVec[ind],
    #             edge_index=self.edge_indexes[ind].t().contiguous()
    #             ))
    #     batch = Batch.from_data_list(data_list)

    #     return batch, self.bboxes, self.edgeIndexs

    def __getitem__(self, index):
        return self.nodeVec[index], self.edgeVec[index], self.edgeIndexs[index]

    def __len__(self):
        return len(self.nodeLabels)


def extractSceneGraphData(sg, vocab_graph):
    scene_graph = SceneGraphExtractor(sg, vocab_graph)
    scene_graph.prepare()
    return scene_graph