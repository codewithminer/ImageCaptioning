from models.Bert import Word2Vector
import torch
from torch_geometric.data import Data

class SceneGraphLoader:
    def __init__(self,scene_graph, bert,word2vec):
        self.sg = scene_graph
        self.nodeLabels = []
        self.edgeLabels = []
        self.edgeIndexs = []
        self.nodeVec = []
        self.edgeVec = []
        self.adjmat = []
        self.batch = []
        self.bert = bert
        self.w2v = word2vec

    def prepareSG(self):
        # extract nodes, edges and edge_index from scene graph
        for item in self.sg:
            self.nodeLabels.append(item['node_labels'])
            self.edgeLabels.append(item['edge_labels'])
            self.edgeIndexs.append(torch.tensor(item['edges']))

        # Convert nodel labels and edge labels to vector (with Bert Model)
        for nodes in self.nodeLabels:
            self.nodeVec.append(self.bert.get_word_vectors(nodes))
        for edges in self.edgeLabels:
            self.edgeVec.append(self.bert.get_word_vectors(edges))

        # get batches for pooling graph vectors
        for i,graph in enumerate(self.sg):
            self.batch.append(torch.tensor([i] * len(graph['node_labels'])))

    def adjacencyMatrix(self):
        E = self.edgeIndexs
        for edges in E:
            size = len(set([n for e in edges for n in e])) 
            # make an empty adjacency list  
            adjacency = [[0]*size for _ in range(size)]
            # populate the list for each edge
            for sink, source in edges:
                adjacency[sink][source] = 1
            self.adjmat.append(adjacency)

    def loadData(self):
        data_list = []
        for ind in range(len(self.nodeVec)):
            data_list.append(Data(
                x=self.nodeVec[ind],
                edge_attr=self.edgeVec[ind],
                edge_index=self.edgeIndexs[ind].t().contiguous(),
                batch=self.batch[ind]
                ))
        return data_list
        # return self.nodeVec, self.edgeVec, self.edgeIndexs, self.batch

    def __getitem__(self, index):
        return self.nodeVec[index], self.edgeVec[index], self.edgeIndexs[index]

    def __len__(self):
        return len(self.nodeLabels)


def getSGData(sg, bert,word2vec):
    scene_graph = SceneGraphLoader(sg, bert,word2vec)
    scene_graph.prepareSG()
    # scene_graph.adjacencyMatrix()
    return scene_graph.loadData()