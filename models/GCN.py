import torch
import torch.nn as nn
from layer.GCNLayer import GCNConv,EdgeConv, DynamicEdgeConv, GraphPooling

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling_type, edge_in_channels):
        super(GCNModel, self).__init__()
        self.edgeconv1 = DynamicEdgeConv(in_channels, hidden_channels, edge_in_channels)
        self.edgeconv2 = DynamicEdgeConv(in_channels, hidden_channels, edge_in_channels)
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.pooling = GraphPooling(pooling_type)
        self.fc = nn.Linear(out_channels, 256)

    def forward(self, data_list):
        x_list = [data.x for data in data_list]
        edge_index_list = [data.edge_index for data in data_list]
        batch_list = [data.batch for data in data_list]
        edge_attr_list = [data.edge_attr for data in data_list]

        x_list = [self.edgeconv1(x, edge_index, edge_attr) for x,edge_index,edge_attr in zip(x_list,edge_index_list,edge_attr_list)]
        x_list = [self.edgeconv2(x, edge_index, edge_attr) for x,edge_index,edge_attr in zip(x_list,edge_index_list,edge_attr_list)]
        x_list = [self.gcn1(x, edge_index) for x,edge_index in zip(x_list,edge_index_list)]
        x_list = [self.pooling(x, edge_index, batch) for x,edge_index,batch in zip(x_list,edge_index_list,batch_list)]
        x_list = [self.fc(x) for x in x_list]
        return x_list



# import torch.nn as nn
# import torch.nn.functional as F
# from layer.GCNLayer import GraphConvolution


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
