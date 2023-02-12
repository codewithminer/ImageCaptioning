import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from layer.GCNLayer import GCNConv,EdgeConv, DynamicEdgeConv, GraphPooling

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling_type, edge_in_channels):
        super(GCN, self).__init__()
        self.edgeconv1 = EdgeConv(in_channels, hidden_channels, edge_in_channels)
        self.edgeconv2 = EdgeConv(hidden_channels, hidden_channels,edge_in_channels)
        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.pooling = GraphPooling(pooling_type)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data_list):
        x_list = [data.x.to('cuda') for data in data_list]
        edge_index_list = [data.edge_index.to('cuda') for data in data_list]
        batch_list = [data.batch.to('cuda') for data in data_list]
        edge_attr_list = [data.edge_attr.to('cuda') for data in data_list]

        x_list = [self.edgeconv1(x, edge_index, edge_attr) for x,edge_index,edge_attr in zip(x_list,edge_index_list,edge_attr_list)]
        x_list = [self.edgeconv2(x, edge_index, edge_attr) for x,edge_index,edge_attr in zip(x_list,edge_index_list,edge_attr_list)]
        x_list = [self.gcn1(x, edge_index) for x,edge_index in zip(x_list,edge_index_list)]
        x_list = [self.pooling(x, edge_index, None) for x,edge_index,batch in zip(x_list,edge_index_list,batch_list)]
        x_list = [self.fc(x) for x in x_list]
        return  torch.stack(x_list)


class ResNet(nn.Module):
    """Load the pretrained Resnet-152 and replace top fc layer"""
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)   # It “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)  # features size (1, 2048, 1, 1)
        features = features.reshape(features.size(0), -1)   # [x,y ... , z] [1, 2048]
        features = self.bn(self.linear(features))   # [1, 256]
        return features


class EncoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_size)

    def forward(self, vec1, vec2):
        data = torch.cat((vec1,vec2.unsqueeze(1)),1)
        hiddens,_ = self.lstm(data)
        outputs = self.linear(hiddens[:,-1,:])
        return outputs


class DecoderLSTM(nn.Module):
    def __init__(self,  embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderLSTM, self).__init__()
        """Set the hyper-parameters and build the layers."""
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding(9956, 256)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # features [128, 256]   features.unsqueeze(1)-> [128, 1, 256]
        embeddings = self.embed(captions)   # [128, 28, 256]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # [128, 29, 256]
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) # pack_padded_sequence provides a way to efficiently process padded sequences in a deep learning model by ignoring the padding elements, without actually removing them.
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs  # [1639, 9956]

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        # The states are updated in each iteration, and they are used to keep track of the information from previous time steps.
        # features [1, 256]
        sampled_ids = []
        inputs = features.unsqueeze(1)  # [1, 1, 256]
        
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size) [1,1,512]
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)   [1, 9956]
            _, predicted = outputs.max(1)                        # predicted: (batch_size)  [1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size) [1, 256]
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size) [1,1,256]
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length) [1,20]
        return sampled_ids


class EncoderDecoder(nn.Module):
    def __init__(self,
        gcn_in_channels, gcn_hidden_channels, gcn_out_channels, gcn_pooling_type, gcn_edge_in_channels,
        resnet_embed_size,
        encoder_lstm_embed_size, encoder_lstm_hidden_size, encoder_lstm_num_layers,
        decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size, decoder_lstm_num_layers, max_seq_length=20):
        super(EncoderDecoder, self).__init__()
        self.gcn = GCN(gcn_in_channels, gcn_hidden_channels, gcn_out_channels, gcn_pooling_type, gcn_edge_in_channels)
        self.resnet = ResNet(resnet_embed_size)
        self.encoder_lstm = EncoderLSTM(encoder_lstm_embed_size, encoder_lstm_hidden_size, encoder_lstm_num_layers)
        self.decoder_lstm = DecoderLSTM(decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size, decoder_lstm_num_layers, max_seq_length)
    
    def forward(self, gcn_data, resnet_data, captions, lengths, training=True):
        x_gcn = self.gcn(gcn_data)
        x_resnet = self.resnet(resnet_data)
        x_concat = self.encoder_lstm(x_gcn, x_resnet)
        if training:
            predicate_caption = self.decoder_lstm(x_concat, captions, lengths)
            return predicate_caption
        else:
            return x_concat


