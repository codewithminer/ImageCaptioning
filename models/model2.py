import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from layer.GCNLayer import GCNConv,EdgeConv, GraphPooling
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torchvision.transforms.functional import crop, resize
from utils.tools import extractGeometricFeatureAsEdgeAttr

class AttentionOnGraph(nn.Module):
    def __init__(self, embed_size, attn_dim):
        super(AttentionOnGraph, self).__init__()
        self.attn_hidden = nn.Linear(attn_dim, attn_dim)
        self.attn_graph = nn.Linear(embed_size, attn_dim)
        self.score = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, batched_graph, hidden):
        data_list = batched_graph
        graphs = []
        for ind,data in enumerate(data_list):
            attn_hidden = self.attn_hidden(hidden)
            graph = data.x                      # [num_node, hidden_dim]
            attn_graph = self.attn_graph(graph)  # [num_node, hidden_dim]
            score = self.score(self.relu(attn_graph + attn_hidden[ind])).squeeze(1) # [num_node]
            score = self.softmax(score)         # [num_node]
            graph = score.unsqueeze(1) * graph  # [num_node, hidden_dim]
            graphs.append(graph.sum(0))         # [hidden_dim, hidden_dim, ...]
        return torch.stack(graphs)             # [batch_size, hidden_dim]

class AttentionOnResnet(nn.Module):
    def __init__(self, encoder_dim, attn_dim):
        super( AttentionOnResnet, self).__init__()
        self.attn_hidden = nn.Linear(attn_dim, attn_dim)
        self.attn_visual = nn.Linear(encoder_dim, attn_dim)  # linear layer to transform encoded image
        self.score = nn.Linear(attn_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
    
    def forward(self, encoder, hidden):
        attn_hidden = self.attn_hidden(hidden)
        attn_visual = self.attn_visual(encoder)  # (batch_size, num_pixels, attention_dim)
        score = self.score(self.relu(attn_visual + attn_hidden.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        score = self.softmax(score)  # (batch_size, num_pixels)
        attention_weighted = (encoder * score.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted

class SemanticGraph(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_in_channels, vocab_graph_size):
        super(SemanticGraph, self).__init__()

        self.edgeconv1 = EdgeConv(in_channels, hidden_channels, edge_in_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        self.edgeconv2 = EdgeConv(hidden_channels, hidden_channels,edge_in_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.dropout3 = nn.Dropout(0.5)

        self.embedding = nn.Embedding(vocab_graph_size, in_channels)
        self.fc = nn.Linear(hidden_channels, 256)
        # self.attention = AttentionSemanticGraph(hidden_channels, hidden_channels)


    def prepare(self, x, edge_attr, edge_index, node_lengths, edge_lengths):
        data_list = []
        for ind in range(x.shape[0]):
            data_list.append(Data(
                x=x[ind, :node_lengths[ind], :],
                edge_attr=edge_attr[ind, :edge_lengths[ind], :],
                edge_index=edge_index[ind].t().contiguous()
                ))
        batch = Batch.from_data_list(data_list)

        return batch

    def forward(self, scene_graph):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = scene_graph.padded_nodes.to(device)
        edge_attr = scene_graph.padded_edges.to(device)
        edge_index = scene_graph.edge_indexes

        x = self.embedding(x)
        edge_attr = self.embedding(edge_attr)

        node_lengths = [len(tokens) for tokens in scene_graph.node_tokens]
        edge_lengths = [len(tokens) for tokens in scene_graph.edge_tokens]

        batched_graph = self.prepare(x, edge_attr, edge_index, node_lengths, edge_lengths)

        x, edge_attr, edge_index = batched_graph.x, batched_graph.edge_attr, batched_graph.edge_index
        edge_index = edge_index.to(device)

        x = self.edgeconv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.edgeconv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.gcn1(x, edge_index)
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.fc(x)
        batched_graph.x = x

        # x = self.attention(batched_graph)   # [batch_size, embed_dim]

        return batched_graph

class GeometricGraph(nn.Module):
    def __init__(self,in_channels, hidden_channels, edge_in_channels):
        super(GeometricGraph, self).__init__()
        self.edgeconv1 = EdgeConv(in_channels, hidden_channels, edge_in_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)

        self.edgeconv2 = EdgeConv(hidden_channels, hidden_channels,edge_in_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)

        self.gcn1 = GCNConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(hidden_channels, 256)
        # self.attention = AttentionSpacialGraph(hidden_channels, 512)

    def forward(self, scene_graph):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_graph = extractGeometricFeatureAsEdgeAttr(scene_graph.bboxes, scene_graph.edge_indexes)
        x, edge_index, edge_attr = batched_graph.x.to(device), batched_graph.edge_index.to(device), batched_graph.edge_attr.to(device)

        x = self.edgeconv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.edgeconv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.gcn1(x, edge_index)

        x = self.fc(x)

        batched_graph.x = x

        # x = self.attention(batched_graph)

        return batched_graph

class ResNet(nn.Module):
    """Load the pretrained Resnet-152 and replace top fc layer"""
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)   # It “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.linear = nn.Linear(embed_size, 1024)
        # self.attention = AttentionResnet(512, 512)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)  # features size (1, 2048, 1, 1)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        encoder_dim = features.size(-1)
        features = features.view(features.size(0), -1, encoder_dim)
        features = self.linear(features)
        # features = self.attention(features)

        return features

class FuseGraphsWithAttention(nn.Module):
    def __init__(self, semantic_dim, spatila_dim, fuse_dim):
        super(FuseGraphsWithAttention, self).__init__()
        
        self.semantic_proj = nn.Linear(semantic_dim, fuse_dim)
        self.spatial_proj = nn.Linear(spatila_dim, fuse_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pooling = GraphPooling('mean')

    def forward(self, semantic_batch, spatial_batch):

        fused_graphs = []
        semantic_graphs = semantic_batch.to_data_list()
        spatial_graphs = spatial_batch.to_data_list()

        for i in range(len(semantic_graphs)):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            semantic_x = semantic_graphs[i].x.to(device)
            spatial_x = spatial_graphs[i].x.to(device)

            semantic_proj = self.relu1(self.semantic_proj(semantic_x))
            spatial_proj = self.relu2(self.spatial_proj(spatial_x))

            similarity_scores = torch.matmul(semantic_proj, spatial_proj.transpose(0, 1))

            attention_semantic = F.softmax(similarity_scores, dim=1)

            weighted_features_spatial = torch.matmul(attention_semantic, spatial_x)

            attention_spatial = F.softmax(similarity_scores.transpose(0, 1), dim=1)

            weighted_features_semantic = torch.matmul(attention_spatial, semantic_x)

            fused_features = torch.cat([semantic_x, weighted_features_spatial, spatial_x, weighted_features_semantic], dim=1)

            fused_graph = Data(x=fused_features)
            fused_graphs.append(fused_graph)

        fused_batch = Batch.from_data_list(fused_graphs)
        return fused_batch

class EncoderDecoder(nn.Module):
    def __init__(self,
        gcn_in_channels, gcn_hidden_channels, gcn_edge_in_channels,
        geo_gcn_in_channels, geo_gcn_hidden_channels, geo_edge_in_channels,
        resnet_embed_size,
        decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size, max_seq_length=20):
        super(EncoderDecoder, self).__init__()

        self.semantic_graph = SemanticGraph(gcn_in_channels, gcn_hidden_channels, gcn_edge_in_channels, 204)
        self.spacial_graph = GeometricGraph(geo_gcn_in_channels, geo_gcn_hidden_channels, geo_edge_in_channels)
        self.resnet = ResNet(resnet_embed_size)
        self.decoder_lstm = LSTMWithAttention(1024, decoder_lstm_embed_size, decoder_lstm_hidden_size, decoder_lstm_vocab_size, max_seq_length)
        

    # def forward(self, gcn_data, resnet_data, captions, lengths, bert_captions, training=True):
    def forward(self, sg_data, images, captions, lengths, training=True):
        x_semantic = self.semantic_graph(sg_data)
        x_geometric = self.spacial_graph(sg_data)
        x_resnet = self.resnet(images)

        if training:
            predicate_caption = self.decoder_lstm(x_resnet, x_semantic, x_geometric, captions, lengths)
            return predicate_caption
        else:
            return  x_resnet, x_semantic, x_geometric
           
class AttentionOnFeatures(nn.Module):
    def __init__(self, encode_size, decode_size):
        super(AttentionOnFeatures, self).__init__()

        # encode_size = 256 decode_size = 512
        self.attn_visual = AttentionOnResnet(encode_size, decode_size)
        self.attn_semantic = AttentionOnGraph(encode_size, decode_size)
        self.attn_geometric = AttentionOnGraph(encode_size, decode_size)
        

    def forward(self, hidden_state, visual_features, graph_features, geo_features):

        attn_visual = self.attn_visual(visual_features, hidden_state)       
        attn_semantic = self.attn_semantic(graph_features, hidden_state)
        attn_geometric = self.attn_geometric(geo_features, hidden_state)

        features = torch.cat([attn_visual, attn_semantic, attn_geometric], dim=-1)

        return features

class LSTMWithAttention(nn.Module):
    def __init__(self, encoder_size, embed_size, hidden_size, vocab_size, max_seq_length=20):
        super(LSTMWithAttention, self).__init__()
        """Set the hyper-parameters and build the layers."""
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding(9956, 512)
        self.attention = AttentionOnFeatures(encoder_size, hidden_size)
        self.lstm = nn.LSTMCell(3 * encoder_size + embed_size, hidden_size, bias=True)
        self.init_h = nn.Linear(3 * encoder_size, hidden_size)
        self.init_c = nn.Linear(3 * encoder_size, hidden_size)
        self.f_beta = nn.Linear(hidden_size, 3 * encoder_size) 
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.vocab_size = vocab_size
        

    def init_hidden_state(self, visual_features, graph_features, geo_features):

        semantic_graphs = []
        for data in graph_features:
            semantic_graphs.append(data.x.mean(dim=0))

        spacial_graphs = []
        for data in geo_features:
            spacial_graphs.append(data.x.mean(dim=0))

        mean_semantic_features = torch.stack(semantic_graphs)
        mean_spacial_features = torch.stack(spacial_graphs)
        mean_visual_features = visual_features.mean(dim=1)
        features = torch.cat((mean_visual_features, mean_semantic_features, mean_spacial_features), dim=1)

        h = self.init_h(features)
        c = self.init_c(features)
        return h, c
    
    def forward(self, visual_features, graph_features, geo_features, captions, lengths):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_features = graph_features.to_data_list()
        geo_features = geo_features.to_data_list()
        vocab_size = self.vocab_size
        batch_size = visual_features.size(0)

        embeddings = self.embed(captions)
        lengths = [num - 1 for num in lengths]

        h, c = self.init_hidden_state(visual_features, graph_features, geo_features)
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(lengths), vocab_size).to(device)

        for t in range(max(lengths)):
            batch_size_t = sum([l>t for l in lengths])
            attention_weighted_features = self.attention( 
                h[:batch_size_t], 
                visual_features[:batch_size_t], graph_features[:batch_size_t], geo_features[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_features = gate * attention_weighted_features
            h, c = self.lstm(
                torch.cat((embeddings[:batch_size_t, t, :], attention_weighted_features), dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
        packed_output = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
        return packed_output

    def caption_image_beam_search(self, encoded_visual, encoded_graph, encoded_geo, word_map, beam_size=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoded_graph = encoded_graph.to_data_list()
        encoded_geo = encoded_geo.to_data_list()

        k = beam_size
        vocab_size = len(word_map)

        encoded_visual = encoded_visual.repeat(k, 1, 1)
        encoded_graph = encoded_graph * k
        encoded_geo = encoded_geo * k

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map('<start>')]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        
        h, c = self.init_hidden_state(encoded_visual, encoded_graph, encoded_geo)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embed(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe = self.attention(h, encoded_visual, encoded_graph, encoded_geo)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
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
                            next_word != word_map('<end>')]
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
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq