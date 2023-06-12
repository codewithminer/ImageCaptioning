import torch
import math
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.data import Batch

def extractGeometricFeaturesForLSTM(bboxes_list, edge_indexes_list, num_features=16):
    bboxes_tensor,bboxes_length = convertBboxListToTensor(bboxes_list)
    batch_size = len(bboxes_tensor)
    max_num_bbox = max([bboxes.shape[0] for bboxes in bboxes_tensor])
    features_list = []
    for bboxes, edge_indexes in zip(bboxes_tensor, edge_indexes_list):
        bbox_features = torch.zeros((max_num_bbox, num_features), dtype=torch.float32)
        bbox_features[:, :2] = (bboxes[:, :2] + bboxes[:, 2:]) / 2  # center
        bbox_features[:, 2:4] = bboxes[:, 2:] - bboxes[:, :2]  # size
        
        for i in range(max_num_bbox):
          relations = []
          for ind,rel in enumerate(edge_indexes):
            if i == rel[0].item():
              src_center = bbox_features[i, :2]
              dst_center = bbox_features[rel[1].item(), :2]
              src_size = bbox_features[i, 2:4]
              dst_size = bbox_features[rel[1].item(), 2:4]
              angle = torch.atan2(dst_center[1] - src_center[1], dst_center[0] - src_center[0])
              angle = angle * 180 / math.pi
              distance = torch.norm(dst_center - src_center, dim=-1)
              overlap = compute_overlap(bbox_features[i, :4], bbox_features[rel[1].item(), :4])
              relations.append(torch.tensor([angle, distance, overlap], dtype=torch.float32))

          if relations:
            # Concatenate the tensors along a new dimension
            stacked_tensor = torch.stack(relations, dim=1)
            # Reshape the tensor into the desired shape
            flattened_tensor = stacked_tensor.view(-1)
            # print(flattened_tensor)
            if len(flattened_tensor) > 12:
              flattened_tensor = flattened_tensor[:12]
            bbox_features[i, 4:len(flattened_tensor)+4] = flattened_tensor
        features_list.append(bbox_features)
    return torch.stack(features_list, dim=0), bboxes_length

def extractGeometricFeaturesForGCN(bboxes_list, edge_indexes_list, num_features=16):
    bboxes_tensor,bboxes_lengths = convertBboxListToTensor(bboxes_list)
    batch_size = len(bboxes_tensor)
    max_num_bbox = max([bboxes.shape[0] for bboxes in bboxes_tensor])
    features_list = []
    for bboxes, edge_indexes in zip(bboxes_tensor, edge_indexes_list):
        bbox_features = torch.zeros((max_num_bbox, num_features), dtype=torch.float32)
        bbox_features[:, :2] = (bboxes[:, :2] + bboxes[:, 2:]) / 2  # center
        bbox_features[:, 2:4] = bboxes[:, 2:] - bboxes[:, :2]  # size
        for i in range(max_num_bbox):
          relations = []
          for ind,rel in enumerate(edge_indexes):
            if i == rel[0].item():
              src_center = bbox_features[i, :2]
              dst_center = bbox_features[rel[1].item(), :2]
            #   src_size = bbox_features[i, 2:4]
            #   dst_size = bbox_features[rel[1].item(), 2:4]
              angle = torch.atan2(dst_center[1] - src_center[1], dst_center[0] - src_center[0])
              angle = angle * 180 / math.pi
              angle = angle % 360
              angle = angle / 360
              distance = torch.norm(dst_center - src_center, dim=-1)
              overlap = compute_overlap(bbox_features[i, :4], bbox_features[rel[1].item(), :4])
              relations.append(torch.tensor([angle, distance, overlap], dtype=torch.float32))

          if relations:
            # Concatenate the tensors along a new dimension
            stacked_tensor = torch.stack(relations, dim=0)
            # Reshape the tensor into the desired shape
            flattened_tensor = stacked_tensor.view(-1)
            # print(flattened_tensor)
            if len(flattened_tensor) > 12:
              flattened_tensor = flattened_tensor[:12]
            bbox_features[i, 4:len(flattened_tensor)+4] = flattened_tensor

        features_list.append(bbox_features)
    features = torch.stack(features_list, dim=0)
    mask = torch.zeros(batch_size, max_num_bbox, dtype=torch.float)
    for i, l in enumerate(bboxes_lengths):
        mask[i, :l] = 1
    # Convert tensor to list
    tensor_list = features.tolist()
    # Remove padding using the mask
    result = [torch.tensor([tensor_list[i][j] for j in range(mask.shape[1]) if mask[i][j] == 1]) for i in range(mask.shape[0])]

    data_list = []
    for ind in range(len(result)):
        data_list.append(Data(
            x=result[ind],
            edge_index=edge_indexes_list[ind].t().contiguous()
            ))
    batch = Batch.from_data_list(data_list)

    return batch,result

def extractGeometricFeatureAsEdgeAttr(bboxes, edge_indexes):
   
    # node_features = []
    # edge_features = []
    data_list = []
    for bbox, edge_index in zip(bboxes, edge_indexes):
        node_attr = torch.zeros((len(bbox), 4), dtype=torch.float32)
        edge_attr = torch.zeros((len(edge_index), 3), dtype=torch.float32)

        for ind, box in enumerate(bbox):
            node_attr[ind, :2] = torch.tensor([(box[0] + box[2])/2, (box[1] + box[3])/2])
            node_attr[ind, 2:4] = torch.tensor([(box[2] - box[0]), (box[3] - box[1])])

        for ind, rel in enumerate(edge_index):
            bbox_src = bbox[rel[0].item()]
            bbox_dst = bbox[rel[1].item()]
            
            src_center = node_attr[rel[0].item(), :2]
            dst_center = node_attr[rel[1].item(), :2]

            angle = torch.atan2(dst_center[1] - src_center[1], dst_center[0] - src_center[0])
            angle = angle * 180 / math.pi
            angle = angle % 360

            distance = torch.norm(dst_center - src_center, dim=-1)

            overlap = compute_overlap(torch.cat((src_center, node_attr[rel[0].item(), 2:4]), dim=0), torch.cat((dst_center, node_attr[rel[1].item(), 2:4]), dim=0))

            edge_attr[ind, 0] = distance
            edge_attr[ind, 1] = angle
            edge_attr[ind, 2] = overlap

        data_list.append(Data(
           x = node_attr,
           edge_attr = edge_attr,
           edge_index = edge_index.t().contiguous()
        ))
          
        # node_features.append(node_attr)
        # edge_features.append(edge_attr)
        
    batch = Batch.from_data_list(data_list)

        # return batch,result
    return batch

# def compute_overlap(bbox1, bbox2):
#     # Calculate the intersection between two bounding boxes
#     # Calculate the intersection between two bounding boxes
#     x1, y1, w1, h1 = bbox1
#     x2, y2, w2, h2 = bbox2
#     inter_x1 = max(x1, x2)
#     inter_y1 = max(y1, y2)
#     inter_x2 = min(x1 + w1, x2 + w2)
#     inter_y2 = min(y1 + h1, y2 + h2)
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     # Calculate the area of object B
#     bbox2_area = w2 * h2
#     # Calculate the percentage of object B inside object A
#     return inter_area / bbox2_area

def compute_overlap(src_centers_sizes, dst_centers_sizes):
    src_x1 = src_centers_sizes[0] - src_centers_sizes[2] / 2
    src_y1 = src_centers_sizes[1] - src_centers_sizes[3] / 2
    src_x2 = src_centers_sizes[0] + src_centers_sizes[2] / 2
    src_y2 = src_centers_sizes[1] + src_centers_sizes[3] / 2
    dst_x1 = dst_centers_sizes[0] - dst_centers_sizes[2] / 2
    dst_y1 = dst_centers_sizes[1] - dst_centers_sizes[3] / 2
    dst_x2 = dst_centers_sizes[0] + dst_centers_sizes[2] / 2
    dst_y2 = dst_centers_sizes[1] + dst_centers_sizes[3] / 2
    x1 = torch.max(src_x1.unsqueeze(0), dst_x1.unsqueeze(0))
    y1 = torch.max(src_y1.unsqueeze(0), dst_y1.unsqueeze(0))
    x2 = torch.min(src_x2.unsqueeze(0), dst_x2.unsqueeze(0))
    y2 = torch.min(src_y2.unsqueeze(0), dst_y2.unsqueeze(0))
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    src_area = src_centers_sizes[2] * src_centers_sizes[3]
    dst_area = dst_centers_sizes[2] * dst_centers_sizes[3]
    union = src_area.unsqueeze(0) + dst_area.unsqueeze(0) - intersection
    overlap = intersection / union
    return overlap

# overlap = 0
# if (bbox_src[0] < bbox_dst[2] and bbox_src[2] > bbox_dst[0] and
#     bbox_src[1] < bbox_dst[3] and bbox_src[3] > bbox_dst[1]):
#     # Calculate intersection area
#     inter_area = (min(bbox_src[2], bbox_dst[2]) - max(bbox_src[0], bbox_dst[0])) * \
#                  (min(bbox_src[3], bbox_dst[3]) - max(bbox_src[1], bbox_dst[1]))
#     # Calculate union area
#     union_area = (bbox_src[2] - bbox_src[0]) * (bbox_src[3] - bbox_src[1]) + \
#                  (bbox_dst[2] - bbox_dst[0]) * (bbox_dst[3] - bbox_dst[1]) - inter_area
#     # Calculate overlap
#     overlap = inter_area / union_area


def convertBboxListToTensor(bbox_list):
    # Convert the nested list to a list of tensors
    tensor_list = [torch.tensor(sublist) for sublist in bbox_list]
    bboxes_length = [len(bbox) for bbox in bbox_list]
    # Pad the list of tensors to make them of equal length
    bboxes = pad_sequence(tensor_list, batch_first=True, padding_value=0.0)
    return bboxes, bboxes_length


def save_model(model, epoch, optimizer, early_stopping_counter, path, name, bleu4):
    
    state = {'model': model,
             'optimizer': optimizer,
             'epoch': epoch,
             'stopping_counter': early_stopping_counter,
             'bleu-4':bleu4}
    
    filename = path + name
    torch.save(state, filename)

