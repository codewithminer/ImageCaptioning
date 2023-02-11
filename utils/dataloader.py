import json
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from utils.build_vocab import Vocabulary
from cocoapi.PythonAPI.pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, sg, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            img_dir: image directory.
            ann_file: coco annotation file path.
            sg: scene graph file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.sg = self.getSceneGraphJson(sg)
        self.ids = self.getIdsHaveSG(list(self.coco.anns.keys()))
        self.vocab = vocab
        self.transform = transform
    
    def getSceneGraphJson(self,sg):
        file = {}
        with (open(sg,'r')) as f:
            file = json.load(f)
        return file

    def getIdsHaveSG(self,ids):
        """Return annotations IDs that have generated scene graph"""
        sg_image_ids = list(self.sg.keys()) # ['coco2014_000123.jpg',...]
        anns_ids = []
        counter = 0
        for id in ids:
            img_id = self.coco.anns[id]['image_id'] # 123
            if  self.coco.loadImgs(img_id)[0]['file_name'] in sg_image_ids:
                anns_ids.append(id)
                counter += 1
        print(counter)
        return anns_ids

    def __getitem__(self,index):
        """Returns one data pair (image, caption and graph)."""
        coco = self.coco
        sg = self.sg
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        #Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, sg[path]

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, scene_graph).
        
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.

        Args:
            data: list of tuple (image, caption, scene_graph). 
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.
                - sene_graph: generated Image scene graph (node_label, node_bboxes, edge_labels, edge_index)

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
            scene_graph: list of scene graph(batch_size)
        """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, sg = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)


    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths, sg

def getData(img_dir, ann_file, sg,  vocab, transform, batch_size, shuffle, num_workers):
    # Using COCOAPI for loading images and captions
    coco = CocoDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        sg=sg,
        vocab=vocab,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=coco,
        batch_size=64,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader