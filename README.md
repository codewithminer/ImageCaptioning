# Image Captioning
This project aims to generate captions for given images using a combination of scene graph features and visual features of the images. The model is built using Graph Convolutional Network (GCN), Residual Network (ResNet), and two Long Short-Term Memory (LSTM) networks.

## Prerequisites
- Python 3.7
- PyTorch
- Torchvision
- Torch_Geometric
- Matplotlib
- Numpy
- Transformers

## Scene Graph Generation
1. Relational Transformer (RelTR) is used to generate the scene graph of the image.
2. The generated scene graph is then passed to the GCN to extract its features.

## Visual Feature Extraction
1. The ResNet network is used to extract visual features from the input image.

## Combination of Features
1. The output of the GCN and ResNet is combined and fed into the first LSTM network.
2. The output of the first LSTM is then passed to another LSTM network along with a vocabulary.

## Caption Generation
The second LSTM network uses the combined features and vocabulary to generate the final caption for the image.

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/codewithminer/image-captioning.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Dataset:
We trained the model on the MS-COCO dataset, which contains over 80K images and their corresponding captions. if you want, you can use another dataset.
- Download the [MS-COCO train2014](http://images.cocodataset.org/zips/train2014.zip) dataset and extract it in the datasets/images/ folder.
- Download the [MS-COCO annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) dataset and extract it in the datasets/annotations/ folder.

4. Preprocessing:
- generate a vocabulary of captions (If you do not want to build the vocabulary from scratch, you can Download vocab.pkl from this [link](https://github.com/yrcong/RelTR) and put it in the datasets folder.):
```bash
 python build_vocab
```
-  resize images to feed neural networks:
```bash
 python resize_image.py
```
- generate scene graph dataset to feed GCN (You must first download the pretrained RelTR model from this [link](https://github.com/yrcong/RelTR) and put it in the ckpt folder):
```bash
 python inference.py
```

5. Run the following command to train the model:
```bash
python train.py
```
If you do not want to train the model from scratch, you can Download [model.pth](https://drive.google.com/file/d/1TLhe4QcaxvznmVyzA8msiug2Bq9-eAqd/view?usp=share_link) and put it in the ckpt folder.

6. Once the training is complete, you can use the following command to generate captions for new images you have in the images folder:
```bash
python sample.py
```

Tree structure of folders:
```bash
├── ckpt
├   ├── encoder-decoder.pth
├   ├── checkpoint0149.pth
├── datasets
├   ├── annotations
├       ├── captions_train2014.json
├   ├── images
├       ├── resized2014
├   ├── SG
├       ├── Detected_Scene_Graph.json
├   ├── vocab.pkl
├── images

```

## Conclusion
The project demonstrates the effectiveness of using a combination of scene graph features and visual features for image captioning. Further improvements can be made by fine-tuning the hyperparameters, training on a larger dataset, or using more advanced techniques like attention mechanisms.
