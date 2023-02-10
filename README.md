# Image Captioning
This project aims to generate captions for given images using a combination of scene graph features and visual features of the images. The model is built using Graph Convolutional Network (GCN), Residual Network (ResNet), and two Long Short-Term Memory (LSTM) networks.

## Prerequisites
- Python 3.7
- PyTorch
- Torchvision
- torch_geometric
- Matplotlib
- Numpy
- transformers

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
3. Download the pre-trained models and put them in the ckpt folder.
- model.pth
- RelTR.pth

4. Run the following command to train the model:
```bash
python train.py
```
5. Once the training is complete, you can use the following command to generate captions for new images:
```bash
python sample.py --img_path <path_to_image>
```

## Dataset
The model is trained on the MS-COCO dataset, which contains over 80K images and their corresponding captions.

## Evaluation
The performance of the model is evaluated using the BLEU score, which measures the similarity between the generated captions and the ground truth captions. A higher BLEU score indicates better performance.

## Conclusion
The project demonstrates the effectiveness of using a combination of scene graph features and visual features for image captioning. Further improvements can be made by fine-tuning the hyperparameters, training on a larger dataset, or using more advanced techniques like attention mechanisms.

