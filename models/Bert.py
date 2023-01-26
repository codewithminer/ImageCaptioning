from transformers import BertTokenizer, BertModel
import torch

class Word2Vector:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    # Return Tensor with shape [N, 768] N: number of words.
    def get_word_vectors(self,labels):
        texts = labels
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        word_vectors = outputs.pooler_output.detach().numpy()
        word_vectors = torch.Tensor(word_vectors)
        return word_vectors

