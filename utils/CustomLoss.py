import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class BertDistancesLoss(nn.Module):
    def __init__(self, vocab):
        super(BertDistancesLoss, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.vocab = vocab

    # convert target Ids and caption Ids to list of sentences
    def convertIdsToSentences(self, captions, targets, lengths):
        # captions [1639, 9956] targets [1639]
        caption_list = []
        counter = 0
        for i,length in enumerate(lengths):
            caption_sentence = []
            _,index = captions[counter:counter+length].max(1)
            for ind in index.tolist():
                pred_words = self.vocab.idx2word[ind]
                caption_sentence.append(pred_words)
            caption_sentence = caption_sentence[1:-1] # remove <start> and <end>
            caption_sentence = ' '.join(caption_sentence)
            caption_list.append(caption_sentence)
            counter = counter+length

        targets,_ = pad_packed_sequence(targets, batch_first=True)
        target_list = []
        target_sentence = []
        for target in targets:
            for word_id in target:
                word = self.vocab.idx2word[int(word_id)]
                if word != '<start>' and word != '<end>':
                    target_sentence.append(word)
                if word == '<end>':
                    target_sentence = ' '.join(target_sentence)
                    target_list.append(target_sentence)
                    target_sentence = []
                    break
        return caption_list, target_list    

    # find the semantic distance between target_list and caption_list
    def loss(self,captions, targets, lengths):
        caption_list, target_list = self.convertIdsToSentences(captions, targets, lengths)

        loss = torch.tensor([], requires_grad=True, device='cuda')
        # initialize dictionary that will contain tokenized sentences
        tokens = {'input_ids': [], 'attention_mask': []}
        sentences = caption_list + target_list
        for sentence in sentences:
                # tokenize sentence and append to dictionary lists
            new_tokens = self.tokenizer.encode_plus(sentence, max_length=max(lengths), truncation=True,
                                            padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
        outputs = self.model(**tokens)
        # The general goal of computing the mean-pooled embeddings is to reduce the dimensions of the embeddings from the output of the BERT model, 
        # which can have a high dimensionality, to a lower-dimensional representation that can be used for comparison or further processing.
        embeddings = outputs.last_hidden_state  # [256, max_length, 768]
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)  # torch.clamp ensures that summed_mask is always greater than zero, avoiding any division by zero errors.
        mean_pooled = summed / summed_mask # [256, 768]
        # convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().numpy()
        # calculate
        for i in range(len(caption_list)):
            Difference = 1 - cosine_similarity(
                [mean_pooled[i]],
                mean_pooled[int((len(sentences)/2))+i:int((len(sentences)/2))+i+1]
            )
            Difference = torch.from_numpy(Difference).to('cuda')
            loss = torch.cat((loss, Difference), dim=0)
        return torch.mean(loss)