import nltk
import pickle
import argparse
from collections import Counter
from transformers import BertTokenizer
import sys
sys.path.append("./")
from cocoapi.PythonAPI.pycocotools.coco import COCO
from utils.tools import RemoveHashtag

class BERTVocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word,token_id):
        if not word in self.word2idx:
            self.word2idx[word] = {'token_ids':token_id, 'idx':self.idx}
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self,word):
        if not word in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        caption_tokens = tokenizer.tokenize(caption.lower())    # ['john', 'is', 'going', 'to', 'bed', '.']   
        caption_tokens = RemoveHashtag(caption_tokens)
        counter.update(caption_tokens)                                  # {'jogn':1 , 'is':1, ...}

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold] # word -> list of words with more than 4 iterate
    # Create a vocab wrapper and add some special tokens.
    vocab = BERTVocabulary()
    vocab.add_word('[PAD]',[0])
    vocab.add_word('[CLS]',[101])
    vocab.add_word('[SEP]',[102])
    vocab.add_word('[UNK]',[100])

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        caption_tokens = tokenizer.tokenize(word)
        token_id = tokenizer.convert_tokens_to_ids(caption_tokens)
        vocab.add_word(word,token_id)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)   # <dump> for serialization(convert to byte stream) and <load> for de-serialization
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='datasets/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./datasets/BERTVocabulary.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)