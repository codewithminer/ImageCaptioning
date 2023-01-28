class DataHolder:
    def __init__(self):
        self.word2vec = {}

    def insert(self, word, vec):
        self.word2vec[word] = vec
    
    def getW2V(self):
        return self.word2vec