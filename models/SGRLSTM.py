import torch
import torch.nn as nn

class SGRLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layer):
        super(SGRLSTM,self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layer,batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_size)

    def forward(self, vec1, vec2):
        print(vec1.shape)
        print(vec2.unsqueeze(1).shape)
        data = torch.cat((vec1,vec2.unsqueeze(1)),1)
        print(data.shape)
        hiddens,_ = self.lstm(data)
        print(hiddens[0].shape)
        print(hiddens.shape)
        print(hiddens[:,-1,:].shape)
        outputs = self.linear(hiddens[:,-1,:])
        return outputs
