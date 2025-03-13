import torch.nn as nn

class RNNModelNoTeacherForcing(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModelNoTeacherForcing, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True) # uses tanh by default
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        seq_len = x.size(1)
        input = self.embedding(x[:, 0]).unsqueeze(1)

        for _ in range(seq_len - 1):
            out, hidden = self.rnn(input, hidden)
            out = self.fc(out).squeeze(1)
            predicted = out.argmax(dim = -1).unsqueeze(1)
            input = self.embedding(predicted)
            
        return out

