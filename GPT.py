
import math
import numpy as np
import openpyxl.comments.shape_writer
import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from torch.utils.data import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)

n_embd =96
max_len = 81
n_layer = 4
dropout = 0.5
n_head = 3


class MultiHeadAttention(nn.Module):

    def __init__(self, voc_embd ,d_model，num_hend):
        super().__init__()
        self.d_model = d_model
        self.num_hend = num_hend
        self.w_q = nn.Linear(voc_embd, d_model, bias= False)
        self.w_k = nn.Linear(voc_embd, d_model, bias=False)
        self.w_v = nn.Linear(voc_embd, d_model, bias=False)
        # 注册一个下三角矩阵
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))
        self.dropout = nn.Dropout(dropout)

        self.line = nn.Linear(d_model,voc_embd)
        self.norm = nn.LayerNorm(voc_embd)

    def forward(self, x):
        B,T,C = x.shape
        residual = x
        head_size = self.d_model/self.num_head
        q = self.w_q(x).view(B,-1,self.num_head, head_size).transpose(1,2)
        k = self.w_k(x).view(B,-1,self.num_head, head_size).transpose(1,2)
        v = self.w_v(x).view(B,-1,self.num_head, head_size).transpose(1,2)
        score = q @ k.transpose(-2,-1) * C**-0.5
        score = score.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        
        atten  = wei @ v
        output = self.line(atten)
        return self.norm(output+residual)




class FeedFoward(nn.Module):

    def __init__(self, voc_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(voc_embd, 4* voc_embd),nn.LeakyReLU(),
            nn.Linear(4* voc_embd, voc_embd), nn.Dropout(dropout),)
        self.norm = nn.LayerNorm(voc_embd)
        

    def forward(self, x):
        return self.norm(self.net(x) + x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd((self.ln2(x)))
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            # 随机抽样，随机抽样num_samples
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 模型训练
net = BigramLanguageModel()
logits, loss = net(batch['input'],batch['target'])

# 模型测试
net = BigramLanguageModel()
net.eval()
logits, loss = net(batch['input'],batch['target'])

# 生成模型
decode = lambda l: ''.join(seq_dict[i] for i in l)
net = torch.load('.pth')
for param in net.parameters():
    param.requires_grad = False

net.eval()
gen_voc = []
for i in range(100):
    voc = decode(net.generate(idx=torch.ones((1,1), dtype=torch.long).to(device),
                              max_new_tokens=40, temperature=1.0))
    voc = str(voc[1:])



class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = 26
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.rnn = nn.RNN(
            input_size=self.rnn_size,
            hidden_size=self.rnn_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.rnn_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length=183):
        return (torch.zeros(self.num_layers, sequence_length, self.rnn_size).to(device))


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = 26
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self,sequence_length=183):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

