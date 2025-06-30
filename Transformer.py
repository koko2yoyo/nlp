import torch
import 
'''
Transforerm: Encoder + Decoder
Encoder：
    src_embedding
    Multi-Head-Attention: softmax(q*k/sqrt(d_k)).msak_fill_*v + line + residual + norm
    PosWiseFeedWard: line + relu + line + dropout + residual + norm

Decoder:
    dec_embedding
    
'''

MutilHeadAttention + PoswiseFeedForward

class MutliHeadAttention(nn.Moudel):
    def __init__(self,d_model,n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)

        self.line = nn.Linear(d_model,d_moedl)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,Q,K,V,atten_mask):
        residual, batch_size = Q,Q.szie(0)
        
        q_s = self.W_Q(Q).view(batch_size,-1,self.n_head,self.d_model/self.n_head).transpose(1,2)
        k_s = self.W_Q(Q).view(batch_size,-1,self.n_head,self.d_model/self.n_head).transpose(1,2)
        v_s = self.W_Q(Q).view(batch_size,-1,self.n_head,self.d_model/self.n_head).transpose(1,2)
        # 沿着维度复制张量
        atten_mask = atten_mask.unsqueeze(1).repeat(1,self.n_head,1,1) #
        # multi-head-attention
        scores = torch.matmul(q_s,k_s.transpose(-1,-2))/np.sqrt(self.self.d_model/self.n_head)
        scores.mask_fill_(atten_mask, -1e9) # mask为True的位置替换为value，mask为flase的位置保持原来的值,mask 为0 的为True
        atten = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(atten,q_s)

        context = context.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        context = self.line(context)
        return self.norm(residual + context)


class PosWiseFeedWard(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.line_1 = nn.Linear(d_model,4*d_model)
        self.line_2 = nn.Linear(4*d_model,d_model)
        
        self.dropout = nn.Dropout(0.4)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,x):
        output = nn.ReLU(self.line_1(x))
        output = nn.self.line_2(output)
        return self.norm(x + self.dropout(output))

class Encoder(nn.Module):
    def __init__(self,d_model,n_head,drop_rate=0.1):
        super().__init__()
        # self.d_model = d_model
        # self.head_num = head_num
        # self.head_size = head_size
        self.attention = MultiHeadAttention(d_model,n_head)
        self.ffn = PoswiseFeedWard(d_model)
       
    def forward(self,x,src_mask=None):
        
        # step1：multi-head-attention
        output, atten = self.self.attention(x,x,x,src_mask)
        # step2：position-wise-feedward
        x = self.ffn(x)
        
        return x, atten
        
            
class TransformerEncoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,d_model,n_head,device):
        super().__init__()
        
        self.embdeeing = TransformerEmbedding(d_model,max_len,enc_voc_size,device)
        self.layer = nn.ModuleList(Encoder(d_model,n_head) for _ in range(num_layers))
        
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return x


class Decoder(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()
        self.attention = MultiHeadAttention(d_model,n_head)

        self.enc_dec_attention = MultiHeadAttention(d_model,n_head)

        self.poswardfeedward = PosWiseFeedWard(d_model)
    def forward(self,dec,enc,dec_mask,enc_mask):
        redicus = dec
        dec_attn = self.attention(dec,dec,dec,dec_mask)
        if enc is not None:
            # computer encoder_decoder attention
            dec_enc = self.enc_dec_attention(dec_attn,enc,end,enc_mask)
        x = self.poswisefeedword()


class TransformerDecoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,n_head,n_layer):
        super().__init__()
        self.embedding = TransformerEmbedding(d_model,max_len,dec_voc_size,device)
        self.layers = nn.ModuleList(Decoder(d_model,n_head) for _ in range(n_layer))

        self.line = nn.Linear(d_model,dec_voc_size)
    def forward(self,dec,enc,dec_mask,enc_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec,enc,dec_mask,enc_mask)

        output = self.line(dec)
        return outpot

class Transformer(nn.Module):
    def __init__(self,seq_len,d_model,n_head):
        super().__init__()
        
        
        
        
    
    

