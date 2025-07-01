import torch
import 
'''
Transforerm: Encoder + Decoder
Encoder：
    src_embedding
    
    MultiHeadAttention:  softmax(q*k/sqrt(d_k)).mask_fill_*v + line  + residual + norm
   
    PosWiseFeedWard     :  line + relu + line + dropout + residual + norm

Decoder:
    dec_embedding
    
    dec-MultiHeadAttention    : softmax(q*k/sqrt(d_k).mask_fill(dec_mask)*v + line  + residual + norm
    dec-enc-MultiHeadAttention: softmax(q*k/sqrt(d_k).mask_fill(dec_enc_mask)*v + line  + residual + norm

    PosWiseFeedWard           : line + relu + line + dropout + residual + norm
    
'''
class PositionEmbedding(nn.Module):
    def __init__(self,max_len,voc_embed):
        self.encoding = torch.zeros(max_len,voc_embed, device)
        self.encoding.requires_grad = False
        pos = torch.arange(0,max_len,device)
        pos = pos.float().unsqueeze(dim=1)

        pos_i = torch.arange(0,voc_embed,2,device).float()

        self.encodeing[:,0::2] = torch.sin(pos/10000**(pos_i/voc_embed))
        self.encodeing[:,1::2] = torch.cos(pos/10000**(pos_i/voc_embed))
    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, voc_embed, max_len, device):
        super().__init__()
        self.token_emb = nn.Enbedding(vocab_size,voc_embed)
        self.pos_emb = PositionEmbedding(voc_embed, max_len, device)

    def forward(self,x):
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)
        
        return token_emb + pos_emb
    
    

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
            dec_enc = self.enc_dec_attention(dec_attn, enc, enc, enc_mask)
        x = self.poswisefeedword(dec_enc)
        return x


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
    def __init__(self, seq_len, voc_embd, d_model, n_head,device):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(seq_len,d_model,n_head)

        self.decoder = TransformerDecoder(seq_len,d_model,n_head)

    def forward(self,src,dec):
        B,T = src.size()
        enc_mask = src.data.eq(0).unsqueeze(1).expand(B,T,T)
        dec_mask = get_dec_mask(dec)

        enc_out = self.encoder(src, enc_mask)
        dec_out = self.decoder(dec, enc_out, dec_mask, enc_mask)

        return dec_out
        

    def get_dec_mask(self,dec):
        B,T = dec.size()
        dec_pad_mask = dec.data.eq(0).unsqueeze(1).expand(B, T, T)
        dec_seq_mask = torch.tril(torch.ones(T, T)).type(torch.ByteTensor).to(self.devic)
        dec_mask = torch.gt(dec_pad_mask+dec_seq_mask,0)
        return dec_mask
        
        


        
        
        
        
    
    

