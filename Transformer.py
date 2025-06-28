import torch
import 

# Transformer主要由Encoder和Decoder组成
# Encoder:MutilHeadAttention + PoswiseFeedForward

class
    
class Transformer_encoder(nn.Module):
  def __init__(self,d_model,head_num,head_size):
    super().__init__()
    self.d_model =d_model
    self.head_num = head_num
    self.head_size =head_size

  def forward(self,input):
    
    

