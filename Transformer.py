import torch
import 

# transformer encoder 主要由attention+残差层+dropout+两层线性层+dropout组成
class mutli_head_attention(nn.Module):
  def __init__(self,d_model,head_num,head_size):
    super().__init__()
    self.d_model =d_model
    self.head_num = head_num
    self.head_size =head_size
    self.Wq = nn.linear(self.d_model,self.head_num*)
    
class Transformer_encoder(nn.Module):
  def __init__(self,d_model,head_num,head_size):
    super().__init__()
    self.d_model =d_model
    self.head_num = head_num
    self.head_size =head_size

  def forward(self,input):
    
    

