import torch
from torch import nn
import math



class ParamVecFormerBlock(nn.Module):
    
    def __init__(self, dim,heads,seq_len,num_paramvec):
            
        super().__init__()
        self.num_heads = heads
        self.hidden_dim = dim
        self.seq_len = seq_len
        self.num_paramvec = num_paramvec
        self.head_dim = dim // self.num_heads
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
        assert self.head_dim * self.num_heads == self.hidden_dim 
        self.paramvecs = nn.Parameter(torch.randn(1,self.num_paramvec,dim)).to('cuda')
       
       
    def forward(self, x):
         
        batch_size,full_len,_ = x.size()
        if full_len != (self.seq_len + self.num_paramvec):
            extends = torch.zeros(batch_size, self.num_paramvec,self.hidden_dim).to('cuda')
            x = torch.cat((x,extends),dim=-2)
        	
        pvecs = self.paramvecs.repeat(batch_size,1,1)
        
       
        
        x[:,self.seq_len:,:] = pvecs + x[:,self.seq_len:,:]
        
        residual = x
        
        x = self.norm(x)
        
       
        P,S = x,x
       

        
        P = P.view(batch_size, self.seq_len + self.num_paramvec, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.view(batch_size, self.seq_len + self.num_paramvec, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        

       
        attention_scores = P @ S.transpose(-1, -2) / math.sqrt(self.head_dim)

       

        
        attention_weights = torch.softmax(attention_scores, dim=-1)

        

       
        context = attention_weights @ S

        
        context = context.transpose(1, 2).contiguous().view(batch_size, self.seq_len + self.num_paramvec, self.hidden_dim)

        out = context + residual
        
        return  out   





class ParamVecFormer(nn.Module):
    def __init__(self, d_model,heads,num_tokens, num_layers,num_paramvec):
        super().__init__()
        
        self.num_paramvec = num_paramvec
        self.d_model = d_model
        self.seq_len = num_tokens
        self.heads = heads    
        self.model = nn.Sequential(
            *[ParamVecFormerBlock(self.d_model,self.heads,self.seq_len,self.num_paramvec) for _ in range(num_layers)]
        )

    def forward(self, x):
    
        
        out = self.model(x)
        return out[:,0:self.seq_len,:]








