import torch
from torch import nn



     
class GlobalMappingUnit(nn.Module):
    
    def __init__(self, dim,heads,num_paramvec):
            
        super().__init__()
        self.num_heads = heads
        self.hidden_dim = dim
        self.num_paramvec = num_paramvec
        self.head_dim = dim // self.num_heads
        self.norm = nn.LayerNorm(dim,elementwise_affine=False)
        assert self.head_dim * self.num_heads == self.hidden_dim 
        self.paramvecs = nn.Parameter(torch.randn(1,self.num_paramvec,dim)).to('cuda')
       
       
    def forward(self, x):
        
        batch_size, seq_len, _ = x.size()
        pvecs = self.paramvecs.repeat(batch_size,1,1)
        
        x = self.norm(x)
        x = torch.cat((x,pvecs),dim=-2)
        
        P,S = x,x
       

        
        P = P.view(batch_size, seq_len + self.num_paramvec, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.view(batch_size, seq_len + self.num_paramvec, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        

       
        attention_scores = P @ S.transpose(-1, -2) / math.sqrt(self.head_dim)

       

        
        attention_weights = torch.softmax(attention_scores, dim=-1)

        

       
        context = attention_weights @ S

        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len + self.num_paramvec, self.hidden_dim)

        
        return context [:,0:seq_len,:]        




class ParamVecFormerBlock(nn.Module):
    def __init__(self, d_model,heads,num_paramvec):
        super().__init__()
       
         
        
        self.global_mapping = GlobalMappingUnit(d_model,heads,num_paramvec)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
       
        
        
        return x



class ParamVecFormer(nn.Module):
    def __init__(self, d_model,heads, num_layers,num_paramvec):
        super().__init__()
        
        self.model = nn.Sequential(
            *[ParamVecFormerBlock(d_model,heads,num_paramvec) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








