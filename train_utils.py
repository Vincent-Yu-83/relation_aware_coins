import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, batch_size, coin_num, window_size, feature_number,
                 d_model_Encoder,d_model_Decoder, encoder, decoder, price_series_pe, local_price_pe, local_context_length):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size=batch_size
        self.coin_num=coin_num
        self.window_size=window_size
        self.feature_number=feature_number
        self.d_model_Encoder=d_model_Encoder
        self.d_model_Decoder=d_model_Decoder
        self.linear_price_series = nn.Linear(in_features=feature_number,out_features=d_model_Encoder)
        self.linear_local_price = nn.Linear(in_features=feature_number,out_features=d_model_Decoder)
        self.price_series_pe = price_series_pe
        self.local_price_pe = local_price_pe
        self.local_context_length=local_context_length
        self.linear_out=nn.Linear(in_features=1+d_model_Encoder,out_features=1)
        self.linear_out2=nn.Linear(in_features=1+d_model_Encoder,out_features=1)
        self.bias = torch.nn.Parameter(torch.zeros([1,1,1]))
        self.bias2 = torch.nn.Parameter(torch.zeros([1,1,1]))

    def forward(self, price_series, local_price_context, previous_w, price_series_mask, local_price_mask, padding_price):  ##[4, 128, 31, 11]
        #price_series:[4,128,31,11] 
        price_series = price_series/price_series[0:1,:,-1:,:]
        price_series = price_series.permute(3,1,2,0) #[4,128,31,11]->[11,128,31,4]
        price_series = price_series.contiguous().view(price_series.size()[0]*price_series.size()[1],self.window_size,self.feature_number) #[11,128,31,4]->[11*128,31,4]     
        price_series = self.linear_price_series(price_series)                            #[11*128,31,3]->[11*128,31,2*12]
        price_series = self.price_series_pe(price_series)                       #[11*128,31,2*12] 
        price_series = price_series.view(self.coin_num,-1,self.window_size,self.d_model_Encoder)  #[11*128,31,2*12]->[11,128,31,2*12]
        encode_out = self.encoder(price_series, price_series_mask)
#        encode_out=self.linear_src_2_embedding(encode_out)
###########################padding price#######################################################################################
        if(padding_price is not None):
            local_price_context = torch.cat([padding_price,local_price_context],2)    #[11,128,5-1,4] cat [11,128,1,4] -> [11,128,5,4]
            local_price_context = local_price_context.contiguous().view(local_price_context.size()[0]*price_series.size()[1],self.local_context_length*2-1,self.feature_number)  #[11,128,5,4]->[11*128,5,4] 
        else:
            local_price_context = local_price_context.contiguous().view(local_price_context.size()[0]*price_series.size()[1],1,self.feature_number)
##############Divide by close price################################
        local_price_context = local_price_context/local_price_context[:,-1:,0:1]
        local_price_context = self.linear_local_price(local_price_context)                   #[11*128,5,4]->[11*128,5,2*12]
        local_price_context = self.local_price_pe(local_price_context)                       #[11*128,5,2*12]
        if(padding_price is not None):
            padding_price = local_price_context[:,:-self.local_context_length,:]                                                    #[11*128,5-1,2*12]
            padding_price = padding_price.view(self.coin_num,-1,self.local_context_length-1,self.d_model_Decoder)   #[11,128,5-1,2*12]
        local_price_context = local_price_context[:,-self.local_context_length:,:]                                                              #[11*128,5,2*12]
        local_price_context = local_price_context.view(self.coin_num,-1,self.local_context_length,self.d_model_Decoder)                         #[11,128,5,2*12]
#################################padding_price=None###########################################################################
        decode_out = self.decoder(local_price_context, encode_out, price_series_mask, local_price_mask, padding_price)
        decode_out = decode_out.transpose(1,0)                                                          #[11,128,1,2*12]->#[128,11,1,2*12]   
        decode_out = torch.squeeze(decode_out,2)      #[128,11,1,2*12]->[128,11,2*12]
        previous_w = previous_w.permute(0,2,1)        #[128,1,11]->[128,11,1]
        out = torch.cat([decode_out,previous_w],2)    #[128,11,2*12]  cat [128,11,1] -> [128,11,2*12+1]
###################################  Decision making ##################################################
        out2 = self.linear_out2(out)                  #[128,11,2*12+1]->[128,11,1]
        out = self.linear_out(out)                    #[128,11,2*12+1]->[128,11,1]

        bias = self.bias.repeat(out.size()[0],1,1)    #[128,1,1]
        bias2 = self.bias2.repeat(out2.size()[0],1,1) #[128,1,1]

        out = torch.cat([bias,out],1)                 #[128,11,1] cat [128,1,1] -> [128,12,1]
        out2 = torch.cat([bias2,out2],1)              #[128,11,1] cat [128,1,1] -> [128,12,1]

        out = out.permute(0,2,1)                      #[128,1,12]
        out2 = out2.permute(0,2,1)                    #[128,1,12]

        out = F.softmax(out, dim = -1)
        out2 = F.softmax(out2, dim = -1)

        out = out*2
        out2 = -out2
        return out+out2                             #[128,1,12]


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):    #[64,10,512]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
#            print("Encoder:",x)
            x = layer(x, mask)
#            print("Encoder:",x.size())
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, None, None))
        return self.sublayer[1](x, self.feed_forward)

######################################Decoder############################################
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, price_series_mask, local_price_mask, padding_price):
        for layer in self.layers:
            x = layer(x, memory, price_series_mask, local_price_mask, padding_price)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, price_series_mask, local_price_mask, padding_price):  
        "Follow Figure 1 (right) for connections."
        m = memory 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, local_price_mask, padding_price, padding_price))  
        x = x[:,:,-1:,:]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, price_series_mask, None, None))
        return self.sublayer[2](x, self.feed_forward)



def subsequent_mask(size):   
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')         
    return torch.from_numpy(subsequent_mask) == 0   



def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) #64
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)  #[30, 8, 9, 9] 
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, asset_atten, h, d_model, dropout, local_context_length):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.local_context_length=local_context_length
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.conv_q= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
        self.conv_k= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)

        self.ass_linears_v = nn.Linear(d_model, d_model)
        self.ass_conv_q= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
        self.ass_conv_k= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
        
        self.attn = None
        self.attn_asset = None
        self.dropout = nn.Dropout(p=dropout)
        self.feature_weight_linear=nn.Linear(d_model, d_model)
        self.asset_atten=asset_atten

    def forward(self, query, key, value, mask, padding_price_q,padding_price_k):
        #query [4,128,1,2*12] or (4,128,31,2*12) key, value(4,128,31,2*12)
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)                      # [128,1,1,31]    [128,1,1,1]
            mask = mask.repeat(query.size()[0], 1, 1, 1)  # [128*3,1,1,31]  [128*3,1,1,1]    #[9, 1, 1, 31]
            mask = mask.cuda()                            
        q_size0 = query.size(0)  #11
        q_size1 = query.size(1)  #128
        q_size2 = query.size(2)  #31 0r 1
        q_size3 = query.size(3)  #2*12
        key_size0 = key.size(0)  
        key_size1 = key.size(1)  
        key_size2 = key.size(2)  
        key_size3 = key.size(3)  
##################################query#################################################    
        if(padding_price_q is not None):
            padding_price_q = padding_price_q.permute((1,3,0,2))   #[11,128,4,2*12]->[128,2*12,11,4]
            padding_q = padding_price_q
        else:
            if(self.local_context_length>1):
                padding_q = torch.zeros((q_size1,q_size3,q_size0,self.local_context_length-1)).cuda()
            else:
                padding_q = None
        query = query.permute((1,3,0,2))
        if(padding_q is not None):
            query = torch.cat([padding_q,query],-1)
##########################################context-agnostic query matrix##################################################
        #linar
        query = self.conv_q(query)
        query = query.permute((0,2,3,1))                                                                                          #[128,2*12,11,31+4]->[128,11,31+4,2*12]
        ########################################### local-attention ######################################################
        local_weight_q = torch.matmul(query[:,:,self.local_context_length-1:,:], query.transpose(-2, -1))/ math.sqrt(q_size3)     #[128,11,31,2*12] *[128,11,2*12,31+4]->[128,11,31,31+4]
        #[128,11,31,31+4]->[128,11,1,5*31]
        local_weight_q_list = [F.softmax(local_weight_q[:,:,i:i+1,i:i+self.local_context_length], dim = -1) for i in range(q_size2)]
        local_weight_q_list = torch.cat(local_weight_q_list,3)
        #[128,11,1,5*31]->[128,11,5*31,1]
        local_weight_q_list = local_weight_q_list.permute(0,1,3,2)
        #[128,11,31+4,2*12]->[128,11,5*31,2*12]
        q_list = [query[:,:,i:i+self.local_context_length,:] for i in range(q_size2)]
        q_list = torch.cat(q_list,2)
        #[128,11,5*31,1]*[128,11,5*31,2*12]->[128,11,5*31,2*12]
        query = local_weight_q_list*q_list
        #[128,11,5*31,2*12]->[128,11,5,31,2*12]
        query = query.contiguous().view(q_size1,q_size0,self.local_context_length,q_size2,q_size3)
        #[128,11,5,31,2*12]->[128,11,31,2*12]
        query = torch.sum(query,2)
        #[128,11,31,2*12]->[128,2*12,11,31]
        query = query.permute((0,3,1,2))
######################################################################################
        query = query.permute((2,0,3,1))                                     #[128,2*12,11,31] ->[11,128,31,2*12] 
        query = query.contiguous().view(q_size0*q_size1,q_size2,q_size3)     #[11,128,31,2*12] ->[11*128,31,2*12] 
        query = query.contiguous().view(q_size0*q_size1,q_size2,self.h, self.d_k).transpose(1, 2)   #[11*128,31,2*12] ->[11*128,31,2,12]->[11*109,2,31,12] 
#####################################key#################################################
        if(padding_price_k is not None):
            padding_price_k =padding_price_k.permute((1,3,0,2))              #[11,128,4,2*12]->#[128,2*12,11,4]
            padding_k=padding_price_k
        else:
            if(self.local_context_length>1):
                padding_k = torch.zeros((key_size1,key_size3,key_size0,self.local_context_length-1)).cuda()
            else:
                padding_k = None
        key = key.permute((1,3,0,2)) 
        if(padding_k is not None):
            key = torch.cat([padding_k,key],-1) 
##########################################context-aware key matrix############################################################################
        #linar
        key=self.conv_k(key)
        key=key.permute((0,2,3,1))                                                                                             #[128,2*12,11,31+4]->[128,11,31+4,2*12]
        ########################################### local-attention ##########################################################################
        local_weight_k=torch.matmul(key[:,:,self.local_context_length-1:,:], key.transpose(-2, -1))/ math.sqrt(key_size3)      #[128,11,31,2*12] *[128,11,2*12,31+4]->[128,11,31,31+4]
        #[128,11,31,31+4]->[128,11,1,5*31]
        local_weight_k_list=[F.softmax(local_weight_k[:,:,i:i+1,i:i+self.local_context_length], dim = -1) for i in range(key_size2)]
        local_weight_k_list=torch.cat(local_weight_k_list,3)
        #[128,11,1,5*31]->[128,11,5*31,1]
        local_weight_k_list=local_weight_k_list.permute(0,1,3,2)
        #[128,11,31+4,2*12]->[128,11,5*31,2*12]
        k_list=[key[:,:,i:i+self.local_context_length,:] for i in range(key_size2)]
        k_list=torch.cat(k_list,2)
        #[128,11,5*31,1]*[128,11,5*31,2*12]->[128,11,5*31,2*12]
        key=local_weight_k_list*k_list
        #[128,11,5*31,2*12]->[128,11,5,31,2*12]
        key=key.contiguous().view(key_size1,key_size0,self.local_context_length,key_size2,key_size3)
        #[128,11,5,31,2*12]->[128,11,31,2*12]
        key=torch.sum(key,2)
        #[128,11,31,2*12]->[128,2*12,11,31]
        key=key.permute((0,3,1,2))
#        key = self.conv_k(key)
        key = key.permute((2,0,3,1))
        key = key.contiguous().view(key_size0*key_size1,key_size2,key_size3)
        key = key.contiguous().view(key_size0*key_size1,key_size2,self.h, self.d_k).transpose(1, 2) 
##################################################### value matrix #############################################################################        
        value=value.view(key_size0*key_size1,key_size2,key_size3)                          #[4,128,31,2*12]->[4*128,31,2*12]
        nbatches=q_size0*q_size1
        value=self.linears[0](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  #[11*128,31,2,12]
        
################################################ Multi-head attention ##########################################################################               
        x, self.attn = attention(query, key, value, mask=None, 
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x=x.view(q_size0,q_size1,q_size2,q_size3)    #D[11,128,1,2*12] or E[11,128,31,2*12]

########################## Relation-attention ######################################################################
        if(self.asset_atten):
#######################################ass_query#####################################################################    
            ass_query=x.permute((2,1,0,3))      #D[11,128,1,2*12]->[1,128,11,2*12] or E[11,128,31,2*12]->[31,128,11,2*12]
            ass_query = ass_query.contiguous().view(q_size2*q_size1,q_size0,q_size3) #[31,128,11,2*12] -> [31*128,11,2*12] 
            ass_query = ass_query.contiguous().view(q_size2*q_size1,q_size0,self.h, self.d_k).transpose(1, 2)    #[31*109,8,11,64]     
########################################ass_key####################################################################
            ass_key=x.permute((2,1,0,3))        #D[11,128,1,2*12]->[1,128,11,2*12] or E[11,128,31,2*12]->[31,128,11,2*12]
            ass_key = ass_key.contiguous().view(q_size2*q_size1,q_size0,q_size3)     #[31,128,11,2*12]->[31*128,11,2*12] 
            ass_key = ass_key.contiguous().view(q_size2*q_size1,q_size0,self.h, self.d_k).transpose(1, 2)    #[31*128,2,11,12]  
####################################################################################################################
            ass_value=x.permute((2,1,0,3))      #D[11,128,1,2*12]->[1,128,11,2*12] or E[11,128,31,2*12]->[31,128,11,2*12]
            ass_value=ass_value.contiguous().view(q_size2*q_size1,q_size0,q_size3) #[31,128,11,2*12]->[31*128,11,2*12]
            ass_value=ass_value.contiguous().view(q_size2*q_size1,-1, self.h, self.d_k).transpose(1, 2)  #[31*128,2,11,12]
######################################################################################################################    
#            ass_mask=torch.ones(q_size2*q_size1,1,1,q_size0).cuda()  #[31*128,1,1,11]
            x, self.attn_asset = attention(ass_query, ass_key, ass_value, mask=None, 
                                 dropout=self.dropout)   
            x = x.transpose(1, 2).contiguous().view(q_size2*q_size1, -1, self.h * self.d_k)  #[31*128,11,2*12]        
            x=x.view(q_size2,q_size1,q_size0,q_size3) #[31,128,11,2*12]
            x=x.permute(2,1,0,3)  #[11,128,31,2*12]            
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
#        print("ffn:",x.size())
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, start_indx, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.start_indx=start_indx
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, self.start_indx:self.start_indx+x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class NoamOpt:
    "Optim wrapper that implements rate."
    #512, 1, 400
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1,beta=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate


    def forward(self, w, y):            # w:[128,1,12]   y:[128,11,4] 
        close_price=y[:,:,0:1].cuda()   #   [128,11,1]
        #future close prise (including cash)
        close_price=torch.cat([torch.ones(close_price.size()[0],1,1).cuda(),close_price],1).cuda()         #[128,11,1]cat[128,1,1]->[128,12,1]
        reward=torch.matmul(w,close_price)                                                                 #[128,1,1]
        close_price=close_price.view(close_price.size()[0],close_price.size()[2],close_price.size()[1])    #[128,1,12] 
###############################################################################################################
        element_reward=w*close_price
        interest=torch.zeros(element_reward.size(),dtype=torch.float).cuda()
        interest[element_reward<0]=element_reward[element_reward<0]
        interest=torch.sum(interest,2).unsqueeze(2)*self.interest_rate  #[128,1,1]
###############################################################################################################
        future_omega=w*close_price/reward  #[128,1,12]           
        wt=future_omega[:-1]               #[128,1,12]
        wt1=w[1:]                          #[128,1,12]
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio   #[128,1]
        pure_pc=pure_pc.cuda()
        pure_pc=torch.cat([torch.ones([1,1]).cuda(),pure_pc],0)
        pure_pc=pure_pc.view(pure_pc.size()[0],1,pure_pc.size()[1])       #[128,1,1]
        
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)
################## Deduct transaction fee ##################
        reward=reward*pure_pc    #reward=pv_vector
################## Deduct loan interest ####################
        reward=reward+interest
        portfolio_value=torch.prod(reward,0)
        batch_loss=-torch.log(reward)
#####################variance_penalty##############################
#        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean() #+ self.gamma*variance_penalty + self.beta*cost_penalty.mean() 
            return loss, portfolio_value[0][0]
        else:
            loss = batch_loss.mean() #+self.gamma*variance_penalty + self.beta*cost_penalty.mean() #(dim=0)                           
            return loss, portfolio_value[0][0]

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        loss, portfolio_value= self.criterion(x,y)         
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value



def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)

class Test_Loss(nn.Module):
    def __init__(self, commission_ratio,interest_rate,gamma=0.1,beta=0.1, size_average=True):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate

    def forward(self, w, y):               # w:[128,10,1,12] y(128,10,11,4)
        close_price = y[:,:,:,0:1].cuda()    #   [128,10,11,1]
        close_price = torch.cat([torch.ones(close_price.size()[0],close_price.size()[1],1,1).cuda(),close_price],2).cuda()       #[128,10,11,1]cat[128,10,1,1]->[128,10,12,1]
        reward = torch.matmul(w,close_price)   #  [128,10,1,12] * [128,10,12,1] ->[128,10,1,1]
        close_price = close_price.view(close_price.size()[0],close_price.size()[1],close_price.size()[3],close_price.size()[2])  #[128,10,12,1] -> [128,10,1,12]
##############################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(),dtype = torch.float).cuda()
        interest[element_reward<0] = element_reward[element_reward<0]
#        print("interest:",interest.size(),interest,'\r\n')
        interest = torch.sum(interest,3).unsqueeze(3)*self.interest_rate  #[128,10,1,1]
##############################################################################
        future_omega = w*close_price/reward    #[128,10,1,12]*[128,10,1,12]/[128,10,1,1]
        wt=future_omega[:,:-1]                 #[128, 9,1,12]   
        wt1=w[:,1:]                            #[128, 9,1,12]
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio     #[128,9,1]
        pure_pc=pure_pc.cuda()
        pure_pc=torch.cat([torch.ones([pure_pc.size()[0],1,1]).cuda(),pure_pc],1)      #[128,1,1] cat  [128,9,1] ->[128,10,1]        
        pure_pc=pure_pc.view(pure_pc.size()[0],pure_pc.size()[1],1,pure_pc.size()[2])  #[128,10,1] ->[128,10,1,1]          
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)                                 #[128, 9, 1]      
################## Deduct transaction fee ##################
        reward = reward*pure_pc                                                        #[128,10,1,1]*[128,10,1,1]  test: [1,2808-31,1,1]
################## Deduct loan interest ####################
        reward= reward+interest
        if not self.size_average:
            tst_pc_array=reward.squeeze()
            sr_reward=tst_pc_array-1
            SR=sr_reward.mean()/sr_reward.std()
#            print("SR:",SR.size(),"reward.mean():",reward.mean(),"reward.std():",reward.std())
            SN=torch.prod(reward,1) #[1,1,1,1]
            SN=SN.squeeze() #
#            print("SN:",SN.size())
            St_v=[]
            St=1.            
            MDD=max_drawdown(tst_pc_array)
            for k in range(reward.size()[1]):  #2808-31
                St*=reward[0,k,0,0]
                St_v.append(St.item())
            CR=SN/MDD            
            TO=cost_penalty.mean()
##############################################
        portfolio_value=torch.prod(reward,1)     #[128,1,1]
        batch_loss=-torch.log(portfolio_value)   #[128,1,1]

        if self.size_average:
            loss = batch_loss.mean() 
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean() 
            return loss, portfolio_value[0][0][0],SR,CR,St_v,tst_pc_array,TO


class SimpleLossCompute_tst:
    "A simple loss compute and train function."
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value= self.criterion(x,y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO = self.criterion(x,y)     
            return loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO   



def make_std_mask(local_price_context,batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size,1,1)==1)            
    local_price_mask = local_price_mask & (subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))   
    return local_price_mask    



def train_one_step(DM,x_window_size,model,loss_compute,local_context_length):
    batch=DM.next_batch()
    batch_input = batch["X"]        #(128, 4, 11, 31)
    batch_y = batch["y"]            #(128, 4, 11)
    batch_last_w = batch["last_w"]  #(128, 11)
    batch_w = batch["setw"]     
#############################################################################
    previous_w=torch.tensor(batch_last_w,dtype=torch.float).cuda()
    previous_w=torch.unsqueeze(previous_w,1)                         #[128, 11] -> [128,1,11]
    batch_input=batch_input.transpose((1,0,2,3))
    batch_input=batch_input.transpose((0,1,3,2))
    src=torch.tensor(batch_input,dtype=torch.float).cuda()   
    price_series_mask = (torch.ones(src.size()[1],1,x_window_size)==1)   #[128, 1, 31] 
    currt_price=src.permute((3,1,2,0))                                   #[4,128,31,11]->[11,128,31,4]
    if(local_context_length>1):
        padding_price=currt_price[:,:,-(local_context_length)*2+1:-1,:] 
    else:
        padding_price=None
    currt_price=currt_price[:,:,-1:,:]                                    #[11,128,31,4]->[11,128,1,4]
    trg_mask = make_std_mask(currt_price,src.size()[1])
    batch_y=batch_y.transpose((0,2,1))                                    #[128, 4, 11] ->#[128,11,4]
    trg_y=torch.tensor(batch_y,dtype=torch.float).cuda()
    out = model.forward(src, currt_price, previous_w,  
                        price_series_mask, trg_mask, padding_price)
    new_w=out[:,:,1:]  #去掉cash
    new_w=new_w[:,0,:]  # #[109,1,11]->#[109,11]
    new_w=new_w.detach().cpu().numpy()
    batch_w(new_w)  
    
    loss, portfolio_value = loss_compute(out,trg_y)           
    return loss, portfolio_value


def test_online(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set_online(DM._test_ind[0], DM._test_ind[-1], x_window_size)
    tst_batch_input = tst_batch["X"]         
    tst_batch_y = tst_batch["y"]              
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).cuda()
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)  

    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))

    long_term_tst_src=torch.tensor(tst_batch_input,dtype=torch.float).cuda()      
#########################################################################################
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1],1,x_window_size)==1)   


    long_term_tst_currt_price=long_term_tst_src.permute((3,1,2,0)) 
    long_term_tst_currt_price=long_term_tst_currt_price[:,:,x_window_size-1:,:]   
###############################################################################################    
    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:,:,0:1,:],long_term_tst_src.size()[1])
   

    tst_batch_y=tst_batch_y.transpose((0,3,2,1))  
    tst_trg_y=torch.tensor(tst_batch_y,dtype=torch.float).cuda()
    tst_long_term_w=[]  
    tst_y_window_size=len(DM._test_ind)-x_window_size-1-1
    for j in range(tst_y_window_size+1): #0-9
        tst_src=long_term_tst_src[:,:,j:j+x_window_size,:]
        tst_currt_price=long_term_tst_currt_price[:,:,j:j+1,:]
        if(local_context_length>1):
            padding_price=long_term_tst_src[:,:,j+x_window_size-1-local_context_length*2+2:j+x_window_size-1,:]
            padding_price=padding_price.permute((3,1,2,0))  #[4, 1, 2, 11] ->[11,1,2,4]
        else:
            padding_price=None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w,  #[109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                        tst_src_mask, tst_trg_mask, padding_price)
        if(j==0):
            tst_long_term_w=out.unsqueeze(0)  #[1,109,1,12] 
        else:
            tst_long_term_w=torch.cat([tst_long_term_w,out.unsqueeze(0)],0)
        out=out[:,:,1:]  #去掉cash #[109,1,11]
        tst_previous_w=out
    tst_long_term_w=tst_long_term_w.permute(1,0,2,3) ##[10,128,1,12]->#[128,10,1,12]
    tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO=evaluate_loss_compute(tst_long_term_w,tst_trg_y)  
    return tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO


def test_net(DM, total_step, output_step, x_window_size, local_context_length, model, loss_compute, evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value=0

    for i in range(total_step):        
        if(is_trn):
            loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):  
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                    (i,loss.item(), portfolio_value.item() , output_step / elapsed))
            start = time.time()
#########################################################tst########################################################   
        tst_total_loss=0
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO = test_online(DM,x_window_size, model, evaluate_loss_compute, local_context_length)
                tst_total_loss += tst_loss.item()                                         
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                        (i, tst_loss.item(), tst_portfolio_value.item() ,SR.item(), CR.item(), TO.item(), 1/elapsed))
                start = time.time()
#                portfolio_value_list.append(portfolio_value.item())
        
                if(tst_portfolio_value>max_tst_portfolio_value):
                    max_tst_portfolio_value=tst_portfolio_value
                    log_SR=SR
                    log_CR=CR
                    log_St_v=St_v
                    log_tst_pc_array=tst_pc_array
    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array,TO



def test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set()
    tst_batch_input = tst_batch["X"]       #(128, 4, 11, 31)
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).cuda()
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)                    #[2426, 1, 11]
    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))
    tst_src=torch.tensor(tst_batch_input,dtype=torch.float).cuda()         
    tst_src_mask = (torch.ones(tst_src.size()[1],1,x_window_size)==1)   #[128, 1, 31]   
    tst_currt_price=tst_src.permute((3,1,2,0))                          #(4,128,31,11)->(11,128,31,3)
#############################################################################
    if(local_context_length>1):
        padding_price=tst_currt_price[:,:,-(local_context_length)*2+1:-1,:]  #(11,128,8,4)
    else:
        padding_price=None
#########################################################################

    tst_currt_price=tst_currt_price[:,:,-1:,:]   #(11,128,31,4)->(11,128,1,4)
    tst_trg_mask = make_std_mask(tst_currt_price,tst_src.size()[1])
    tst_batch_y=tst_batch_y.transpose((0,2,1))   #(128, 4, 11) ->(128,11,4)
    tst_trg_y=torch.tensor(tst_batch_y,dtype=torch.float).cuda()
###########################################################################################################
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w, #[128,1,11]   [128, 11, 31, 4]) 
                    tst_src_mask, tst_trg_mask,padding_price)

    tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y) 
    return tst_loss, tst_portfolio_value


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index, loss_compute,evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value=0
    for i in range(total_step):
        if(is_trn):
            model.train()
            loss, portfolio_value=train_one_step(DM,x_window_size,model,loss_compute,local_context_length)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):  
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                    (i,loss.item(), portfolio_value.item() , output_step / elapsed))
            start = time.time()
#########################################################tst########################################################     
        tst_total_loss=0
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value=test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length)
#                tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y)
                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                        (i,tst_loss.item(), tst_portfolio_value.item() , 1/elapsed))
                start = time.time()
                
                if(tst_portfolio_value>max_tst_portfolio_value):
                    max_tst_portfolio_value=tst_portfolio_value
                    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
#    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
                    print("save model!")
    return tst_loss, tst_portfolio_value



def make_model(batch_size, coin_num, window_size, feature_number,N=6, 
               d_model_Encoder=512,d_model_Decoder=16, d_ff_Encoder=2048, d_ff_Decoder=64, h=8, dropout=0.0,local_context_length=3):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy   
    attn_Encoder = MultiHeadedAttention(True, h, d_model_Encoder,0.1,local_context_length)
    attn_Decoder = MultiHeadedAttention(True, h, d_model_Decoder,0.1,local_context_length)
    attn_En_Decoder = MultiHeadedAttention(False, h, d_model_Decoder,0.1,1)
    ff_Encoder = PositionwiseFeedForward(d_model_Encoder, d_ff_Encoder, dropout)
    ff_Decoder = PositionwiseFeedForward(d_model_Decoder, d_ff_Decoder, dropout)
    position_Encoder = PositionalEncoding(d_model_Encoder,0, dropout)
    position_Decoder = PositionalEncoding(d_model_Decoder, window_size-local_context_length*2+1,dropout)
    
    model = EncoderDecoder(batch_size, coin_num, window_size, feature_number,d_model_Encoder,d_model_Decoder,
        Encoder(EncoderLayer(d_model_Encoder, c(attn_Encoder), c(ff_Encoder), dropout), N),
        Decoder(DecoderLayer(d_model_Decoder, c(attn_Decoder), c(attn_En_Decoder), c(ff_Decoder), dropout), N),
        c(position_Encoder),                  #price series position ecoding
        c(position_Decoder),                  #local_price_context position ecoding
        local_context_length              )    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model             
