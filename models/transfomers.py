import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_size,
                 n_layers,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device,
                 max_length=100):

        super(Encoder,self).__init__()

        self.device = device

        self.token_embedding = nn.Embedding(input_dim,hidden_size)
        self.pos_embedding = nn.Embedding(max_length,hidden_size)

        self.layers = nn.ModuleList([EncoderLayer(hidden_size,
                                                  n_heads,
                                                  pf_dim,
                                                  drop_out,
                                                  device) for _ in range(n_layers)])
        
        self.drop_out = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
    
    def forward(self,src,src_mask):

        #src =[batch_size, src_len]
        #src_mask = [batch_size,src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(self.device)

        #pos = [batch_size, src_len]

        src = self.drop_out((self.token_embedding(src)*self.scale) + self.pos_embedding(pos))

        #src = [batch_size, src_len, hidden_size]

        for layer in self.layers:

            src = layer(src, src_mask)
        
        return src

class EncoderLayer(nn.Module):

    def __init__(self,
                 hidden_size,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device):

        super(EncoderLayer,self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size,
                                                      n_heads,
                                                      drop_out,
                                                      device)

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_size, 
                                                                     pf_dim, 
                                                                     drop_out)
        
        self.drop_out = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)

    
    def forward(self,src,src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention

        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.drop_out(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.layer_norm(src + self.drop_out(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self,
                 hidden_size,
                 n_heads,
                 drop_out,
                 device):

        super(MultiHeadAttentionLayer,self).__init__()

        assert hidden_size % n_heads ==0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.fc_q = nn.Linear(hidden_size,hidden_size)
        self.fc_k = nn.Linear(hidden_size,hidden_size)
        self.fc_v = nn.Linear(hidden_size,hidden_size)

        self.fc_o = nn.Linear(hidden_size,hidden_size)

        self.drop_out = nn.Dropout(drop_out)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    
    def forward(self,query,key,value,mask = None):
        
        batch_size = query.shape[0]

        #query =[batch_size, query_len, hidden_size]
        #key = [batch_size,key_len,hidden_size]
        #value = [batch_size,value_len,hidden_size]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q =[batch_size,query_len,hidden_size]
        # K = [batch_size, key_len,hidden_size]
        # V=[batch_size,value_len,hidden_size]

        Q = Q.view(batch_size, -1 , self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query_len, key_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy,dim=-1)

        #attention = [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.drop_out(attention), V)

        #x= [batch_size, n_heads, seq_len, head_dim]

        x = x.permute(0,2,1,3).contiguous()

        #x = [batch size, seq len, n heads, head dim]

        x = x.view(batch_size, -1, self.hidden_size)

        #x = [batch size, seq len, hid dim]

        x= self.fc_o(x)

        #x = [batch_size, seq_len, hidden_size]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):


      def __init__(self,hidden_size,pf_dim,drop_out):

          super(PositionwiseFeedforwardLayer,self).__init__()

          self.fc_1 = nn.Linear(hidden_size,pf_dim)
          self.fc_2 = nn.Linear(pf_dim,hidden_size)

          self.drop_out = nn.Dropout(drop_out)
      
      def forward(self,x):

          #x = [batch_size,seq_len,hidden_size]

          x = self.drop_out(torch.relu(self.fc_1(x)))

          # x= [batch_size, seq_len, pf_dim]

          x = self.fc_2(x)

          #x = [batch_size, seq_len, hidden_size]

          return x

class Decoder(nn.Module):

    def __init__(self,
                 output_dim,
                 hidden_size,
                 n_layers,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device,
                 max_length=100):

        super().__init__()
        self.device = device
        self.token_embedding = nn.Embedding(output_dim,hidden_size)
        self.pos_embedding = nn.Embedding(max_length,hidden_size)

        self.layers = nn.ModuleList([DecoderLayer(hidden_size, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  drop_out, 
                                                  device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_size,output_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)


    def forward(self,trg,enc_src,trg_mask,src_mask):

        #rtrg =[batch_size, trg_len]
        #enc_src =[batch_size, src_len, hidden_size]
        #trg_mask = [batch_size,trg_len]
        #src_mask = [batch_size, src_Len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).to(self.device)

        #pos = [batch_size,trg_len]

        trg = self.drop_out((self.token_embedding(trg)* self.scale)+ self.pos_embedding(pos))

        #trg= [batch_size,trg_len,hidden_size]

        for layer in self.layers:

            trg,attention = layer(trg,enc_src,trg_mask,src_mask)
        
        #trg =[batch_size,trg_len,hidden_size]
        #attention = [batch_size,n_heads,trg_len,src_len]

        output = self.fc_out(trg)

        #output = [batch_size,trg_len, output_dim]

        return output, attention

class DecoderLayer(nn.Module):

    def __init__(self,
                 hidden_size,
                 n_heads,
                 pf_dim,
                 drop_out,
                 device):

        super(DecoderLayer,self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size,
                                                      n_heads,
                                                      drop_out,
                                                      device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_size,
                                                         n_heads,
                                                         drop_out,
                                                         device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_size,
                                                                     pf_dim,
                                                                     drop_out)
        
        self.drop_out = nn.Dropout(drop_out)
    
    def forward(self,trg,enc_src,trg_mask,src_mask):

        #trg =[batch_size,trg_len,hidden_size]
        #enc_src =[batch_size, src_len,hidden_size]
        #trg_mask =[batch_size,trg_len]
        #src_mask = [batch_size,src_len]

        #self attention

        _trg,_ = self.self_attention(trg,trg,trg,trg_mask)
        #drop_out, residual connection and layer norm

        trg = self.layer_norm(trg + self.drop_out(_trg))

        #trg = [batch_size, trg_len, hidden_size]

        #encoder attention

        _trg, attention = self.encoder_attention(trg,enc_src,enc_src,src_mask)

        #drop_out, residual connection and layer norm

        trg = self.layer_norm(trg + self.drop_out(_trg))

        #trg = [batch_size,trg_len,hidden_size]

        #positionwise feedforward

        trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm

        trg = self.layer_norm(trg + self.drop_out(_trg))

        #trg = [batch_size,trg_len,hidden_size]
        #attention = [batch_size,n_heads,trg_len,src_len]

        return trg,attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        
        #trg_pad_mask = [batch size, 1, trg len, 1]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention