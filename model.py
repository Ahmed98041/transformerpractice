import torch 
import torch.nn as nn
import math 

class ImportEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.seq_len = seq_len

       # matrix of seq_len x d_model
        pe = torch.zeros(seq_len, d_model)

        # create a vector of (seq_len, 1) and (1, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # apply the sin to even positions 
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNorm(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #muliplier
        self.bias = nn.Parameter(torch.zeros(1)) #additive

    def forward(self, x):
        mean =  x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FFBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # First Matrix W1 AND B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Second Matrix W2 AND B2

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = heads
        assert d_model % heads == 0, "Embedding dimension must be divisible by number of heads"

        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, d_model) # Query Matrix
        self.w_k = nn.Linear(d_model, d_model) # Key Matrix
        self.w_v = nn.Linear(d_model, d_model) # Value Matrix

        self.w_o = nn.Linear(d_model, d_model) # Output Matrix
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, d_k, mask, dropout=None):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (Batch, heads, Seq_len, d_k) * (Batch, heads, d_k, Seq_len) -> (Batch, heads, Seq_len, Seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (Batch, heads, Seq_len, Seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores # (Batch, heads, Seq_len, Seq_len) * (Batch, heads, Seq_len, d_k) -> (Batch, heads, Seq_len, d_k)
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the embedding into self.h heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # (Batch, Seq_len, d_model) -> (Batch, Seq_len, heads, d_k) -> (Batch, heads, Seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores= MultiHeadAttention.attention(query, key, value, self.d_k, mask, self.dropout)

        # Concatenate the heads
        x = x.transpose(1,2).contiguous().view(x.shape[0], self.h * self.d_k)

        return self.w_o(x) # (Batch, Seq_len, d_model)

class ResConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_foward_block: FFBlock, dropout: float) -> None:
        super().__init__()
        self.feed_forward_block = feed_foward_block
        self.residual_connections = nn.ModuleList([ResConnection(self_attention_block.d_model, dropout) for _ in range(2)])


    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm: LayerNorm) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FFBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResConnection(self_attention_block.d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: ImportEmbeddings, tgt_embed: ImportEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj_layer: ProjLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)
 
class TransformerBuilder:
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 513, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.d_model = d_model
        self.N = N
        self.h = h
        self.dropout = dropout
        self.d_ff = d_ff
    
    def build(self):
        # Create the embedding layers 
        src_embed = ImportEmbeddings(self.src_vocab_size, self.d_model)
        tgt_embed = ImportEmbeddings(self.tgt_vocab_size, self.d_model)
        
        # Create the positional encoding layers
        src_pos = PositionalEncoding(self.d_model, self.src_seq_len, self.dropout)
        tgt_pos = PositionalEncoding(self.d_model, self.tgt_seq_len, self.dropout)

        # Create the encoder 
        encoder_blocks = [EncoderBlock(MultiHeadAttention(self.d_model, self.h, self.dropout), FFBlock(self.d_model, self.d_ff, self.dropout), self.dropout) for _ in range(self.N)]
        
        # Create the decoder
        decoder_blocks = []
        for _ in range(self.N):
            decoder_self_attention_block = MultiHeadAttention(self.d_model, self.h, self.dropout)
            decoder_cross_attention_block = MultiHeadAttention(self.d_model, self.h, self.dropout)
            feed_forward_block = FFBlock(self.d_model, self.d_ff, self.dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, self.dropout)
            decoder_blocks.append(decoder_block)

        # Create the encoder and the decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks), LayerNorm(self.d_model))
        decoder = Decoder(nn.ModuleList(decoder_blocks), LayerNorm(self.d_model))

        # Projection layer
        proj_layer = ProjLayer(self.d_model, self.tgt_vocab_size)
        
        # Create the transformer
        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

        # intialize the parameters 

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return transformer








        
