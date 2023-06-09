# transformerpractice
This project involves the creation of a customized transformer model from scratch using PyTorch, a popular deep learning library. The transformer model is a type of deep learning model that uses self-attention mechanisms and has become a state-of-the-art choice for a variety of tasks in natural language processing, such as translation, summarization, and entity recognition.

This implementation provides modularized transformer components that can be reused and combined, offering a clear structure and high flexibility. It includes:

ImportEmbeddings: This module is used for converting input tokens into their corresponding embeddings.

PositionalEncoding: A module that applies positional encodings to the inputs, which are necessary because the self-attention mechanism does not have any inherent sense of position/order.

LayerNorm: This is the layer normalization module, a type of normalization technique that normalizes the inputs across the features.

MultiHeadAttention: This module implements the multi-head attention mechanism, a key component of the transformer model that allows the model to focus on different words in the input sequence.

ResConnection: This module implements the residual connection, another vital component of the transformer architecture.

EncoderBlock and DecoderBlock: These are the fundamental building blocks of the transformer model, containing the self-attention and feed-forward networks.

Encoder and Decoder: These are the primary components of the transformer model, with the encoder processing the input sequence and the decoder generating the output sequence.

ProjLayer: This is the final projection layer, mapping the output of the decoder to the target vocabulary size.

Transformer: The main transformer model that combines all the previous components.

TransformerBuilder: A convenient builder class that facilitates the creation of a transformer model with custom parameters.




        

![transformer arch](https://github.com/Ahmed98041/transformerpractice/assets/45014346/2c30d5cc-f466-4789-8b41-fbbbd36e4b4f)
[attention.pdf](https://github.com/Ahmed98041/transformerpractice/files/11705179/attention.pdf)
