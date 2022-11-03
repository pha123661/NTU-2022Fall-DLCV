import math

import timm
import torch
from torch import nn


class ImageCaptioningTransformer(nn.Module):
    BOS_Token = 2
    EOS_Token = 3

    def __init__(self, vocab_size, encoder, num_layers, nhead, d_model, activation='gelu', batch_first=True) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)

        self.encoder = timm.create_model(encoder, pretrained=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation=activation,
            batch_first=batch_first
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, batch_image, input_ids):
        # encoder pass
        features = self.encoder.forward_features(
            batch_image)  # shape = (B, 14*14+1, d_model)

        # word embedding + positional embedding
        # 0 ~ n-1 tokens as input
        in_embed = self.word_embedding(input_ids[:, :-1])
        in_embed += self.positional_embedding(in_embed)

        # decoder pass
        mask = self._generate_square_subsequent_mask(
            in_embed.shape[1]).to(in_embed.device)
        logits = self.decoder(tgt=in_embed, memory=features, tgt_mask=mask)
        logits = self.head(logits)  # shape (B, seq_len, vocab)

        # 1 ~ n tokens as target output
        logits = torch.swapaxes(logits, 1, 2)
        loss = self.criterion(logits, input_ids[:, 1:])
        return loss

    def generate_one(self, batch_image):
        # TODO: implement beam search
        batch_size = batch_image.shape[0]
        device = batch_image.device
        gen = torch.ones(batch_size, 1) * self.BOS_token

    # https://github.com/pytorch/examples/blob/5551061414d3bcf202de520d20e8163f58eb664a/word_language_model/model.py#L126
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float(
        ).masked_fill(
            mask == 0, float('-inf')
        ).masked_fill(
            mask == 1, float(0.0)
        )
        return mask


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
