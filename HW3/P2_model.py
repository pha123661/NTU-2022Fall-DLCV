import torch
from torch import nn

import timm


class ImageCaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, encoder, num_layers, nhead, d_model, activation='gelu', batch_first=True) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
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
        # batch_image: (B, 3, 224, 224)
        features = self.encoder.forward_features(
            batch_image)  # (B, 14*14+1, d_model)

        embeds = self.word_embedding(input_ids)

        in_embed = embeds[:, :-1, :]
        logits = self.decoder(tgt=in_embed, memory=features)
        logits = self.head(logits)  # shape (B, seq_len, vocab)
        logits = torch.swapaxes(logits, 1, 2)
        loss = self.criterion(logits, input_ids[:, 1:])
        return loss
