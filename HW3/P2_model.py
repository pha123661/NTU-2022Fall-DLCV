import math

import timm
import torch
from torch import nn

from beam_search import Beam


class ImageCaptioningTransformer(nn.Module):
    BOS_Token = 2
    EOS_Token = 3

    def __init__(self, vocab_size, encoder, num_layers, nhead, d_model, activation='gelu', batch_first=True, dropout=0.1) -> None:
        super().__init__()
        self.config = dict(locals())
        del self.config['self']
        for k in dict(locals()):
            if k.startswith('_'):
                del self.config[k]

        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)

        self.encoder = timm.create_model(encoder, pretrained=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            activation=activation,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
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

    def generate_one(self, img, num_candidates=5, beam_size=8, max_length=30):
        if img.dim() < 4:
            img = img.unsqueeze(0)
        device = img.device
        with torch.no_grad():
            memory = self.encoder.forward_features(img)

        current_state = torch.tensor([self.BOS_Token]).to(device).unsqueeze(1)
        for _ in range(max_length):
            in_embed = self.word_embedding(current_state)
            in_embed += self.positional_embedding(in_embed)

            with torch.no_grad():
                logits = self.decoder(tgt=in_embed, memory=memory)
                logits = self.head(logits[:, -1, :])
            next_word = logits.argmax(dim=-1).unsqueeze(0)
            if next_word.item() == self.EOS_Token:
                break
            current_state = torch.concat((current_state, next_word), dim=-1)
        return current_state[0, 1:].cpu().tolist()  # remove [BOS]

        # memory_beam = memory.detach().repeat(beam_size, 1, 1)
        # beam = Beam(
        #     beam_size=beam_size,
        #     min_length=0,
        #     n_top=num_candidates,
        #     ranker=None,
        # )

        # for _ in range(max_length):
        #     new_input_ids = beam.get_current_state().unsqueeze(1)
        #     in_embed = self.word_embedding(new_input_ids)
        #     in_embed += self.positional_embedding(in_embed)
        #     with torch.no_grad():
        #         decoder_outputs = self.decoder(
        #             tgt=in_embed, memory=memory_beam)
        #         # decoder_outputs.shape = (B, seq_len, vocab)
        #     print(
        #         self.decoder.layers)
        #     break


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


if __name__ == "__main__":
    from pathlib import Path

    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from tokenizers import Tokenizer

    from ICDataset import ICDataset
    Transformer = ImageCaptioningTransformer(
        vocab_size=18022,
        encoder="vit_base_patch16_224",
        num_layers=4,
        nhead=12,
        d_model=768,
    )

    transform = create_transform(
        **resolve_data_config({}, model="vit_base_patch16_224"))
    tokenizer = Tokenizer.from_file('./hw3_data/caption_tokenizer.json')
    train_set = ICDataset(
        image_dir=Path('hw3_data/p2_data/images/train'),
        json_file=Path('hw3_data/p2_data/train.json'),
        transform=transform,
        tokenizer=tokenizer
    )

    data = next(iter(train_set))
    seq = Transformer.generate_one(data['image'])
    print(seq)
    print(tokenizer.decode(seq))
