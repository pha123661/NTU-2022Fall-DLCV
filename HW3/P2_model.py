import math

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ImageCaptioningTransformer(nn.Module):
    PAD_Token = 0
    UNK_Token = 1
    BOS_Token = 2
    EOS_Token = 3

    def __init__(self, vocab_size, encoder, num_layers, nhead, d_model, activation='gelu', batch_first=True, dropout=0.1, pretrained=True) -> None:
        super().__init__()
        self.config = dict(locals())
        del self.config['self']
        for k in dict(locals()):
            if k.startswith('_'):
                del self.config[k]

        self.word_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=self.PAD_Token)
        self.positional_embedding = PositionalEmbedding(d_model=d_model)

        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, num_classes=0)
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_Token)

    def forward(self, batch_image, input_ids, sampleing_ratio=0):
        # encoder pass
        features = self.encoder.forward_features(
            batch_image)  # shape = (B, 14*14+1, d_model)

        if sampleing_ratio > 0:
            with torch.no_grad():
                in_embed = self.word_embedding(input_ids[:, :-1])
                in_embed += self.positional_embedding(in_embed)
                mask = self._generate_square_subsequent_mask(
                    in_embed.shape[1]).to(in_embed.device)
                logits = self.decoder(
                    tgt=in_embed, memory=features, tgt_mask=mask)
                logits = self.head(logits)  # shape (B, seq_len, vocab)
            pred_ids = logits.argmax(dim=-1)  # (B, seq_len)
            # place BOS in pred
            pred_ids = torch.cat(
                [self.BOS_Token * torch.ones(pred_ids.shape[0], 1, dtype=input_ids.dtype).to(input_ids.device), pred_ids], dim=1)

            # replace input_ids
            '''
            loop-style:
            for batch_idx in range(input_ids.shape[0]):
                for seq_idx in range(1, input_ids.shape[1]):
                    if input_ids[batch_idx][seq_idx] == self.EOS_Token:  # EOS
                        break
                    if random.random() > sampleing_ratio:
                        # pred_ids[B][0] = output of 0th word -> replaces the 1st word
                        input_ids[batch_idx][seq_idx] = pred_ids[batch_idx][seq_idx]
            '''
            to_be_replaced = torch.rand_like(
                pred_ids, dtype=float) > sampleing_ratio
            # don't replace PAD and EOS
            to_be_replaced = torch.logical_and(
                to_be_replaced,
                input_ids != self.PAD_Token
            )
            to_be_replaced = torch.logical_and(
                to_be_replaced,
                input_ids != self.EOS_Token
            )
            input_ids[to_be_replaced] = pred_ids[to_be_replaced]

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

    def greedy_search(self, img, max_length=30):
        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        device = img.device
        with torch.no_grad():
            memory = self.encoder.forward_features(img)

        current_state = torch.tensor([self.BOS_Token]).to(device).unsqueeze(1)
        for _ in range(max_length):
            with torch.no_grad():
                in_embed = self.word_embedding(current_state)
                in_embed += self.positional_embedding(in_embed)
                logits = self.decoder(tgt=in_embed, memory=memory)
                logits = self.head(logits[:, -1, :])
            next_word = logits.argmax(dim=-1).unsqueeze(0)
            if next_word.item() == self.EOS_Token:
                break
            current_state = torch.concat((current_state, next_word), dim=-1)
        return current_state[0, 1:].cpu().tolist()  # remove [BOS]

    @torch.no_grad()
    def batch_beam_search(self, img, beams=3, max_length=30):
        def forward_prob(memory, input_ids):
            in_embed = self.word_embedding(input_ids)
            in_embed += self.positional_embedding(in_embed)
            logits = self.decoder(tgt=in_embed, memory=memory)
            logits = self.head(logits[:, -1, :])
            return logits

        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        memory = self.encoder.forward_features(img)

        device = memory.device
        current_state = self.BOS_Token * \
            torch.ones(memory.shape[0], 1, dtype=torch.long).to(device)

        next_probs = forward_prob(memory, current_state)
        vocab_size = next_probs.shape[-1]
        probs, next_chars = \
            next_probs.reshape(next_probs.shape[0], next_probs.shape[-1])\
            .log_softmax(-1)\
            .topk(k=beams, axis=-1)  # shape = (B, topk)
        current_state = current_state.repeat((beams, 1))
        next_chars = next_chars.reshape(-1, 1)
        current_state = torch.cat((current_state, next_chars), axis=1)
        memory = memory.repeat((beams, 1, 1))

        for i in range(2, max_length + 1):
            next_probs = forward_prob(memory, current_state).log_softmax(-1)
            next_probs = next_probs.reshape((-1, beams, next_probs.shape[-1]))
            probs = probs.unsqueeze(-1) + next_probs
            probs = probs.flatten(start_dim=1)
            probs, idx = probs.topk(k=beams, axis=-1)
            next_chars = torch.remainder(
                idx, vocab_size).flatten().unsqueeze(-1)
            top_candidates = (idx / vocab_size).long()
            top_candidates += torch.arange(current_state.shape[0] //
                                           beams, device=device).unsqueeze(-1) * beams
            current_state = current_state[top_candidates].flatten(end_dim=-2)
            current_state = torch.cat([current_state, next_chars], axis=1)

        current_state = current_state.reshape(-1,
                                              beams, current_state.shape[-1])
        current_state = current_state[..., 1:]  # remove BOS
        # get best Y
        best_idx = probs.argmax(-1)
        current_state = current_state[torch.arange(len(best_idx)), best_idx, :]
        exit()
        return current_state.cpu().tolist()

    # reference: https://github.com/jarobyte91/pytorch_beam_search
    @torch.no_grad()
    def beam_search(self, img, beams=3, max_length=30):
        def forward_prob(memory, input_ids):
            in_embed = self.word_embedding(input_ids)
            in_embed += self.positional_embedding(in_embed)
            logits = self.decoder(tgt=in_embed, memory=memory)
            logits = self.head(logits[:, -1, :])
            return logits

        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        memory = self.encoder.forward_features(img)
        device = memory.device
        current_state = torch.tensor([self.BOS_Token]).to(device).unsqueeze(1)

        # get top k words
        next_probs = forward_prob(memory, current_state)
        vocab_size = next_probs.shape[-1]
        current_probs, next_chars = next_probs.log_softmax(
            -1).topk(k=beams, axis=-1)
        current_probs = current_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)

        # gen first k beams
        current_state = current_state.repeat((beams, 1))
        current_state = torch.cat((current_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            # get top k beams for beam*beam candidates
            next_probs = forward_prob(memory.repeat(
                (beams, 1, 1)), current_state).log_softmax(-1)
            current_probs = current_probs.unsqueeze(-1) + next_probs
            current_probs = current_probs.flatten()  # (beams*vocab)

            # length normalization
            _, idx = (current_probs / (len(current_state[0]) + 1))\
                .topk(k=beams, dim=-1)
            current_probs = current_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()
            current_state = current_state[top_candidates]
            current_state = torch.cat((current_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == self.EOS_Token:
                    ans_ids.append(current_state[idx].cpu().tolist())
                    ans_probs.append(
                        current_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1
            to_keep_idx = [i for i in range(
                len(current_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            current_state = current_state[to_keep_idx]
            current_probs = current_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()
        return ans_ids[max_idx]


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
    from torch.utils.data import DataLoader

    from ICDataset import ICDataset
    Transformer = ImageCaptioningTransformer(
        vocab_size=18022,
        encoder="beitv2_large_patch16_224_in22k",
        num_layers=12,
        nhead=16,
        d_model=1024,
        dropout=0.1,
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

    td = DataLoader(train_set, 8, collate_fn=train_set.collate_fn)
    data = next(iter(td))
    seq = Transformer.Beam_search(
        data['images'], max_length=6, return_probs=True)
    print(seq)
