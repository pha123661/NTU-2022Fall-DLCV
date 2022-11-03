import torch


# https://github.com/dreamgonfly/transformer-pytorch
class Beam:

    def __init__(self, beam_size=8, min_length=0, n_top=1, ranker=None,
                 start_token_id=2, end_token_id=3):
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(
            start_token_id)]  # remove padding

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # The attentions (matrix) for each time.
        self.all_attentions = []

        self.finished = []

        # Time and k pair for finished.
        self.finished = []
        self.n_top = n_top

        self.ranker = ranker

    def advance(self, next_log_probs, current_attention):
        # next_probs : beam_size X vocab_size
        # current_attention: (target_seq_len=1, beam_size, source_seq_len)

        vocabulary_size = next_log_probs.size(1)
        # current_beam_size = next_log_probs.size(0)

        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -1e10

        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + \
                self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            # Don't let EOS have children.
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    # -1e20 raises error when executing
                    beam_scores[beam_index] = -1e10
        else:
            beam_scores = next_log_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(
            k=self.beam_size, dim=0, largest=True, sorted=True)

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)

        prev_k = top_score_ids / vocabulary_size  # (beam_size, )
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)
        # for RNN, dim=1 and for transformer, dim=0.
        # (target_seq_len=1, beam_size, source_seq_len)
        prev_attention = current_attention.index_select(dim=0, index=prev_k)
        self.all_attentions.append(prev_attention)

        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id:
                # skip scoring
                self.finished.append(
                    (self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        hypothesis, attentions = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            # for RNN, [:, k, :], and for trnasformer, [k, :, :]
            attentions.append(self.all_attentions[j][k, :, :])
            k = self.prev_ks[j][k]
        attentions_tensor = torch.stack(
            attentions[::-1]).squeeze(1)  # (timestep, source_seq_len)
        return hypothesis[::-1], attentions_tensor

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks


# Usage
class Predictor:

    def __init__(self, preprocess, postprocess, model, checkpoint_filepath, max_length=30, beam_size=8):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size

        self.model.eval()
        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def predict_one(self, source, num_candidates=5):
        source_preprocessed = self.preprocess(source)
        source_tensor = torch.tensor(
            source_preprocessed).unsqueeze(0)  # why unsqueeze?
        length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)

        memory = self.model.encoder(source_tensor)

        decoder_state = self.model.decoder.init_decoder_state()
        # print('decoder_state src', decoder_state.src.shape)
        # print('previous_input previous_input', decoder_state.previous_input)
        # print('previous_input previous_layer_inputs ', decoder_state.previous_layer_inputs)

        # Repeat beam_size times
        # (beam_size, seq_len, hidden_size)
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)

        beam = Beam(beam_size=self.beam_size, min_length=0,
                    n_top=num_candidates, ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            decoder_outputs, decoder_state = self.model.decoder(new_inputs, memory_beam,
                                                                state=decoder_state)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            attention = self.model.decoder.decoder_layers[-1].memory_attention_layer.sublayer.attention
            beam.advance(decoder_outputs.squeeze(1), attention)

            beam_current_origin = beam.get_current_origin()  # (beam_size, )
            decoder_state.beam_update(beam_current_origin)

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=num_candidates)
        hypothesises, attentions = [], []
        for i, (times, k) in enumerate(ks[:num_candidates]):
            hypothesis, attention = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
            attentions.append(attention)

        self.attentions = attentions
        self.hypothesises = [[token.item() for token in h]
                             for h in hypothesises]
        hs = [self.postprocess(h) for h in self.hypothesises]
        return list(reversed(hs))
