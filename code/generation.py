import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class GreedyOutput:
    sequences: torch.LongTensor
    scores: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    encoder_attentions: Optional[Tuple[torch.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None


class Generator:
    """
    Generator wrapper with optional lookahead for greedy decoding.
    Works for seq2seq models like T5/Flan-T5.
    """

    def __init__(self, model, lookahead=None):
        self.model = model
        self.lookahead = lookahead

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        output_scores: bool = False,
        return_dict_in_generate: bool = True,
        **model_kwargs,
    ):
        eos_token_id = eos_token_id or self.model.config.eos_token_id
        pad_token_id = pad_token_id or self.model.config.pad_token_id
        max_length = max_length or self.model.config.max_length

        batch_size = input_ids.shape[0]

        # ---------------- Encode input once ----------------
        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        # ---------------- Initialize decoder ----------------
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        cur_len = 1
        all_token_scores = []

        # ---------------- Decoding loop ----------------
        while cur_len < max_length and unfinished_sequences.max() > 0:
            # Prepare inputs manually for seq2seq
            model_inputs = {
                "encoder_outputs": encoder_outputs,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
            }

            outputs = self.model(**model_inputs, return_dict=True)

            next_token_logits = outputs.logits[:, -1, :]
            next_tokens_scores = F.log_softmax(next_token_logits, dim=-1)

            # Store scores
            if output_scores:
                all_token_scores.append(next_token_logits)

            # Lookahead scoring
            if self.lookahead is not None:
                lookahead_scores = self.lookahead.score(decoder_input_ids, next_tokens_scores, **model_kwargs)
                next_tokens_scores = next_tokens_scores + lookahead_scores

            # Greedy selection
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Append token to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Update unfinished sequences mask
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
            cur_len += 1

        # ---------------- Return results ----------------
        if return_dict_in_generate:
            return GreedyOutput(
                sequences=decoder_input_ids,
                scores=tuple(all_token_scores) if output_scores else None
            )
        else:
            return decoder_input_ids
