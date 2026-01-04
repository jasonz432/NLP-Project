import copy
import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Lookahead:
    """
    Object that performs lookahead scoring during generation.
    Supports greedy decoding.
    """

    def __init__(self, model, tokenizer, scorer,
                 lookahead_length=1,
                 lookahead_lambda=1.0,
                 lookahead_top_k=5,
                 decoding_type="greedy",
                 max_length=None,
                 pad_token_id=None,
                 eos_token_id=None,
                 output_scores=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 return_dict_in_generate=True):
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer

        self.lookahead_length = max_length if lookahead_length == -1 else lookahead_length
        self.lookahead_lambda = lookahead_lambda
        self.lookahead_top_k = lookahead_top_k
        self.decoding_type = decoding_type

        self.pad_token_id = pad_token_id or getattr(model.config, "pad_token_id", None)
        self.eos_token_id = eos_token_id or getattr(model.config, "eos_token_id", None)
        self.max_length = max_length
        self.output_scores = output_scores
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict_in_generate = return_dict_in_generate

        if decoding_type == "greedy":
            self.decoding_func = self.greedy_search
        else:
            raise ValueError(f"Unsupported decoding_type: {decoding_type}")

        self.logits_processor = LogitsProcessorList()
        self.stopping_criteria = StoppingCriteriaList()

    def score(self, input_ids, next_token_scores, num_beams=1, **model_kwargs):
        batch_size = input_ids.shape[0]

        # Top-k expansion
        _, top_k_indices = torch.topk(next_token_scores, k=self.lookahead_top_k, dim=-1)
        top_k_indices = top_k_indices.reshape(-1)
        indices = torch.arange(batch_size, device=input_ids.device).repeat_interleave(self.lookahead_top_k)
        input_ids_exp = torch.cat([input_ids[indices], top_k_indices.unsqueeze(1)], dim=1)

        model_kwargs_exp = self.expand_model_kwargs(model_kwargs, indices)

        # Run greedy decoding
        dec_out = self.greedy_search(input_ids_exp, **model_kwargs_exp)
        seq = dec_out["sequences"]

        dec_seq = self.tokenizer.batch_decode(seq, skip_special_tokens=True)

        # Compute lookahead scores
        _lookahead_scores = self.scorer.score(dec_seq, torch.div(indices, num_beams, rounding_mode="trunc"))
        _lookahead_scores = torch.clamp(_lookahead_scores, min=1e-9).log()

        lookahead_scores = torch.full_like(next_token_scores, 1e-9).log()
        lookahead_scores[indices, top_k_indices] = _lookahead_scores.view(-1)

        return self.lookahead_lambda * lookahead_scores

    def greedy_search(self, input_ids, **model_kwargs):
        """
        Greedy decoding for T5 without using _update_model_kwargs_for_generation.
        """
        cur_len = input_ids.shape[-1]
        lookahead_length = self.lookahead_length + cur_len

        unfinished_sequences = input_ids.new_ones(input_ids.shape[0], dtype=torch.long)

        while cur_len < lookahead_length:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.model(**model_inputs, return_dict=True)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_tokens = torch.argmax(next_token_scores, dim=-1)

            if self.eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences *= (next_tokens != self.eos_token_id).long()

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            cur_len += 1
            if unfinished_sequences.max() == 0:
                break

        return {"sequences": input_ids}

    def expand_model_kwargs(self, model_kwargs, indices):
        model_kwargs = copy.deepcopy(model_kwargs)
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"][indices]
        if "encoder_outputs" in model_kwargs:
            for k, v in model_kwargs["encoder_outputs"].items():
                if v is not None:
                    model_kwargs["encoder_outputs"][k] = v[indices]
        if "past_key_values" in model_kwargs:
            # Only keep the selected past_key_values
            model_kwargs["past_key_values"] = tuple([tuple([p[indices] for p in layer]) for layer in model_kwargs["past_key_values"]])
        return model_kwargs
