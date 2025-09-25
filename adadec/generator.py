import logging
import torch
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import os
import json
from typing import Union
import re



class Generator:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            model_name: str,
            beam_size: int = 3,
            decoding_mode: str = 'Traditional',
            entropy_threshold: Union[str, float]='Learned',
            stop_words_file: str = "data/stop_words.json"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.beam_size = beam_size
        self.tradition_times = 0
        self.lookahead_times = 0
        self.lookahead_beam_size = 3
        self.generation_counter = 0
        self.decoding_mode = decoding_mode
        
        if os.path.exists(stop_words_file):
            with open(stop_words_file, "r", encoding="utf-8") as f:
                raw_stop_words = json.load(f)
            if not isinstance(raw_stop_words, list):
                raise ValueError(f"stop_words.json format error: should be a list, but got {type(raw_stop_words)}")

            self.stop_words = []
            for sw in raw_stop_words:
                try:
                    self.stop_words.append(re.compile(sw, re.MULTILINE))
                except re.error as e:
                    raise ValueError(f"Invalid regular expression: {sw}, error: {e}")
        else:
            logging.warning(f"stop_words file {stop_words_file} not found, using an empty list.")
            self.stop_words = []

        self.entropy_threshold = None
        if decoding_mode == 'Traditional':
            self.entropy_threshold = float('inf')
        elif decoding_mode == 'AdaFixL':
            if entropy_threshold == 'Learned':
                self.entropy_threshold = self._load_learned_threshold("data/learned_thresholds.json")
            else:
                try:
                    self.entropy_threshold = float(entropy_threshold)
                except ValueError:
                    raise ValueError("Entropy threshold must be a number or 'Learned'")
        else:
            raise ValueError(f"Unsupported decoding_mode: expected 'AdaFixL' , got '{self.decoding_mode}'")


    def _load_learned_threshold(self, threshold_file) -> float:
        if not os.path.exists(threshold_file):
            raise FileNotFoundError(f"Entropy threshold file '{threshold_file}' not found.")
        
        with open(threshold_file, 'r') as f:
            threshold_dict = json.load(f)

        if self.model_name not in threshold_dict:
            raise KeyError(f"Model '{self.model_name}' not found in entropy threshold file.")
        
        return threshold_dict[self.model_name]

    def calculate_entropy(self, next_token_logits):
        next_token_probs_exp = torch.nn.functional.softmax(next_token_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        return -torch.sum(next_token_probs_exp * log_probs, dim=-1)

    def select_top_beam_scores(self, beam_size, topk_scores, topk_indices, mode):
        
        if mode == 'Traditional':
            if isinstance(topk_scores, list):
                topk_scores = torch.cat(topk_scores, dim=0)
                topk_indices = torch.cat(topk_indices, dim=0)

            total_candidates = topk_scores.size(0)

            if total_candidates == beam_size:
                selected_groups = torch.arange(beam_size, dtype=torch.long, device=topk_scores.device)
                return topk_scores, topk_indices, selected_groups

            elif total_candidates == beam_size * beam_size:
                final_scores, flat_indices = torch.topk(topk_scores, beam_size)  # [beam_size]

                selected_groups = flat_indices // beam_size
                token_pos_in_group = flat_indices % beam_size

                final_indices = []
                for group, pos in zip(selected_groups, token_pos_in_group):
                    index = group * beam_size + pos
                    final_indices.append(topk_indices[index])
                final_indices = torch.stack(final_indices)

                return final_scores, final_indices, selected_groups

            else:
                raise ValueError(f"Unsupported topk_scores size: expected {beam_size} or {beam_size * beam_size}, got {total_candidates}")
        
        elif mode == 'AdaFixL':
            total_candidates = topk_scores.size(0)
            assert total_candidates % beam_size == 0, "topk_scores size must be divisible by beam_size"
            
            batch_size = total_candidates // beam_size

            # reshape to [batch_size, beam_size]
            topk_scores = topk_scores.view(batch_size, beam_size)
            topk_indices = topk_indices.view(batch_size, beam_size)

            final_scores, local_indices = torch.topk(topk_scores, beam_size, dim=-1)  # [batch_size, beam_size]

            batch_indices = torch.arange(batch_size).unsqueeze(1).to(topk_indices.device)  # [batch_size, 1]
            final_indices = topk_indices[batch_indices, local_indices]  # [batch_size, beam_size]

            selected_groups = batch_indices.expand(-1, beam_size)

            return final_scores, final_indices, selected_groups
        
        else:
           raise ValueError(f"Unsupported decoding_mode: got {self.decoding_mode}")

    def scoring_function(self, next_token_logits, beam_scores, beam_size):
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  # [batch, vocab]
        topk_scores, topk_indices = torch.topk(log_probs, beam_size, dim=-1)    # [batch, beam_size]

        if beam_scores.dim() == 1:
            beam_scores = beam_scores.unsqueeze(1).expand(-1, beam_size)

        topk_scores = topk_scores + beam_scores

        topk_scores = topk_scores.view(-1)
        topk_indices = topk_indices.view(-1)

        return topk_scores, topk_indices
    
    def generate(
            self,
            prompt,
            beam_size=1,
            max_new_tokens=512,
            lambda_value=1.0,
            lookahead_length=5,
            lookahead_beam_size=3,
    ):
        self.beam_size = beam_size
        self.lookahead_beam_size = lookahead_beam_size
        
        self.tradition_times = 0
        self.lookahead_times = 0

        token_ids = self.tokenizer([prompt], add_special_tokens=True, padding=True, truncation=True,
                                   return_tensors="pt").input_ids
        token_ids = token_ids.to(self.model.device)
        
        decoded_prompt = self.tokenizer.decode(
            self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.model.device),
            skip_special_tokens=True
        )

        token_ids = token_ids.repeat(beam_size, 1)
        attention_mask = torch.ones_like(token_ids).to(self.model.device)

        beam_scores = torch.zeros(beam_size, dtype=torch.float).to(self.model.device)

        is_finished = [False] * beam_size

        beam_indices = torch.zeros(beam_size, dtype=torch.long, device=self.model.device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                if all(is_finished):
                    break

                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                topk_k_scores = []
                topk_k_indices = []

                entropy = self.calculate_entropy(next_token_logits)

                for i in range(self.beam_size):
                    if entropy[i] < self.entropy_threshold or is_finished[i]:
                        curr_topk_scores, curr_topk_indices = self.scoring_function(next_token_logits[i],
                                                                                    beam_scores[i],
                                                                                    beam_size=beam_size)
                        topk_k_scores.append(curr_topk_scores)
                        topk_k_indices.append(curr_topk_indices)
                        self.tradition_times += 1
                    else:
                        curr_topk_scores, curr_topk_indices = self.lookahead_scoring_function(
                            decoded_prompt,
                            next_token_logits[i],
                            token_ids[i],
                            beam_scores[i],
                            lookahead_length=lookahead_length,
                            lambda_value=lambda_value,
                        )
                        topk_k_scores.append(curr_topk_scores)
                        topk_k_indices.append(curr_topk_indices)
                        self.lookahead_times += 1

                if not topk_k_scores:
                    break

                if step == 0:
                    topk_scores = topk_k_scores[0]
                    topk_indices = topk_k_indices[0]
                else:
                    topk_scores, topk_indices, beam_indices = self.select_top_beam_scores(
                        beam_size=beam_size,
                        topk_scores=topk_k_scores,
                        topk_indices=topk_k_indices,
                        mode='Traditional'
                    )

                token_indices = topk_indices % next_token_logits.shape[-1]

                token_ids = torch.cat([
                    token_ids[beam_indices],
                    token_indices.unsqueeze(-1)
                ], dim=-1)
                beam_scores = topk_scores

                for j in range(beam_size):
                    if token_indices[j] == self.tokenizer.eos_token_id:
                        is_finished[j] = True
                        continue

                    prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.model.device)
                    decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    decoded_seq = self.tokenizer.decode(token_ids[j], skip_special_tokens=True)
                    current_sequence = decoded_seq[len(decoded_prompt):]
                    
                    is_stop = any(pattern.search(current_sequence) for pattern in self.stop_words)

                    if is_stop:
                        is_finished[j] = True

                attention_mask = token_ids.ne(self.tokenizer.pad_token_id).to(self.model.device)

                torch.cuda.empty_cache()

        decoded_sequences = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in token_ids
        ]
        

        torch.cuda.empty_cache()

        return [gen[len(prompt):] for gen in decoded_sequences]

    def lookahead_scoring_function(self, decoded_prompt, next_token_logits, token_ids, beam_scores,
                                lookahead_length, lambda_value):
        lookahead_beam_size = self.lookahead_beam_size
        beam_size = self.beam_size

        history_topk_score = beam_scores

        topk_scores, topk_indices = self.scoring_function(
            next_token_logits, beam_scores, beam_size=lookahead_beam_size)

        current_topk_scores = topk_scores - history_topk_score
        token_indices = topk_indices % next_token_logits.shape[-1]

        token_ids = token_ids.repeat(lookahead_beam_size, 1)
        token_ids = torch.cat([token_ids, token_indices.unsqueeze(-1)], dim=-1)

        lookahead_scores, actual_lengths = self.get_lookahead_score(
            token_ids=token_ids,
            lookahead_length=lookahead_length,
            decoded_prompt=decoded_prompt
        )
        total_scores = history_topk_score + (current_topk_scores + lookahead_scores) / (actual_lengths + 1) * lambda_value

        topk_scores, topk_indices_temp = torch.topk(total_scores, beam_size)
        topk_indices = topk_indices[topk_indices_temp]

        return topk_scores, topk_indices

    def get_lookahead_score(self, token_ids, lookahead_length, decoded_prompt):
        batch_size = token_ids.shape[0]
        beam_size = self.beam_size

        device = self.model.device

        token_ids = token_ids.repeat_interleave(beam_size, dim=0)
        attention_mask = torch.ones_like(token_ids).to(device)

        lookahead_scores = torch.zeros(token_ids.size(0), dtype=torch.float).to(device)
        is_finished = torch.zeros(token_ids.size(0), dtype=torch.bool).to(device)
        actual_lookahead_length = torch.zeros(token_ids.size(0), dtype=torch.long).to(device)

        with torch.no_grad():
            for _ in range(lookahead_length):
                if is_finished.all():
                    break

                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                topk_scores, topk_indices = self.scoring_function(next_token_logits, lookahead_scores, beam_size=beam_size)

                selected_scores, selected_indices, beam_idx = self.select_top_beam_scores(
                    beam_size=beam_size,
                    topk_scores=topk_scores,
                    topk_indices=topk_indices,
                    mode='AdaFixL'
                )

                lookahead_scores = selected_scores
                actual_lookahead_length += (~is_finished).long()
                next_tokens = selected_indices % next_token_logits.shape[-1]

                token_ids = torch.cat([
                    token_ids[beam_idx],
                    next_tokens.unsqueeze(-1)
                ], dim=-1)
                
                if token_ids.ndim == 1:
                    token_ids = token_ids.unsqueeze(0) 
                else:
                    token_ids = token_ids.view(-1, token_ids.shape[-1])

                decoded_seqs = self.tokenizer.batch_decode(token_ids.tolist(), skip_special_tokens=True)

                for i in range(token_ids.size(0)):
                    if is_finished[i]:
                        continue
                    current_seq = decoded_seqs[i][len(decoded_prompt):]
                    lines = current_seq.split('\n')
                    is_consecutive_empty = len(lines) >= 4 and all(line.strip() == '' for line in lines[-4:])
                    all_lines_valid = all(not line or line.startswith((' ', '\t')) for line in lines)
                    is_finished[i] = is_consecutive_empty or not all_lines_valid

                attention_mask = token_ids.ne(self.tokenizer.pad_token_id).to(device)

        lookahead_scores = lookahead_scores.view(batch_size, beam_size)
        actual_lookahead_length = actual_lookahead_length.view(batch_size, beam_size)

        max_scores, _ = lookahead_scores.max(dim=1)
        max_lengths, _ = actual_lookahead_length.max(dim=1)

        return max_scores, max_lengths

    def generate_base_on_ground_truth(
            self,
            prompt,
            ground_truth,
            filename,
    ):
        generation_id = self.generation_counter
        self.generation_counter += 1

        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gt_ids = self.tokenizer(ground_truth, return_tensors="pt").input_ids[0].to(device)
        current_sequence = prompt_ids

        log_data = []

        for i, gt_token in enumerate(gt_ids):
            with torch.no_grad():
                outputs = self.model(current_sequence)
                logits = outputs.logits
                next_logits = logits[0, -1, :]
            # topk_values, topk_indices = torch.topk(next_logits, k=50)
            entropy = self.calculate_entropy(next_logits)
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            rank_tensor = (sorted_indices == gt_token).nonzero()
            rank = rank_tensor.item() + 1 if rank_tensor.numel() > 0 else -1

            log_data.append({
                "GenerationID": generation_id,
                "Iteration": i,
                "Entropy": float(entropy),
                "GroundTruthToken": gt_token.item(),
                "Rank": rank,
                # "Top50_TokenIDs": topk_indices.tolist(),
                # "Top50_TokenLogits": topk_values.tolist()
            })

            gt_token_id = gt_token.unsqueeze(0).unsqueeze(0).to(device)
            current_sequence = torch.cat([current_sequence, gt_token_id], dim=1)

            torch.cuda.empty_cache()

        new_df = pd.DataFrame(log_data)

        if os.path.exists(filename):
            old_df = pd.read_parquet(filename, engine="pyarrow")
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.to_parquet(filename, engine="pyarrow", index=False)
        else:
            new_df.to_parquet(filename, engine="pyarrow", index=False)