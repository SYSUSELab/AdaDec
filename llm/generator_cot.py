import logging
import torch
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import os
import json
import re




# few_shot_prompt = """
# # RULE:
# # When a line containing "# Thinking:" appears, the model must first continue that line by adding a brief "thinking comment" starting with "# Thinking:" on the same line, summarizing the upcoming code logic in one line, and then proceed to write the corresponding code line.
# # The model must never generate "# Thinking:" on its own. Only when "# Thinking:" already exists in the input should it output a thinking comment line starting with "# Thinking:". Otherwise, just generate normal code.
# # "Thinking comments" should always start at the beginning of the line with no indentation; code should retain normal indentation.
# # Do not generate extra explanations or blank lines.


# def factorial(n):
#     \"\"\"
#     Write a function to compute the factorial of a number.
#     \"\"\"
# # Thinking: Base case: factorial(0) = 1
#     if n == 0:
#         return 1
# # Thinking: Recursive case: n * factorial(n-1)
#     return n * factorial(n-1)

# ---

# def is_prime(n):
#     \"\"\"
#     Write a function to check if a number is prime.
#     \"\"\"
# # Thinking: A prime number is greater than 1
#     if n < 2:
#         return False
# # Thinking: Check divisibility up to sqrt(n)
#     for i in range(2, int(n**0.5) + 1):
#         if n % i == 0:
#             return False
#     return True

# ---

# # CONTEXT

# {}
# # Thinking: 
# """

# -------------------------------
# 函数一：构造 UnCert-CoT prompt
# -------------------------------
def build_uncert_cot_prompt(context: str) -> str:
    """
    给定当前上下文代码（context），构造触发 UnCert-CoT 的完整 prompt。
    """
    RULE = """RULE:
1) When "# Thinking:" appears in the input, the model must not wrap. The model must continue that line with exactly ONE short comment (<= 15 words), all on the SAME line, starting with "# Thinking:".
2) After finishing that one comment line, the NEXT line must be a single line of code that does not start with "#".
3) The model must never generate "# Thinking:" on its own. Only continue it when it already appears in the input.
4) For each line of code, there can be at most ONE "# Thinking:" line immediately before it. Absolutely never generate more than one "# Thinking:" in a row.
5) Thinking comments must always be one line only. Never break a thinking comment into multiple lines.
6) If unsure, always write exactly ONE "# Thinking:" line followed by CODE lines.
"""
# 7) Never output decorative characters such as "= = = =", "- - - -", "* * * *", or repeated symbols. Only valid Python code and "# Thinking:" comments are allowed.

    FEW_SHOT = """Here are some examples of how to complete functions following the RULE:

def factorial(n):
    \"\"\"
    Compute factorial of a non-negative integer.
    \"\"\"
    # Thinking: Base case: factorial(0) = 1
    if n == 0:
        return 1
    # Thinking: Recursive case: n * factorial(n-1)
    return n * factorial(n-1)


def reverse_words(s):
    \"\"\"
    Reverse the order of words in a string.
    \"\"\"
    # Thinking: Split string by spaces
    parts = s.split()
    # Thinking: Reverse the word list
    parts.reverse()
    return " ".join(parts)


def find_max(nums):
    \"\"\"
    Return the maximum value in a list.
    \"\"\"
    # Thinking: Initialize max with first element
    maximum = nums[0]
    # Thinking: Update maximum while iterating
    for x in nums[1:]:
        if x > maximum:
            maximum = x
    return maximum
"""

    # 拼接完整 prompt
    prompt = (
        RULE
        + "\n\n"
        + FEW_SHOT
        + "\n\nHere is your code to complete:\n\n"
        + context
        + "\n    # Thinking:"
    )
    return prompt



# -------------------------------
# 函数二：解包装 UnCert-CoT prompt
# -------------------------------
def unwrap_uncert_cot_prompt(full_prompt: str) -> str:
    """
    给定完整的 UnCert-CoT prompt（包含 RULE、few-shot、# CONTEXT 等），
    解包装得到纯净的上下文代码。
    """
    # 找到 "# CONTEXT" 出现的位置
    if "Here is your code to complete:" not in full_prompt:
        raise ValueError("Invalid prompt: missing '# CONTEXT' section")

    # 截取 "# CONTEXT" 后的内容
    context_part = full_prompt.split("Here is your code to complete:", 1)[1]

    # 去掉 "# CONTEXT" 标记本身
    context_part = context_part.replace("Here is your code to complete:", "", 1)

    # # 去掉最后一个 "# Thinking:"（因为这是触发标记，不是代码）
    # if "# Thinking:" in context_part:
    #     context_part = context_part.rsplit("# Thinking:", 1)[0]

    # 去掉前后空白行
    return context_part.strip() + "\n"




class Generator_CoT:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            model_name: str,
            beam_size: int = 3,
            decoding_mode: str = 'Traditional',     # Traditional or AdaFixL or AdaDynL
            entropy_threshold='Learned'             # 'Learned' or a number
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.beam_size = beam_size
        self.tradition_times = 0
        self.lookahead_times = 0
        self.logging_gen: str = ''
        self.lookahead_beam_size = 3
        self.generation_counter = 0
        self.decoding_mode = decoding_mode

        self.entropy_threshold = None
        if decoding_mode == 'Traditional':
            self.entropy_threshold = float('inf')
        elif decoding_mode == 'AdaFixL' or decoding_mode == 'AdaDynL':
            if entropy_threshold == 'Learned':
                self.entropy_threshold = self._load_learned_threshold("data/learned_thresholds.json")
            else:
                try:
                    self.entropy_threshold = float(entropy_threshold)
                except ValueError:
                    raise ValueError("Entropy threshold must be a number or 'Learned'")
        else:
            raise ValueError(f"Unsupported decoding_mode: expected AdaFixL or AdaDynL, got {self.decoding_mode}")


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
        # return -torch.sum(next_token_probs_exp * log_probs, dim=-1)
        entropy = -torch.sum(next_token_probs_exp * log_probs, dim=-1)
        
        # 获取词汇表长度（logits的最后一个维度）
        vocab_size = next_token_logits.size(-1)
        # print("vocab_size:", vocab_size)
        
        # 归一化熵：除以log(vocab_size)得到0-1范围的归一化熵
        normalized_entropy = entropy / torch.log(torch.tensor(vocab_size, dtype=torch.float))
        
        return normalized_entropy

    def select_top_beam_scores(self, beam_size, topk_scores, topk_indices, mode):
        
        if mode == 'AdaDynL' or mode == 'Traditional':
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

        if self.decoding_mode == 'AdaDynL':
            next_token_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1).squeeze(0)
            topk_scores, topk_indices = torch.topk(next_token_probs, beam_size)
            topk_scores += beam_scores.item()
            return topk_scores, topk_indices
        
        elif self.decoding_mode == 'AdaFixL' or self.decoding_mode == 'Traditional':
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  # [batch, vocab]
            topk_scores, topk_indices = torch.topk(log_probs, beam_size, dim=-1)    # [batch, beam_size]

            if beam_scores.dim() == 1:
                beam_scores = beam_scores.unsqueeze(1).expand(-1, beam_size)

            topk_scores = topk_scores + beam_scores

            topk_scores = topk_scores.view(-1)
            topk_indices = topk_indices.view(-1)

            return topk_scores, topk_indices
        
        else:
           raise ValueError(f"Unsupported decoding_mode: {self.decoding_mode}")
    
    def k_sampling_function(self, next_token_logits, beam_scores, beam_size, temperature=1.0):
        """
        按温度采样beam_size个token
        
        Args:
            next_token_logits: 下一个token的logits
            beam_scores: 当前beam的分数
            beam_size: 需要采样的token数量
            temperature: 温度参数，控制采样的随机性
        
        Returns:
            sampled_scores: 采样得到的分数
            sampled_indices: 采样得到的token索引
        """
        # 应用温度缩放
        scaled_logits = next_token_logits / temperature
        
        # 计算概率分布
        next_token_probs = torch.nn.functional.softmax(scaled_logits, dim=-1).squeeze(0)
        
        # 按概率分布采样beam_size个token（无重复采样）
        sampled_indices = torch.multinomial(next_token_probs, beam_size, replacement=False)
        
        # 获取对应的log概率
        sampled_log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1).squeeze(0)[sampled_indices]
        
        # 将采样得到的log概率加到beam_scores上
        sampled_scores = sampled_log_probs + beam_scores.item()
        
        return sampled_scores, sampled_indices
    
    def generate(
            self,
            prompt,
            beam_size=1,
            max_new_tokens=512,
            lambda_value=1,
            lookahead_length=5,
            lookahead_beam_size=3,
            logging_detail = False,
    ):
        self.beam_size = beam_size
        self.lookahead_beam_size = lookahead_beam_size

        token_ids = self.tokenizer([prompt], add_special_tokens=True, padding=True, truncation=True,
                                   return_tensors="pt").input_ids
        token_ids = token_ids.to(self.model.device)
        
        decoded_prompt = self.tokenizer.decode(
            self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.model.device),
            skip_special_tokens=True
        )

        token_ids = token_ids.repeat(beam_size, 1)  # [beam_size, seq_len]
        attention_mask = torch.ones_like(token_ids).to(self.model.device)

        beam_scores = torch.zeros(beam_size, dtype=torch.float).to(self.model.device)

        is_finished = [False] * beam_size

        beam_indices = torch.zeros(beam_size, dtype=torch.long, device=self.model.device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                if all(is_finished):
                    break

                self.logging_gen = ''
                self.logging_gen += f'STEP:{step}\n\n'

                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]  # [beam_size, vocab_size]
                
                topk_k_scores = []
                topk_k_indices = []
                topk_k_token_ids = []

                entropy = self.calculate_entropy(next_token_logits)
                self.logging_gen += f"entropy:{entropy}\n"

                for i in range(self.beam_size):
                    if entropy[i] < self.entropy_threshold or is_finished[i] or (not self.tokenizer.decode(token_ids[i], skip_special_tokens=True).strip(' \t').endswith('\n')):
                        self.logging_gen += f"scoring function: \n"
                        curr_topk_scores, curr_topk_indices = self.scoring_function(next_token_logits[i],
                                                                                    beam_scores[i],
                                                                                    beam_size=beam_size)
                        topk_k_scores.append(curr_topk_scores)
                        topk_k_indices.append(curr_topk_indices)
                        # 对于scoring_function，记录当前beam的token_ids，稍后统一添加一个token
                        topk_k_token_ids.append(None)  # 标记为需要后续处理
                        self.tradition_times += 1
                    else:
                        self.logging_gen += f"lookahead scoring function:  (lookahead_beam_size: {self.lookahead_beam_size})\n"
                        curr_topk_scores, curr_topk_indices, curr_token_ids = self.lookahead_scoring_function(
                            decoded_prompt,
                            next_token_logits[i],
                            token_ids[i],
                            beam_scores[i],
                            lookahead_length=lookahead_length,
                            lambda_value=lambda_value,
                        )
                        topk_k_scores.append(curr_topk_scores)
                        topk_k_indices.append(curr_topk_indices)
                        # 对于lookahead_scoring_function，直接使用返回的curr_token_ids
                        topk_k_token_ids.append(curr_token_ids)
                        self.lookahead_times += 1

                if not topk_k_scores:
                    break

                if step == 0:
                    topk_scores = topk_k_scores[0]
                    topk_indices = topk_k_indices[0]
                    # 处理token_ids
                    if topk_k_token_ids[0] is not None:
                        # lookahead情况，直接使用预测的token_ids
                        token_ids = topk_k_token_ids[0]
                    else:
                        # scoring_function情况，添加一个token
                        token_indices = topk_indices % next_token_logits.shape[-1]
                        token_ids = torch.cat([
                            token_ids,
                            token_indices.unsqueeze(-1)
                        ], dim=-1)
                else:
                    topk_scores, topk_indices, beam_indices = self.select_top_beam_scores(
                        beam_size=beam_size,
                        topk_scores=topk_k_scores,
                        topk_indices=topk_k_indices,
                        mode='Traditional'
                    )
                    
                    # 根据beam_indices重建token_ids
                    new_token_ids = []
                    for beam_idx in beam_indices:
                        beam_group_idx = beam_idx // beam_size  # 确定来自哪个beam组
                        within_group_idx = beam_idx % beam_size  # 确定beam组内的索引
                        
                        if topk_k_token_ids[beam_group_idx] is not None:
                            # lookahead情况，使用预测的token_ids
                            new_token_ids.append(topk_k_token_ids[beam_group_idx][within_group_idx])
                        else:
                            # scoring_function情况，添加一个token
                            token_index = topk_indices[len(new_token_ids)] % next_token_logits.shape[-1]
                            new_token_ids.append(torch.cat([
                                token_ids[beam_group_idx],
                                token_index.unsqueeze(0)
                            ], dim=0))
                    
                    token_ids = torch.stack(new_token_ids)

                self.logging_gen += f"\ntopk_scores:{topk_scores}\n"
                self.logging_gen += f"beam_indices:{beam_indices}\n\n"

                beam_scores = topk_scores

                for j in range(beam_size):
                    # if token_indices[j] == self.tokenizer.eos_token_id:
                    #     is_finished[j] = True
                    #     continue

                    prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.model.device)
                    decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    decoded_seq = self.tokenizer.decode(token_ids[j], skip_special_tokens=True)
                    current_sequence = decoded_seq[len(decoded_prompt):]
                    lines = current_sequence.split('\n')

                    # 检查生成的token数是否超过512
                    # generated_tokens = len(token_ids[j]) - len(prompt_ids)
                    # is_max_length_reached = generated_tokens >= 512
                    is_repeat_signal = "= = = = = = =" in lines[-1]

                    # 取出最后所有行里非空的那些
                    last_non_empty = [ln for ln in lines if ln.strip() != ''][-5:]

                    is_consecutive_empty = (
                        len(lines) >= 4 and
                        (
                            all(line.strip() == '' for line in lines[-4:]) or
                            (len(last_non_empty) == 5 and
                            all(ln.lstrip().startswith('# Thinking') for ln in last_non_empty))
                        )
                    )

                    is_endswith_Human = lines and lines[-1].strip().endswith('Human')

                    all_lines_valid = all(line.startswith((' ', '\t', '# Thinking:')) for line in lines if line.strip())
                    
                    # 新增：倒数第二行是否仅有一个缩进层级且以 return 开头
                    is_second_last_return = False
                    if len(lines) >= 2:
                        second_last = lines[-2]
                        # 匹配“2或4个空格，或1个Tab”作为唯一缩进
                        m = re.match(r'^((?: {2}| {4}|\t))return', second_last)
                        is_second_last_return = bool(m)

                    # 原来的 is_finished 逻辑，把新增条件一起或进去
                    is_finished[j] = (
                        not all_lines_valid
                        or is_consecutive_empty
                        or is_endswith_Human
                        # or is_max_length_reached
                        or is_repeat_signal
                        or is_second_last_return
                    )

                attention_mask = token_ids.ne(self.tokenizer.pad_token_id).to(self.model.device)

                if logging_detail:
                    decoded_sequences = [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in token_ids
                    ]

                    gen_list = []
                    for gen in decoded_sequences:
                        gen_list.append(gen)

                    for count_gen in range(beam_size):
                        self.logging_gen += f'----------candidate:{count_gen}----------\n' + gen_list[count_gen][
                                                                                            len(prompt):] + '\n\n'

                    logging.info(self.logging_gen)

                torch.cuda.empty_cache()

        decoded_sequences = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in token_ids
        ]
        
        if logging_detail:
            logging.info(f"Total tradition_times:{self.tradition_times}\n Total lookahead_times:{self.lookahead_times}\n")

        torch.cuda.empty_cache()

        return [gen[len(prompt):] for gen in decoded_sequences]

    def lookahead_scoring_function(self, decoded_prompt, next_token_logits, token_ids, beam_scores,
                                lookahead_length, lambda_value):
        lookahead_beam_size = self.lookahead_beam_size
        beam_size = self.beam_size

        history_topk_score = beam_scores
        
        decoded_seq = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        uncert_cot_seq = build_uncert_cot_prompt(decoded_seq)
        
        # 将uncert_cot_seq转换为token_ids
        prompt_token_ids = self.tokenizer.encode(uncert_cot_seq, return_tensors='pt', add_special_tokens=False).squeeze(0)

        # 确保在正确的设备上
        prompt_token_ids = prompt_token_ids.to(token_ids.device)

        # 以温度t采样k个样本（k个开头的token）
        sampled_scores, sampled_indices = self.k_sampling_function(
            next_token_logits, beam_scores, beam_size=lookahead_beam_size, temperature=0.4)

        current_sampled_scores = sampled_scores - history_topk_score
        token_indices = sampled_indices % next_token_logits.shape[-1]

        token_ids = prompt_token_ids.repeat(lookahead_beam_size, 1)
        token_ids = torch.cat([token_ids, token_indices.unsqueeze(-1)], dim=-1)

        if self.decoding_mode == 'AdaDynL':
            # 创建一个新的token_ids列表来存储更新后的序列
            updated_token_ids = []
            
            for i in range(lookahead_beam_size):
                confidence, actual_length, cur_token_ids = self.get_lookahead_score_DynL(
                    token_ids=token_ids[i].unsqueeze(0).to(self.model.device),
                    lookahead_length=lookahead_length,
                    decoded_prompt=decoded_prompt
                )
                # 解码cur_token_ids为文本
                decoded_text = self.tokenizer.decode(cur_token_ids.squeeze(0), skip_special_tokens=True)
                
                # 调用unwrap_uncert_cot_prompt处理文本
                print(decoded_text)
                processed_text = unwrap_uncert_cot_prompt(decoded_text)
                
                # 重新编码为token_ids
                processed_token_ids = self.tokenizer.encode(processed_text, return_tensors='pt', add_special_tokens=False).squeeze(0).to(cur_token_ids.device)
                
                # 将处理后的token_ids添加到新列表中
                updated_token_ids.append(processed_token_ids)
                
                sampled_scores[i] = history_topk_score + (current_sampled_scores[i] + confidence) / (actual_length + 1) * lambda_value

            # 将列表转换为tensor
            # 由于不同beam的序列长度可能不同，需要进行padding或截断处理
            max_length = max(seq.shape[0] for seq in updated_token_ids)
            
            # 创建padding后的tensor
            padded_token_ids = []
            for seq in updated_token_ids:
                if seq.shape[0] < max_length:
                    # 如果序列较短，在末尾填充pad_token_id（通常是0或特殊值）
                    pad_length = max_length - seq.shape[0]
                    pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
                    padded_seq = torch.cat([seq, torch.full((pad_length,), pad_token_id, 
                                                        dtype=seq.dtype, device=seq.device)])
                else:
                    padded_seq = seq[:max_length]  # 截断到最大长度
                padded_token_ids.append(padded_seq)
            
            token_ids = torch.stack(padded_token_ids)

            sampled_scores, sampled_indices_temp = torch.topk(sampled_scores, beam_size)
            sampled_indices = sampled_indices[sampled_indices_temp]
            
            # 更新token_ids以保持与sampled_indices的对应关系
            token_ids = token_ids[sampled_indices_temp]

            return sampled_scores, sampled_indices, token_ids
        
        else:
           raise ValueError(f"Unsupported decoding_mode: expected AdaFixL or AdaDynL, got {self.decoding_mode}")


    def get_lookahead_score_fixL(self, token_ids, lookahead_length, decoded_prompt):
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

    def get_lookahead_score_DynL(self, token_ids, lookahead_length, decoded_prompt):
        '''
        生成一行Thinking注释和一行代码，返回代码行置信度和总context
        '''
        beam_size = self.beam_size
        actual_lookahead_length = 0

        token_ids = token_ids.repeat(beam_size, 1)  # [beam_size, seq_len]
        attention_mask = torch.ones_like(token_ids).to(self.model.device)

        lookahead_scores = torch.zeros(beam_size, dtype=torch.float).to(self.model.device)
        beam_indices = torch.zeros(beam_size, dtype=torch.long, device=token_ids.device)

        is_finished = [False] * beam_size
        margins = [[] for _ in range(beam_size)]  # Track margins for each beam

        with torch.no_grad():
            # Thinking行
            for _ in range(lookahead_length):
                if all(is_finished):
                    break

                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]  # [beam_size, vocab_size]

                topk_k_scores = []
                topk_k_indices = []

                for i in range(self.beam_size):
                    curr_topk_scores, curr_topk_indices = self.scoring_function(
                        next_token_logits[i:i + 1],
                        lookahead_scores[i:i + 1],
                        beam_size=beam_size
                    )
                    topk_k_scores.append(curr_topk_scores)
                    topk_k_indices.append(curr_topk_indices)

                if not topk_k_scores:
                    break

                topk_scores, topk_indices, beam_indices = self.select_top_beam_scores(
                    beam_size=beam_size,
                    topk_scores=topk_k_scores,
                    topk_indices=topk_k_indices,
                    mode='AdaDynL'
                )

                actual_lookahead_length += 1
                token_indices = topk_indices % next_token_logits.shape[-1]

                beam_indices = beam_indices.to(token_ids.device)
                token_ids = torch.cat([token_ids[beam_indices], token_indices.unsqueeze(-1)], dim=-1)
                lookahead_scores = topk_scores

                for j in range(beam_size):
                    if token_indices[j] == self.tokenizer.eos_token_id:
                        is_finished[j] = True
                        continue

                    decoded_seq = self.tokenizer.decode(token_ids[j], skip_special_tokens=True)
                    # 截取 "# CONTEXT" 后的内容
                    context_part = decoded_seq.split("Here is your code to complete:", 1)[1]
                    # 去掉 "# CONTEXT" 标记本身
                    context_part = context_part.replace("Here is your code to complete:", "", 1)
                    current_sequence = context_part[len(decoded_prompt):]

                    # lines = current_sequence.split('\n')

                    # is_consecutive_empty = (
                    #     len(lines) >= 4 and
                    #     all(line.strip() == '' for line in lines[-4:])
                    # )

                    # all_lines_valid = all(
                    #     not line or line.startswith((' ', '\t'))
                    #     for line in lines
                    # )
                    
                    ends_with_newline = current_sequence.endswith('\n')

                    is_finished[j] = ends_with_newline

                    # is_finished[j] = (not all_lines_valid) or is_consecutive_empty or ends_with_newline

                attention_mask = token_ids.ne(self.tokenizer.pad_token_id).to(self.model.device)
            
            # 代码行
            for _ in range(99999999):  # 实际迭代次数由is_finished控制
                if all(is_finished):
                    break

                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]  # [beam_size, vocab_size]
                
                # Compute probabilities and margins for each current beam
                probs = torch.softmax(next_token_logits, dim=-1)
                beam_margins = []
                for i in range(beam_size):
                    sorted_p = torch.sort(probs[i], descending=True).values
                    margin = sorted_p[0] - sorted_p[1]
                    beam_margins.append(margin)

                topk_k_scores = []
                topk_k_indices = []

                for i in range(self.beam_size):
                    curr_topk_scores, curr_topk_indices = self.scoring_function(
                        next_token_logits[i:i + 1],
                        lookahead_scores[i:i + 1],
                        beam_size=beam_size
                    )
                    topk_k_scores.append(curr_topk_scores)
                    topk_k_indices.append(curr_topk_indices)

                if not topk_k_scores:
                    break

                topk_scores, topk_indices, beam_indices = self.select_top_beam_scores(
                    beam_size=beam_size,
                    topk_scores=topk_k_scores,
                    topk_indices=topk_k_indices,
                    mode='AdaDynL'
                )

                actual_lookahead_length += 1
                token_indices = topk_indices % next_token_logits.shape[-1]

                beam_indices = beam_indices.to(token_ids.device)
                token_ids = torch.cat([token_ids[beam_indices], token_indices.unsqueeze(-1)], dim=-1)
                lookahead_scores = topk_scores
                
                # Update margins for new beams
                new_margins = [[] for _ in range(beam_size)]
                for j in range(beam_size):
                    parent = beam_indices[j]
                    new_margins[j] = margins[parent] + [beam_margins[parent]]
                margins = new_margins

                for j in range(beam_size):
                    if token_indices[j] == self.tokenizer.eos_token_id:
                        is_finished[j] = True
                        continue

                    decoded_seq = self.tokenizer.decode(token_ids[j], skip_special_tokens=True)
                    # 截取 "# CONTEXT" 后的内容
                    context_part = decoded_seq.split("Here is your code to complete:", 1)[1]
                    # 去掉 "# CONTEXT" 标记本身
                    context_part = context_part.replace("Here is your code to complete:", "", 1)
                    current_sequence = context_part[len(decoded_prompt):]

                    lines = current_sequence.split('\n')

                    is_consecutive_empty = (
                        len(lines) >= 4 and
                        all(line.strip() == '' for line in lines[-4:])
                    )

                    all_lines_valid = all(
                        not line or line.startswith((' ', '\t'))
                        for line in lines
                    )
                    
                    ends_with_newline = current_sequence.endswith('\n')

                    is_finished[j] = (not all_lines_valid) or is_consecutive_empty or ends_with_newline


                attention_mask = token_ids.ne(self.tokenizer.pad_token_id).to(self.model.device)
                
        self.logging_gen += f"actual_lookahead_length:{actual_lookahead_length}\n"
        
        # Find the best beam and compute its confidence as average margin
        best_beam = torch.argmax(lookahead_scores)
        if margins[best_beam]:
            confidence = sum(margins[best_beam]) / len(margins[best_beam])
        else:
            confidence = 0.0

        return confidence, actual_lookahead_length, token_ids