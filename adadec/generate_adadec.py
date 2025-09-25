from typing import Optional, Union, List, Dict, Any
import torch

from adadec.generator import Generator


def _ensure_model_and_tokenizer(model: Optional[object], tokenizer: Optional[object], model_name: Optional[str]):
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    model_obj = model
    tokenizer_obj = tokenizer
    resolved_name = model_name or getattr(model_obj, "name", "model")
    return model_obj, tokenizer_obj, resolved_name


def generate_adadec(
    model: Optional[object],
    tokenizer: Optional[object] = None,
    prompts: Union[str, List[str]] = "",
    model_name: Optional[str] = None,
    beam_size: int = 1,
    decoding_mode: str = 'AdaFixL',
    entropy_threshold: Union[str, float] = 'Learned',
    max_new_tokens: int = 512,
    lambda_value: float = 1.0,
    lookahead_length: int = 5,
    lookahead_beam_size: int = 3,
    device: Optional[str] = None,
    stop_words_file: str = ""
) -> Dict[str, Any]:
    is_single = isinstance(prompts, str)
    prompt_list = [prompts] if is_single else list(prompts)

    model_obj, tokenizer_obj, resolved_name = _ensure_model_and_tokenizer(model, tokenizer, model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_obj.to(device)
    model_obj.eval()

    gen = Generator(
        model=model_obj,
        tokenizer=tokenizer_obj,
        model_name=resolved_name,
        beam_size=beam_size,
        decoding_mode=decoding_mode,
        entropy_threshold=entropy_threshold,
        stop_words_file=stop_words_file
    )

    results = []
    for prompt in prompt_list:
        try:
            candidates = gen.generate(
                prompt=prompt,
                beam_size=beam_size,
                max_new_tokens=max_new_tokens,
                lambda_value=lambda_value,
                lookahead_length=lookahead_length,
                lookahead_beam_size=lookahead_beam_size,
            )
        except Exception as e:
            results.append({
                "prompt": prompt,
                "error": str(e),
                "candidates": [],
                "tradition_times": getattr(gen, "tradition_times", None),
                "lookahead_times": getattr(gen, "lookahead_times", None),
            })
            continue

        entry = {
            "prompt": prompt,
            "candidates": candidates,
            "tradition_times": gen.tradition_times,
            "lookahead_times": gen.lookahead_times,
        }
        results.append(entry)

    return {
        "model_name": resolved_name,
        "device": device,
        "results": results
    }