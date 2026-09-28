import random

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from tqdm import tqdm


def set_generation_seed(seed):
    """Make direct model.generate baselines reproducible per dataset item."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def generation_cache_kwargs(model) -> dict:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    model_name = model.__class__.__name__.lower()
    if model_type != "phi3" and "phi3" not in model_name:
        return {}
    try:
        major = int(str(transformers.__version__).split(".", 1)[0])
    except Exception:
        major = 0
    return {"cache_implementation": "dynamic_full"} if major >= 5 else {}


@torch.inference_mode()
def generate_completion(model, tokenizer, input_ids, max_new_tokens, temperature=None):
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": False,
        "do_sample": True,
        "repetition_penalty": 1.0,
    }
    kwargs.update(generation_cache_kwargs(model))
    if temperature is not None:
        kwargs["temperature"] = temperature
    output = model.generate(**kwargs)
    return output.sequences[0]


class AutoregressiveSampler:
    """Thin wrapper around a causal LM for token-level next-token log-probs."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    @torch.no_grad()
    def next_token(self, prefix):
        """Return log-probabilities for the next token given a tokenized prefix."""
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = (
            torch_prefix
            if torch_prefix.size(1) <= self.block_size
            else torch_prefix[:, -self.block_size :]
        )
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


def _as_eos_token_ids(value):
    """Normalize a Transformers EOS setting to an ordered tuple of ids."""
    if value is None:
        return ()
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)

    token_ids = []
    for token_id in values:
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id not in token_ids:
            token_ids.append(token_id)
    return tuple(token_ids)


def _generation_eos_token_ids(p: AutoregressiveSampler):
    """Use the model's generation stops, falling back to the tokenizer EOS."""
    generation_config = getattr(p.model, "generation_config", None)
    token_ids = _as_eos_token_ids(getattr(generation_config, "eos_token_id", None))
    if token_ids:
        return token_ids
    return _as_eos_token_ids(getattr(p.tokenizer, "eos_token_id", None))


def _truncate_at_generated_eos(
    gen,
    log_probs_norm,
    log_probs_unnorm,
    context_length,
    eos_token_ids,
):
    """Crop at the first EOS generated after the immutable prompt prefix."""
    eos_token_ids = set(eos_token_ids)
    if not eos_token_ids:
        return None

    eos_idx = next(
        (
            idx
            for idx, token_id in enumerate(gen[context_length:], context_length)
            if int(token_id) in eos_token_ids
        ),
        None,
    )
    if eos_idx is None:
        return None

    generated_tokens_to_keep = eos_idx + 1 - context_length
    return (
        gen[: eos_idx + 1],
        log_probs_norm[:generated_tokens_to_keep],
        log_probs_unnorm[:generated_tokens_to_keep],
    )


def naive_temp(p: AutoregressiveSampler, context, temp, seq_len):
    """Generate a continuation with temperature sampling and token log-probs."""
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    eos_token_ids = _generation_eos_token_ids(p)
    if len(eos_token_ids) == 1:
        generation_eos = eos_token_ids[0]
    elif eos_token_ids:
        generation_eos = list(eos_token_ids)
    else:
        generation_eos = None

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    if generation_eos is not None:
        generate_kwargs["eos_token_id"] = generation_eos
    generate_kwargs.update(generation_cache_kwargs(p.model))
    output = p.model.generate(**generate_kwargs)
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (
        (1 / temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx))
        .view(-1)
        .tolist()
    )
    log_probs_norm = (
        torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()
    )

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


def mcmc_power_samp(
    p: AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16
):
    """Metropolis-Hastings autoregressive power sampler over generated suffixes."""
    c = len(context)
    print(f"alpha: {1 / temp}")
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0
    eos_token_ids = _generation_eos_token_ids(p)

    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(
            p, gen, temp=temp, seq_len=jump_size + len(gen)
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)

            prop, log_prob_prop, target_log_prob_prop = naive_temp(
                p, gen[:idx], temp=temp, seq_len=t
            )
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]
            log_r = (
                sum(target_log_prob_prop)
                + sum(log_prob_cur)
                - sum(target_log_prob_cur)
                - sum(log_prob_prop)
            )

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        truncated = _truncate_at_generated_eos(
            gen,
            log_probs_norm,
            log_probs_unnorm,
            c,
            eos_token_ids,
        )
        if truncated is not None:
            gen, log_probs_norm, log_probs_unnorm = truncated
            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio
