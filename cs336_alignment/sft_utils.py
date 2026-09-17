import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:

    all_full_ids = []
    all_full_response_masks = []
    
    for prompt, output in zip(prompt_strs, output_strs):

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens = False,
        )['input_ids']

        output_ids = tokenizer(
            output,
            add_special_tokens = False,
        )['input_ids']

        full_ids = prompt_ids + output_ids

        full_response_mask = (
            [0] * (len(prompt_ids)) 
            + [1] * (len(output_ids))
        )

        all_full_ids.append(full_ids)
        all_full_response_masks.append(full_response_mask)

    max_len = max(len(x) for x in all_full_ids)

    pad_token_id = tokenizer.pad_token_id

    padded_full_ids = []
    padded_full_response_masks = []

    for full_ids, full_response_masks in zip(
        all_full_ids,
        all_full_response_masks,
    ):
        pad_len = max_len - len(full_ids)

        padded_full_ids.append(
            full_ids + [pad_token_id] * pad_len
        )

        padded_full_response_masks.append(
            full_response_masks + [0] * pad_len
        )

    padded_full_ids = torch.tensor(
        padded_full_ids,
        dtype = torch.long,
    )

    padded_full_response_masks = torch.tensor(
        padded_full_response_masks,
        dtype = torch.bool,
    )

    input_ids = padded_full_ids[:, :-1]
    labels = padded_full_ids[:, 1:]

    response_mask = padded_full_response_masks[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits

    all_log_probs = torch.log_softmax(logits, dim = -1)

    log_probs = torch.gather(
        all_log_probs,
        dim = -1,
        index = labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        'log_probs': log_probs
    }

    if return_token_entropy:
        probs = all_log_probs.exp()

        token_entropy = -(
            probs * all_log_probs
        ).sum(-1)

        result['token_entropy'] = token_entropy

    return result