import torch
from typing import Callable, Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:

    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
        )
    ]

    raw_rewards = torch.tensor(
        [reward_dict['reward'] for reward_dict in reward_dicts],
    )

    metadata = {
        'reward_mean': raw_rewards.mean().item(),
        'mean_format_reward': sum(
            reward_dict['format_reward'] for reward_dict in reward_dicts
        ) / len(reward_dicts),
        'mean_total_reward': sum(
            reward_dict['answer_reward'] for reward_dict in reward_dicts
        ) / len(reward_dicts)
    }

    return raw_rewards, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal['mean', 'none'] = 'mean',
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal['std', 'none', 'mean'] = 'std',
) -> tuple[torch.Tensor, dict[str, float]]:
    
    group_rewards = raw_rewards.reshape(-1, group_size)

    group_mean = group_rewards.mean(
        dim = 1,
        keepdim = True,
    )

    if baseline == 'mean':
        advantages = group_rewards - group_mean
    elif baseline == 'none':
        advantages = group_rewards
    else:
        raise ValueError(f'Unknown baseline:{baseline}')
    
    if advantage_normalizer == 'std':
        group_std = group_rewards.std(
            dim = 1,
            keepdim = True,
        )

        advantages = advantages / (group_std + advantage_eps)

    elif advantage_normalizer == 'none':
        pass
    
    elif advantage_normalizer == 'mean':
        advantages = advantages / (group_mean + advantage_eps)

    else:
        raise ValueError(f'Unknown advantage_normalizer{advantage_normalizer}')
    
    advantages = advantages.reshape(-1)

    metadata = {
        'reward_mean': raw_rewards.mean().item(),
        'reward_std': raw_rewards.std().item(),
        'reward_max': raw_rewards.max().item(),
        'reward_min': raw_rewards.min().item(),
    }

    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo'] = 'none',
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    metadata = {}

    if importance_reweighting_method == 'none':
        per_token_loss = -advantages * policy_log_probs #广播

    elif importance_reweighting_method == 'noclip':
        if old_log_probs is None:
            raise ValueError('old_log_probs is required for noclip')

        ratio = torch.exp(policy_log_probs - old_log_probs)

        per_token_loss = -ratio * advantages

        metadata['importance_ration_mean'] = ration.detach().mean()

    elif importance_reweighting_method == 'grpo':
        if old_log_probs is None:
            raise ValueError('old_log_probs is required for grpo')
        if cliprange is None:
            raise ValueError('cliprange is required for grpo')

        ratio = torch.exp(policy_log_probs - old_log_probs)

        unclipped_objective = ratio * advangages

        clipped_ration = torch.clamp(
            ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )

        clipped_objective = clipped_ratio * advantages

        per_token_loss = -torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        metadata["clip_fraction"] = (
            (ratio != clipped_ratio)
            .float()
            .detach()
            .mean()
        )

    elif importance_reweighting_method == "gspo":
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for gspo")
        if cliprange is None:
            raise ValueError("cliprange is required for gspo")
        if response_mask is None:
            raise ValueError("response_mask is required for gspo")

        log_ratio = policy_log_probs - old_log_probs

        sequence_log_ratio = (
            (log_ratio * response_mask).sum(dim = 1, keepdim = True)
            / response_mask.sum(dim = 1, keepdim = True)
        )

        sequence_ratio = torch.exp(sequence_log_ratio)

        unclipped_objective = sequence_ratio * advantages

        clipped_ratio = torch.clamp(
            sequence_ratio,
            1.0 - cliprange,
            1.0 + cliprange,
        )

        clipped_objective = clipped_ratio * advantages

        sequence_loss = -torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        per_token_loss = sequence_loss.expand_as(policy_log_probs)

        metadata["clip_fraction"] = (
            (sequence_ratio != clipped_ratio)
            .float()
            .detach()
            .mean()
        )

    else:
        raise ValueError(
            f"Unknown importance_reweighting_method: "
            f"{importance_reweighting_method}"
        )

    return per_token_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal['sequence', 'constant'] = 'sequence',
    normalization_constant: int | None = None,
) -> torch.Tensor:

    masked_loss = per_token_policy_gradient_loss * mask

    if loss_normalization == 'sequence':
        per_seq_loss = (
            masked_loss.sum(dim = -1)
            / mask.sum(dim = -1)
        )

        loss = per_seq_loss.mean()

    elif loss_normalization == 'constant':
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required "
                "when loss_normalization='constant'"
            )
        loss = masked_loss.sum() / normalization_constant

    else:
        raise ValueError(
            f"Unknown loss_normalization: {loss_normalization}"
        )

    return loss

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,

    baseline: Literal['mean', 'none'] = 'mean',
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal['std', 'none', 'mean'] = 'std',

    importance_reweighting_method: Literal['none', 'noclip', 'grpo', 'gspo'] = 'none',
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,

    loss_normalization: Literal['sequence', 'constant'] = 'sequence',
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs

    tokenize_result = tokenize_prompt_and_output(
        prompt_strs = repeated_prompts,
        output_strs = rollout_responses,
        tokenizer = tokenizer,
    ) 

    input_ids = tokenize_result['input_ids']
    labels = tokenize_result['labels']
    response_mask = tokenize_result['response_mask'] 

    raw_rewards, raw_rewards_metadata = compute_rollout_rewards(
        reward_fn = reward_fn,
        rollout_responses = rollout_responses,
        repeated_ground_truths = repeated_ground_truths,
    )

    advantages, advantages_metadata = compute_group_normalized_rewards(
        raw_rewards = raw_rewards,
        group_size = group_size,
        baseline = baseline,
        advantage_eps = advantage_eps,
        advantage_normalizer = advantage_normalizer,
    )

    device = next(model.parameters()).device

    batch_size = input_ids.shape[0]
    micro_batch_size = batch_size // gradient_accumulation_steps

    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    total_response_tokens = torch.zeros((), device=device)

    optimizer.zero_grad()

    for start in range(0, batch_size, micro_batch_size):
        end = start + micro_batch_size

        micro_input_ids = input_ids[start : end].to(device)
        micro_labels = labels[start : end].to(device)
        micro_response_mask = response_mask[start : end].to(device)
        micro_advantages = advantages[start : end].to(device)

        micro_policy_result = get_response_log_probs(
            model = model,
            input_ids = micro_input_ids,
            labels = micro_labels,
            return_token_entropy = True,
        )

        micro_policy_log_probs = micro_policy_result['log_probs']
        micro_token_entropy = micro_policy_result['token_entropy']

        per_token_loss, pre_token_loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages = micro_advantages,
            policy_log_probs = micro_policy_log_probs,
            importance_reweighting_method = importance_reweighting_method,
            old_log_probs = None,
            cliprange = cliprange,
            response_mask = micro_response_mask,
        )

        micro_batch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss = per_token_loss,
            mask = micro_response_mask,
            loss_normalization = loss_normalization,
            normalization_constant = normalization_constant,
        )

        scaled_loss = micro_batch_loss * (
            len(micro_input_ids) / batch_size
        )

        scaled_loss.backward()

        
        total_loss += scaled_loss.detach()

        total_entropy += (
            micro_token_entropy * micro_response_mask
        ).sum().detach()

        total_response_tokens += micro_response_mask.sum().detach()

    mean_token_entropy = total_entropy / total_response_tokens

    metadata = {}
    metadata.update(raw_rewards_metadata)
    metadata.update(advantages_metadata)

    metadata["loss"] = total_loss
    metadata["token_entropy"] = mean_token_entropy

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        metadata["grad_norm"] = grad_norm
    else:
        metadata["grad_norm"] = None

    optimizer.step()
    optimizer.zero_grad()

    return total_loss, metadata
    