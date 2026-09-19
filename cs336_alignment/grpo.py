import torch
from typing import Callable, Literal

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