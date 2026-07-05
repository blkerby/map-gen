from __future__ import annotations

from env import (
    Actions,
    CandidateStats,
    CandidateSlot,
    DoorMatchCounts,
    Engine,
    EndOutcomes,
    EnvironmentGroup,
    EpisodeData,
    EpisodeOutcomes,
    GenerateConfig,
    WaveFrameData,
    StepOutcomes,
    ProposalCandidateMask,
    WaveProposalData,
    WaveProposalCandidateMask,
    FeatureRequirements,
    FeatureSlot,
    Features,
    extract_candidate_features,
    merge_wave_frame_data,
)
from loss import compute_step_balance_score_target_logits
from model import BalancePredictions, Predictions
from dataclasses import dataclass
import logging
import math
import threading
import time
import torch

from train_config import Config

type ProfileReport = list[tuple[str, int, int]]
type GenerationStats = dict[str, float]


class GenerationProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.counts: dict[str, int] = {}
        self.nanos: dict[str, int] = {}
        self.lock = threading.Lock()

    def add(self, name: str, start: int) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter_ns() - start
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + 1
            self.nanos[name] = self.nanos.get(name, 0) + elapsed

    def report(self) -> ProfileReport:
        with self.lock:
            return [(name, self.counts[name], self.nanos[name]) for name in sorted(self.counts)]


def profile_start(enabled: bool) -> int:
    return time.perf_counter_ns() if enabled else 0


def sync_profile_device(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def outcome_reward(model_logprobs: torch.Tensor, known_invalid: torch.Tensor) -> torch.Tensor:
    if known_invalid.ndim == model_logprobs.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    known_reward = torch.zeros_like(model_logprobs)
    return torch.where(known_invalid < 0, model_logprobs, known_reward)


def balance_reward(
    balance_score: torch.Tensor,
    door_invalid: torch.Tensor,
    known_invalid: torch.Tensor,
) -> torch.Tensor:
    if known_invalid.ndim == balance_score.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    match_probability = torch.sigmoid(-door_invalid)
    known_match_probability = torch.where(
        known_invalid == 0,
        torch.ones_like(match_probability),
        torch.zeros_like(match_probability),
    )
    match_probability = torch.where(
        known_invalid < 0,
        match_probability,
        known_match_probability,
    )
    model_reward = -balance_score * match_probability
    known_reward = torch.zeros_like(model_reward)
    return torch.where(known_invalid == 0, known_reward, model_reward)


def toilet_balance_reward(
    toilet_balance_score: torch.Tensor,
    toilet_invalid: torch.Tensor,
    known_invalid: torch.Tensor,
) -> torch.Tensor:
    if known_invalid.ndim == toilet_balance_score.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    valid_probability = torch.sigmoid(-toilet_invalid)
    known_valid_probability = torch.where(
        known_invalid == 0,
        torch.ones_like(valid_probability),
        torch.zeros_like(valid_probability),
    )
    valid_probability = torch.where(
        known_invalid < 0,
        valid_probability,
        known_valid_probability,
    )
    return -toilet_balance_score * valid_probability


def total_proximity_utility(utility: torch.Tensor) -> torch.Tensor:
    utility = utility.to(torch.float32)
    return torch.sum(utility, dim=2)


# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
# preds.toilet_invalid: [batch_size, max_candidates]
# preds.phantoon_invalid: [batch_size, max_candidates]
def compute_expected_reward(
    preds,
    outcomes,
    config: GenerateConfig,
):
    def batch_weight(value: float | torch.Tensor) -> float | torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(preds.door_invalid.device).view(-1, 1)
        return value

    door_logprobs = torch.nn.functional.logsigmoid(-preds.door_invalid)
    connection_logprobs = torch.nn.functional.logsigmoid(-preds.connection_invalid)
    toilet_logprobs = torch.nn.functional.logsigmoid(-preds.toilet_invalid)
    phantoon_logprobs = torch.nn.functional.logsigmoid(-preds.phantoon_invalid)
    door_logprobs = outcome_reward(door_logprobs, outcomes.door_invalid)
    connection_logprobs = outcome_reward(connection_logprobs, outcomes.connection_invalid)
    toilet_logprobs = outcome_reward(toilet_logprobs, outcomes.toilet_invalid)
    phantoon_logprobs = outcome_reward(phantoon_logprobs, outcomes.phantoon_invalid)
    balance_scores = balance_reward(
        preds.balance_score,
        preds.door_invalid,
        outcomes.door_invalid,
    )
    toilet_balance_scores = toilet_balance_reward(
        preds.toilet_balance_score,
        preds.toilet_invalid,
        outcomes.toilet_invalid,
    )
    return (
        batch_weight(config.reward_door) * torch.sum(door_logprobs, dim=2)
        + batch_weight(config.reward_connection) * torch.sum(connection_logprobs, dim=2)
        + batch_weight(config.reward_toilet) * toilet_logprobs
        + batch_weight(config.reward_phantoon) * phantoon_logprobs
        + batch_weight(config.reward_balance) * torch.sum(balance_scores, dim=2)
        + batch_weight(config.reward_toilet_balance) * toilet_balance_scores
        - batch_weight(config.reward_frontier) * preds.avg_frontiers.to(torch.float32)
        - batch_weight(config.reward_graph_diameter) * preds.graph_diameter.to(torch.float32)
        + batch_weight(config.reward_save_distance)
        * (
            total_proximity_utility(preds.save_to_room_utility)
            + total_proximity_utility(preds.save_from_room_utility)
        )
        + batch_weight(config.reward_refill_distance)
        * (
            total_proximity_utility(preds.refill_to_room_utility)
            + total_proximity_utility(preds.refill_from_room_utility)
        )
        + (
            batch_weight(config.reward_missing_connect_utility)
            * total_proximity_utility(preds.missing_connect_utility)
        )
    )


def index_generate_config(config: GenerateConfig, row_env_idx: torch.Tensor) -> GenerateConfig:
    def index_value(value: float | torch.Tensor) -> float | torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=row_env_idx.device)[row_env_idx]
        return value

    return GenerateConfig(
        episode_length=config.episode_length,
        recommended_candidates=config.recommended_candidates,
        shortlist_candidates=config.shortlist_candidates,
        gpu_prefetch_batches=config.gpu_prefetch_batches,
        temperature=index_value(config.temperature),
        proposal_temperature=index_value(config.proposal_temperature),
        reward_door=index_value(config.reward_door),
        reward_connection=index_value(config.reward_connection),
        reward_toilet=index_value(config.reward_toilet),
        reward_phantoon=index_value(config.reward_phantoon),
        reward_balance=index_value(config.reward_balance),
        reward_toilet_balance=index_value(config.reward_toilet_balance),
        reward_frontier=index_value(config.reward_frontier),
        reward_graph_diameter=index_value(config.reward_graph_diameter),
        reward_save_distance=index_value(config.reward_save_distance),
        reward_refill_distance=index_value(config.reward_refill_distance),
        reward_missing_connect_utility=index_value(config.reward_missing_connect_utility),
        generation_variable_floats=index_value(config.generation_variable_floats),
        distance_proximity_scale=config.distance_proximity_scale,
        autocast=config.autocast,
    )


def transfer_features(
    features: Features,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None = None,
) -> Features:
    if transfer_stream is None or device.type != "cuda":
        result = features.to(device)
        result.mark_dynamic()
        return result
    current_stream = torch.cuda.current_stream(device)
    with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
        result = features.to(device, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(transfer_stream)
    current_stream.wait_event(ready)
    result.mark_dynamic()
    return result


def select_candidate_actions(actions: Actions, mask: torch.Tensor) -> Actions:
    return Actions(
        room_idx=actions.room_idx.flatten(0, 1)[mask].unsqueeze(1),
        room_x=actions.room_x.flatten(0, 1)[mask].unsqueeze(1),
        room_y=actions.room_y.flatten(0, 1)[mask].unsqueeze(1),
    )


def select_candidate_step_outcomes(outcomes: StepOutcomes, mask: torch.Tensor) -> StepOutcomes:
    return StepOutcomes(
        door_invalid=outcomes.door_invalid.flatten(0, 1)[mask].unsqueeze(1),
        connection_invalid=outcomes.connection_invalid.flatten(0, 1)[mask].unsqueeze(1),
        toilet_invalid=outcomes.toilet_invalid.flatten(0, 1)[mask].unsqueeze(1),
        phantoon_invalid=outcomes.phantoon_invalid.flatten(0, 1)[mask].unsqueeze(1),
        door_match=outcomes.door_match.flatten(0, 1)[mask].unsqueeze(1),
    )


@dataclass
class WaveCandidateBatch:
    row_env_idx: torch.Tensor
    candidates: Actions
    proposal_frontier_idx: torch.Tensor
    proposal_door_variant_idx: torch.Tensor
    reward_outcomes: StepOutcomes
    post_candidate_outcomes: StepOutcomes
    feature_requirements: FeatureRequirements
    stats: CandidateStats

    def to(self, device: torch.device, non_blocking: bool) -> "WaveCandidateBatch":
        return WaveCandidateBatch(
            row_env_idx=self.row_env_idx.to(device, non_blocking=non_blocking),
            candidates=self.candidates.to(device, non_blocking=non_blocking),
            proposal_frontier_idx=self.proposal_frontier_idx.to(device, non_blocking=non_blocking),
            proposal_door_variant_idx=self.proposal_door_variant_idx.to(
                device, non_blocking=non_blocking
            ),
            reward_outcomes=self.reward_outcomes.to(device, non_blocking=non_blocking),
            post_candidate_outcomes=self.post_candidate_outcomes.to(
                device, non_blocking=non_blocking
            ),
            feature_requirements=self.feature_requirements,
            stats=self.stats.to(device, non_blocking=non_blocking),
        )


@dataclass
class GenerationGroup:
    env: EnvironmentGroup
    config: GenerateConfig
    feature_slot: FeatureSlot
    candidate_slot: CandidateSlot
    balance_preds: BalancePredictions
    previous_lookahead_outcomes: StepOutcomes | None


@dataclass
class WaveProposalInputs:
    features: Features
    mask: WaveProposalCandidateMask


def create_generation_environment_groups(
    config: Config,
    engine: Engine,
    generation_devices: list[torch.device],
) -> list[list[EnvironmentGroup]]:
    num_generation_groups = config.generation.num_devices * config.generation.pipeline_groups
    generation_group_environments = config.generation.num_environments // num_generation_groups
    generation_group_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.generation.pipeline_groups
    )
    logging.info(
        "Using %s pipeline group(s) per generation device with %s environment(s) and %s Rust worker(s) per group.",
        config.generation.pipeline_groups,
        generation_group_environments,
        generation_group_threads if generation_group_threads is not None else "automatic",
    )
    return [
        [
            engine.create_environment_group(
                config.map_size,
                generation_group_environments,
                config.generation.candidate_spatial_cell_size,
                seed=device_index * config.generation.pipeline_groups + group_index,
                frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
                frontier_neighbor_count=config.generation.frontier_neighbor_count,
                frontier_window_size=config.generation.frontier_window_size,
                num_threads=generation_group_threads,
            )
            for group_index in range(config.generation.pipeline_groups)
        ]
        for device_index in range(len(generation_devices))
    ]


def get_wave_candidate_batch(
    group: GenerationGroup,
    sampled_frontier_idx: torch.Tensor,
    sampled_door_variant_idx: torch.Tensor,
) -> WaveCandidateBatch:
    (
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    ) = group.env.extract_wave_candidates_from_proposals(
        group.candidate_slot,
        sampled_frontier_idx,
        sampled_door_variant_idx,
        group.config.recommended_candidates,
    )
    max_frontiers = sampled_frontier_idx.shape[1]
    row_env_idx = torch.arange(group.env.num_envs, dtype=torch.int64).repeat_interleave(
        max_frontiers
    )
    return WaveCandidateBatch(
        row_env_idx=row_env_idx,
        candidates=candidates,
        proposal_frontier_idx=proposal_frontier_idx,
        proposal_door_variant_idx=proposal_door_variant_idx,
        reward_outcomes=reward_outcomes,
        post_candidate_outcomes=post_candidate_outcomes,
        feature_requirements=feature_requirements,
        stats=stats,
    )


def get_compact_wave_candidate_batch(
    group: GenerationGroup,
    row_env_idx: torch.Tensor,
    sampled_frontier_idx: torch.Tensor,
    sampled_door_variant_idx: torch.Tensor,
) -> WaveCandidateBatch:
    (
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    ) = group.env.extract_compact_wave_candidates_from_proposals(
        group.candidate_slot,
        row_env_idx,
        sampled_frontier_idx,
        sampled_door_variant_idx,
        group.config.recommended_candidates,
    )
    return WaveCandidateBatch(
        row_env_idx=row_env_idx,
        candidates=candidates,
        proposal_frontier_idx=proposal_frontier_idx,
        proposal_door_variant_idx=proposal_door_variant_idx,
        reward_outcomes=reward_outcomes,
        post_candidate_outcomes=post_candidate_outcomes,
        feature_requirements=feature_requirements,
        stats=stats,
    )


def unpack_proposal_mask(mask: ProposalCandidateMask, device: torch.device) -> torch.Tensor:
    packed = mask.mask.to(device)
    shifts = torch.arange(8, device=device, dtype=packed.dtype)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool).flatten(1)
    return bits[:, : mask.door_variant_count]


def row_scores_for_wave_mask(
    proposal_output: torch.nn.Module,
    proposal_state: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_frontier_idx: torch.Tensor,
    wave_mask: WaveProposalCandidateMask,
    device: torch.device,
) -> torch.Tensor:
    proposal_frontier_idx = wave_mask.proposal_frontier_idx.to(device)
    result = torch.full(
        (
            proposal_frontier_idx.shape[0],
            proposal_frontier_idx.shape[1],
            proposal_output.out_features,
        ),
        float("-inf"),
        dtype=proposal_output.output_dtype,
        device=device,
    )
    if proposal_state.shape[0] == 0:
        return result
    row_snapshot_idx = row_snapshot_idx.to(device=device, dtype=torch.int64)
    row_frontier_idx = row_frontier_idx.to(device=device, dtype=torch.int64)
    row_valid = (
        (row_snapshot_idx >= 0)
        & (row_snapshot_idx < proposal_frontier_idx.shape[0])
        & (row_frontier_idx >= 0)
        & (row_frontier_idx < proposal_frontier_idx.shape[1])
        & (proposal_frontier_idx[row_snapshot_idx, row_frontier_idx] >= 0)
    )
    if torch.any(row_valid):
        result[row_snapshot_idx[row_valid], row_frontier_idx[row_valid]] = proposal_output(
            proposal_state[row_valid].to(proposal_output.output_dtype)
        )
    return result


def sample_wave_proposal_shortlist(
    proposal_scores: torch.Tensor,
    wave_mask: WaveProposalCandidateMask,
    config: GenerateConfig,
    proposal_temperature: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    env_count, max_frontiers, door_variant_count = proposal_scores.shape
    if door_variant_count == 0:
        empty_sampled = torch.full(
            (env_count, max_frontiers, config.shortlist_candidates),
            -1,
            dtype=torch.int16,
            device=device,
        )
        return empty_sampled, empty_sampled
    frontier_idx = wave_mask.proposal_frontier_idx.to(device)
    valid_frontier = frontier_idx >= 0
    flat_mask = flatten_wave_mask(wave_mask)
    valid = unpack_proposal_mask(flat_mask, device)[:, :door_variant_count]
    valid = valid.view(env_count, max_frontiers, door_variant_count)
    valid = valid & valid_frontier.unsqueeze(2)
    sample_keys = proposal_scores.to(dtype=torch.float32, copy=True)
    proposal_temperature = proposal_temperature.to(device).view(env_count, 1, 1)
    sample_keys.div_(proposal_temperature.clamp_min(1e-6))
    sample_keys.masked_fill_(~valid, float("-inf"))
    shortlist_candidates = min(config.shortlist_candidates, sample_keys.shape[2])
    gumbel = torch.empty_like(sample_keys).exponential_().log_().neg_()
    sample_keys.add_(gumbel)
    sampled_flat = torch.topk(
        sample_keys,
        shortlist_candidates,
        dim=2,
        sorted=True,
    ).indices
    sampled_is_valid = valid.gather(2, sampled_flat)
    sampled_door_variant_idx = torch.where(
        sampled_is_valid,
        sampled_flat,
        torch.full_like(sampled_flat, -1),
    )
    if shortlist_candidates < config.shortlist_candidates:
        padding = torch.full(
            (
                env_count,
                max_frontiers,
                config.shortlist_candidates - shortlist_candidates,
            ),
            -1,
            dtype=sampled_flat.dtype,
            device=device,
        )
        sampled_door_variant_idx = torch.cat([sampled_door_variant_idx, padding], dim=2)
    sampled_frontier_idx = frontier_idx.unsqueeze(2).expand(
        -1, -1, config.shortlist_candidates
    )
    sampled_frontier_idx = torch.where(
        sampled_door_variant_idx >= 0,
        sampled_frontier_idx,
        torch.full_like(sampled_frontier_idx, -1),
    )
    return sampled_frontier_idx.to(torch.int16), sampled_door_variant_idx.to(torch.int16)


def flatten_wave_mask(mask: WaveProposalCandidateMask) -> ProposalCandidateMask:
    return ProposalCandidateMask(
        proposal_frontier_idx=mask.proposal_frontier_idx.flatten(0, 1),
        mask=mask.mask.flatten(0, 1),
        valid_counts=mask.valid_counts.flatten(0, 1),
        door_variant_count=mask.door_variant_count,
    )


def candidate_log_inputs(
    config: GenerateConfig,
    candidate_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_log_temperature = config.temperature.to(torch.device("cpu")).log().unsqueeze(1)
    candidate_log_temperature = candidate_log_temperature.expand(candidate_shape).contiguous()
    candidate_log_recommended_candidates = torch.full(
        candidate_shape,
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    candidate_generation_variable_floats = (
        config.generation_variable_floats.to(torch.device("cpu"))
        .unsqueeze(1)
        .expand(*candidate_shape, config.generation_variable_floats.shape[1])
        .contiguous()
    )
    return (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_generation_variable_floats,
    )


def compact_candidate_log_inputs(
    config: GenerateConfig,
    row_env_idx: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_env_idx = row_env_idx.to(dtype=torch.int64, device=torch.device("cpu"))
    candidate_log_temperature = (
        config.temperature.to(torch.device("cpu"))[row_env_idx].log().unsqueeze(1)
    )
    candidate_log_temperature = candidate_log_temperature.expand(
        row_env_idx.shape[0],
        candidate_count,
    ).contiguous()
    candidate_log_recommended_candidates = torch.full(
        (row_env_idx.shape[0], candidate_count),
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    candidate_generation_variable_floats = (
        config.generation_variable_floats.to(torch.device("cpu"))[row_env_idx]
        .unsqueeze(1)
        .expand(
            row_env_idx.shape[0],
            candidate_count,
            config.generation_variable_floats.shape[1],
        )
        .contiguous()
    )
    return (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_generation_variable_floats,
    )


def indexed_state_log_inputs(
    config: GenerateConfig,
    env_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    env_idx = env_idx.to(dtype=torch.int64, device=torch.device("cpu"))
    log_temperature = config.temperature.to(torch.device("cpu"))[env_idx].log()
    log_recommended_candidates = torch.full(
        [env_idx.shape[0]],
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    generation_variable_floats = config.generation_variable_floats.to(torch.device("cpu"))[
        env_idx
    ]
    return log_temperature, log_recommended_candidates, generation_variable_floats


def index_step_outcomes(outcomes: StepOutcomes, env_idx: torch.Tensor) -> StepOutcomes:
    env_idx = env_idx.to(dtype=torch.int64, device=outcomes.door_invalid.device)
    return StepOutcomes(
        door_invalid=outcomes.door_invalid[env_idx],
        connection_invalid=outcomes.connection_invalid[env_idx],
        toilet_invalid=outcomes.toilet_invalid[env_idx],
        phantoon_invalid=outcomes.phantoon_invalid[env_idx],
        door_match=outcomes.door_match[env_idx],
    )


def state_log_inputs(
    config: GenerateConfig,
    environment_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_temperature = config.temperature.to(torch.device("cpu")).log()
    log_recommended_candidates = torch.full(
        [environment_count],
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    generation_variable_floats = config.generation_variable_floats.to(torch.device("cpu"))
    return log_temperature, log_recommended_candidates, generation_variable_floats


def reshape_wave_actions_for_env(
    actions: Actions,
    env_count: int,
    max_frontiers: int,
    candidate_count: int,
) -> Actions:
    env_candidate_shape = torch.Size([env_count, max_frontiers * candidate_count])
    return Actions(
        room_idx=actions.room_idx.contiguous().view(env_candidate_shape),
        room_x=actions.room_x.contiguous().view(env_candidate_shape),
        room_y=actions.room_y.contiguous().view(env_candidate_shape),
    )


def reshape_wave_outcomes_for_env(
    outcomes: StepOutcomes,
    env_count: int,
    max_frontiers: int,
    candidate_count: int,
) -> StepOutcomes:
    env_candidate_shape = torch.Size([env_count, max_frontiers * candidate_count])

    def reshape_values(values: torch.Tensor) -> torch.Tensor:
        return values.contiguous().view(*env_candidate_shape, *values.shape[2:])

    return StepOutcomes(
        door_invalid=reshape_values(outcomes.door_invalid),
        connection_invalid=reshape_values(outcomes.connection_invalid),
        toilet_invalid=reshape_values(outcomes.toilet_invalid),
        phantoon_invalid=reshape_values(outcomes.phantoon_invalid),
        door_match=reshape_values(outcomes.door_match),
    )


def reshape_wave_row_outcomes_for_env(
    outcomes: StepOutcomes,
    env_count: int,
    max_frontiers: int,
    candidate_count: int,
) -> StepOutcomes:
    def reshape_values(values: torch.Tensor) -> torch.Tensor:
        env_frontier_shape = torch.Size([env_count, max_frontiers])
        values = values.contiguous().view(*env_frontier_shape, *values.shape[1:])
        tail_shape = values.shape[2:]
        values = values.unsqueeze(2).expand(
            env_count,
            max_frontiers,
            candidate_count,
            *tail_shape,
        )
        return values.contiguous().view(
            env_count,
            max_frontiers * candidate_count,
            *tail_shape,
        )

    return StepOutcomes(
        door_invalid=reshape_values(outcomes.door_invalid),
        connection_invalid=reshape_values(outcomes.connection_invalid),
        toilet_invalid=reshape_values(outcomes.toilet_invalid),
        phantoon_invalid=reshape_values(outcomes.phantoon_invalid),
        door_match=reshape_values(outcomes.door_match),
    )


def mask_finished_wave_proposals(
    proposal_mask: WaveProposalCandidateMask,
    active_env: torch.Tensor,
) -> WaveProposalCandidateMask:
    if active_env.ndim != 1 or active_env.shape[0] != proposal_mask.valid_counts.shape[0]:
        raise ValueError("active environment mask must match proposal environment count")
    inactive_env = ~active_env.to(device=proposal_mask.valid_counts.device, dtype=torch.bool)
    if torch.any(inactive_env):
        proposal_mask.proposal_frontier_idx[inactive_env] = -1
        proposal_mask.mask[inactive_env] = 0
        proposal_mask.valid_counts[inactive_env] = 0
    return proposal_mask


def index_wave_proposal_mask(
    proposal_mask: WaveProposalCandidateMask,
    env_idx: torch.Tensor,
) -> WaveProposalCandidateMask:
    env_idx = env_idx.to(dtype=torch.int64, device=proposal_mask.valid_counts.device)
    return WaveProposalCandidateMask(
        proposal_frontier_idx=proposal_mask.proposal_frontier_idx[env_idx],
        mask=proposal_mask.mask[env_idx],
        valid_counts=proposal_mask.valid_counts[env_idx],
        door_variant_count=proposal_mask.door_variant_count,
        max_frontiers=proposal_mask.max_frontiers,
    )


def prepare_wave_proposal_inputs(
    group: GenerationGroup,
    proposal_mask: WaveProposalCandidateMask,
    active_env_idx: torch.Tensor,
) -> WaveProposalInputs:
    if group.previous_lookahead_outcomes is None:
        raise ValueError("wave proposal features require previous lookahead outcomes")
    lookahead_outcomes = index_step_outcomes(
        group.previous_lookahead_outcomes,
        active_env_idx,
    )
    (
        log_temperature,
        log_recommended_candidates,
        generation_variable_floats,
    ) = indexed_state_log_inputs(group.config, active_env_idx)
    return WaveProposalInputs(
        features=group.env.extract_features_for_env_indices(
            group.feature_slot,
            active_env_idx,
            log_temperature,
            group.env.engine.features.temperature,
            log_recommended_candidates,
            group.env.engine.features.recommended_candidates,
            generation_variable_floats,
            group.env.engine.features.generation_variable_floats,
            lookahead_outcomes,
            group.env.engine.features.lookahead_outcomes,
        ),
        mask=proposal_mask,
    )


def prepare_wave_candidate_features(
    group: GenerationGroup,
    candidate_batch: WaveCandidateBatch,
) -> Features | None:
    candidates = candidate_batch.candidates
    if candidates.room_idx.shape[1] == 1:
        return None
    num_rooms = len(group.env.engine.rooms)
    real_candidate = candidates.room_idx != num_rooms
    if not torch.any(real_candidate):
        return None
    flat_real_candidate = real_candidate.flatten(0, 1)
    if not torch.all(real_candidate):
        compact_row_env_idx = candidate_batch.row_env_idx.unsqueeze(1).expand_as(real_candidate)[
            real_candidate
        ]
        candidates = select_candidate_actions(candidates, flat_real_candidate)
        post_candidate_outcomes = select_candidate_step_outcomes(
            candidate_batch.post_candidate_outcomes,
            flat_real_candidate,
        )
        candidate_count = 1
    else:
        compact_row_env_idx = candidate_batch.row_env_idx
        post_candidate_outcomes = candidate_batch.post_candidate_outcomes
        candidate_count = candidates.room_idx.shape[1]
    (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_generation_variable_floats,
    ) = compact_candidate_log_inputs(
        group.config,
        compact_row_env_idx,
        candidate_count,
    )
    return extract_candidate_features(
        group.env,
        candidates,
        candidate_log_temperature,
        group.env.engine.features.temperature,
        candidate_log_recommended_candidates,
        group.env.engine.features.recommended_candidates,
        candidate_generation_variable_floats,
        group.env.engine.features.generation_variable_floats,
        post_candidate_outcomes,
        group.env.engine.features.lookahead_outcomes,
        candidate_batch.feature_requirements,
        group.feature_slot,
    )


def score_candidate_logits(
    group: GenerationGroup,
    model,
    candidates: Actions,
    outcomes: StepOutcomes,
    post_candidate_door_match: torch.Tensor,
    features: Features,
    device: torch.device,
    num_rooms: int,
    row_env_idx: torch.Tensor,
) -> torch.Tensor:
    environment_count, candidate_count = candidates.room_idx.shape
    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and group.config.autocast,
    ):
        preds = model(features, return_proposal_state=False)
    balance_score = preds.balance_score.view(environment_count, candidate_count, -1)
    row_balance_preds = BalancePredictions(
        left=group.balance_preds.left[row_env_idx],
        right=group.balance_preds.right[row_env_idx],
        up=group.balance_preds.up[row_env_idx],
        down=group.balance_preds.down[row_env_idx],
        toilet_crossed_room=group.balance_preds.toilet_crossed_room[row_env_idx],
    )
    actual_balance_score, actual_balance_score_mask = compute_step_balance_score_target_logits(
        row_balance_preds,
        post_candidate_door_match,
    )
    balance_score = torch.where(actual_balance_score_mask, actual_balance_score, balance_score)
    row_config = index_generate_config(group.config, row_env_idx)
    expected_reward = compute_expected_reward(
        Predictions(
            door_invalid=preds.door_invalid.view(environment_count, candidate_count, -1),
            connection_invalid=preds.connection_invalid.view(
                environment_count,
                candidate_count,
                -1,
            ),
            toilet_invalid=preds.toilet_invalid.view(environment_count, candidate_count),
            phantoon_invalid=preds.phantoon_invalid.view(environment_count, candidate_count),
            balance_score=balance_score,
            toilet_balance_score=preds.toilet_balance_score.view(
                environment_count,
                candidate_count,
            ),
            avg_frontiers=preds.avg_frontiers.view(environment_count, candidate_count),
            graph_diameter=preds.graph_diameter.view(environment_count, candidate_count),
            save_to_room_utility=preds.save_to_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            save_from_room_utility=preds.save_from_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            refill_to_room_utility=preds.refill_to_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            refill_from_room_utility=preds.refill_from_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            missing_connect_utility=preds.missing_connect_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            proposal_score=preds.proposal_score,
            proposal_state=preds.proposal_state,
            proposal_row_snapshot_idx=preds.proposal_row_snapshot_idx,
            proposal_row_frontier_idx=preds.proposal_row_frontier_idx,
        ),
        outcomes,
        row_config,
    )
    temperature = row_config.temperature.to(device)
    candidate_logits = expected_reward / torch.unsqueeze(temperature, 1)
    dummy_candidate = candidates.room_idx == num_rooms
    return torch.where(
        dummy_candidate,
        torch.full_like(candidate_logits, float("-inf")),
        candidate_logits,
    )


def compute_wave_proposal_scores(
    group: GenerationGroup,
    proposal_model,
    features: Features,
    proposal_mask: WaveProposalCandidateMask,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    env_features = transfer_features(features, device, transfer_stream)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and group.config.autocast,
    ):
        preds = proposal_model(env_features, return_proposal_state=True)
    return row_scores_for_wave_mask(
        proposal_model.proposal_output,
        preds.proposal_state,
        preds.proposal_row_snapshot_idx,
        preds.proposal_row_frontier_idx,
        proposal_mask,
        device,
    )


def initial_generation_stats() -> GenerationStats:
    return {
        "wave_groups": 0.0,
        "wave_iterations": 0.0,
        "wave_active_envs": 0.0,
        "wave_env_slots": 0.0,
        "wave_active_frontier_slots": 0.0,
        "wave_dense_frontier_slots": 0.0,
        "wave_valid_proposal_rows": 0.0,
        "wave_candidate_rows": 0.0,
        "wave_candidate_slots": 0.0,
        "wave_real_candidate_slots": 0.0,
        "wave_apply_env_rows": 0.0,
        "wave_applied_env_rows": 0.0,
        "wave_applied_actions": 0.0,
        "proposal_mask_rows": 0.0,
        "proposal_valid_cells": 0.0,
        "proposal_full_set_rows": 0.0,
        "proposal_clean_candidates": 0.0,
        "proposal_evaluated_candidates": 0.0,
        "proposal_rejected_candidates": 0.0,
        "proposal_exhausted_rows": 0.0,
    }


def finalize_generation_stats(stat_totals: GenerationStats) -> GenerationStats:
    proposal_rows = max(stat_totals["proposal_mask_rows"], 1.0)
    evaluated = max(stat_totals["proposal_evaluated_candidates"], 1.0)
    wave_groups = max(stat_totals["wave_groups"], 1.0)
    wave_iterations = max(stat_totals["wave_iterations"], 1.0)
    wave_env_slots = max(stat_totals["wave_env_slots"], 1.0)
    active_frontier_slots = max(stat_totals["wave_active_frontier_slots"], 1.0)
    dense_frontier_slots = max(stat_totals["wave_dense_frontier_slots"], 1.0)
    candidate_slots = max(stat_totals["wave_candidate_slots"], 1.0)
    apply_env_rows = max(stat_totals["wave_apply_env_rows"], 1.0)
    active_env_fraction = stat_totals["wave_active_envs"] / wave_env_slots
    return {
        "wave_iterations": stat_totals["wave_iterations"] / wave_groups,
        "wave_active_env_fraction": active_env_fraction,
        "wave_finished_env_fraction": 1.0 - active_env_fraction,
        "wave_active_envs_per_iteration": stat_totals["wave_active_envs"] / wave_iterations,
        "wave_active_frontier_slots_per_iteration": (
            stat_totals["wave_active_frontier_slots"] / wave_iterations
        ),
        "wave_active_frontier_slot_fraction": (
            stat_totals["wave_active_frontier_slots"] / dense_frontier_slots
        ),
        "wave_valid_proposal_rows_per_iteration": (
            stat_totals["wave_valid_proposal_rows"] / wave_iterations
        ),
        "wave_valid_proposal_row_fraction": (
            stat_totals["wave_valid_proposal_rows"] / active_frontier_slots
        ),
        "wave_candidate_rows_per_iteration": stat_totals["wave_candidate_rows"] / wave_iterations,
        "wave_candidate_row_fraction_of_active_frontiers": (
            stat_totals["wave_candidate_rows"] / active_frontier_slots
        ),
        "wave_real_candidate_slot_fraction": (
            stat_totals["wave_real_candidate_slots"] / candidate_slots
        ),
        "wave_apply_env_rows_per_iteration": (
            stat_totals["wave_apply_env_rows"] / wave_iterations
        ),
        "wave_apply_env_fraction": stat_totals["wave_apply_env_rows"] / wave_env_slots,
        "wave_applied_env_rows_per_iteration": (
            stat_totals["wave_applied_env_rows"] / wave_iterations
        ),
        "wave_applied_env_fraction": stat_totals["wave_applied_env_rows"] / apply_env_rows,
        "wave_applied_actions_per_iteration": (
            stat_totals["wave_applied_actions"] / wave_iterations
        ),
        "proposal_valid_cells": stat_totals["proposal_valid_cells"] / proposal_rows,
        "proposal_full_set_rate": stat_totals["proposal_full_set_rows"] / proposal_rows,
        "proposal_clean_candidates": stat_totals["proposal_clean_candidates"] / proposal_rows,
        "proposal_rejection_rate": stat_totals["proposal_rejected_candidates"] / evaluated,
        "proposal_exhaustion_rate": stat_totals["proposal_exhausted_rows"] / proposal_rows,
    }


def add_wave_candidate_stats(
    stat_totals: GenerationStats,
    mask: WaveProposalCandidateMask,
    stats: CandidateStats,
    shortlist_candidates: int,
    recommended_candidates: int,
    active_env: torch.Tensor,
) -> None:
    valid_counts = mask.valid_counts
    active_rows = active_env.to(device=valid_counts.device, dtype=torch.bool).unsqueeze(1)
    active_rows = active_rows.expand_as(valid_counts)
    active_rows_flat = active_rows.flatten()
    if not torch.any(active_rows):
        return
    active_valid_counts = valid_counts[active_rows]
    stat_totals["proposal_mask_rows"] += float(active_valid_counts.numel())
    stat_totals["proposal_valid_cells"] += float(active_valid_counts.sum().item())
    stat_totals["proposal_full_set_rows"] += float(
        (active_valid_counts <= shortlist_candidates).sum().item()
    )
    stat_totals["proposal_clean_candidates"] += float(
        stats.clean_counts[active_rows_flat].sum().item()
    )
    stat_totals["proposal_evaluated_candidates"] += float(
        stats.evaluated_counts[active_rows_flat].sum().item()
    )
    stat_totals["proposal_rejected_candidates"] += float(
        stats.rejected_counts[active_rows_flat].sum().item()
    )
    stat_totals["proposal_exhausted_rows"] += float(
        (
            (stats.clean_counts[active_rows_flat] < recommended_candidates)
            & (active_valid_counts > shortlist_candidates)
        )
        .sum()
        .item()
    )


def add_compact_wave_candidate_stats(
    stat_totals: GenerationStats,
    valid_counts: torch.Tensor,
    stats: CandidateStats,
    shortlist_candidates: int,
    recommended_candidates: int,
) -> None:
    if valid_counts.numel() == 0:
        return
    stat_totals["proposal_mask_rows"] += float(valid_counts.numel())
    stat_totals["proposal_valid_cells"] += float(valid_counts.sum().item())
    stat_totals["proposal_full_set_rows"] += float(
        (valid_counts <= shortlist_candidates).sum().item()
    )
    stat_totals["proposal_clean_candidates"] += float(stats.clean_counts.sum().item())
    stat_totals["proposal_evaluated_candidates"] += float(stats.evaluated_counts.sum().item())
    stat_totals["proposal_rejected_candidates"] += float(stats.rejected_counts.sum().item())
    stat_totals["proposal_exhausted_rows"] += float(
        ((stats.clean_counts < recommended_candidates) & (valid_counts > shortlist_candidates))
        .sum()
        .item()
    )


def empty_wave_proposal_data(
    max_candidates: int,
    device: torch.device,
) -> WaveProposalData:
    return WaveProposalData(
        episode_idx=torch.empty((0,), dtype=torch.int64, device=device),
        prefix_idx=torch.empty((0,), dtype=torch.int64, device=device),
        frontier_idx=torch.empty((0,), dtype=torch.int16, device=device),
        door_variant_idx=torch.empty((0, max_candidates), dtype=torch.int16, device=device),
        target_logits=torch.empty((0, max_candidates), dtype=torch.float32, device=device),
    )


def wave_candidate_logits(
    group: GenerationGroup,
    model,
    candidate_batch: WaveCandidateBatch,
    features: Features | None,
    device: torch.device,
    num_rooms: int,
) -> torch.Tensor:
    batch = candidate_batch.to(device, non_blocking=False)
    candidates = batch.candidates
    candidate_count = candidates.room_idx.shape[1]
    if features is None:
        logits = torch.zeros(
            candidates.room_idx.shape,
            dtype=torch.float32,
            device=device,
        )
        return torch.where(
            candidates.room_idx == num_rooms,
            torch.full_like(logits, float("-inf")),
            logits,
        )
    device_features = transfer_features(features, device)
    real_candidate = candidates.room_idx != num_rooms
    if not torch.any(real_candidate):
        return torch.full(
            candidates.room_idx.shape,
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )

    if not torch.all(real_candidate):
        flat_real_candidate = real_candidate.flatten(0, 1)
        compact_candidates = select_candidate_actions(candidates, flat_real_candidate)
        compact_post_candidate_outcomes = select_candidate_step_outcomes(
            batch.post_candidate_outcomes,
            flat_real_candidate,
        )
        compact_row_env_idx = (
            batch.row_env_idx.unsqueeze(1).expand_as(real_candidate)[real_candidate]
        )

        def select_reward_values(values: torch.Tensor) -> torch.Tensor:
            values = values.unsqueeze(1)
            values = values.expand(values.shape[0], candidate_count, *values.shape[2:])
            return values.flatten(0, 1)[flat_real_candidate].unsqueeze(1).contiguous()

        compact_reward_outcomes = StepOutcomes(
            door_invalid=select_reward_values(batch.reward_outcomes.door_invalid),
            connection_invalid=select_reward_values(batch.reward_outcomes.connection_invalid),
            toilet_invalid=select_reward_values(batch.reward_outcomes.toilet_invalid),
            phantoon_invalid=select_reward_values(batch.reward_outcomes.phantoon_invalid),
            door_match=batch.reward_outcomes.door_match.new_empty(
                (compact_candidates.room_idx.shape[0], 1, 0)
            ),
        )
        compact_logits = score_candidate_logits(
            group,
            model,
            compact_candidates,
            compact_reward_outcomes,
            compact_post_candidate_outcomes.door_match,
            device_features,
            device,
            num_rooms,
            compact_row_env_idx.to(device=device, dtype=torch.int64),
        )
        logits = torch.full(
            candidates.room_idx.shape,
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        flat_logits = logits.flatten(0, 1)
        flat_logits[flat_real_candidate] = compact_logits.flatten(0, 1)
        return logits

    def expand_row_values(values: torch.Tensor) -> torch.Tensor:
        values = values.unsqueeze(1)
        return values.expand(values.shape[0], candidate_count, *values.shape[2:]).contiguous()

    row_reward_outcomes = StepOutcomes(
        door_invalid=expand_row_values(batch.reward_outcomes.door_invalid),
        connection_invalid=expand_row_values(batch.reward_outcomes.connection_invalid),
        toilet_invalid=expand_row_values(batch.reward_outcomes.toilet_invalid),
        phantoon_invalid=expand_row_values(batch.reward_outcomes.phantoon_invalid),
        door_match=batch.reward_outcomes.door_match.new_empty(
            (candidates.room_idx.shape[0], candidate_count, 0)
        ),
    )
    row_logits = score_candidate_logits(
        group,
        model,
        candidates,
        row_reward_outcomes,
        batch.post_candidate_outcomes.door_match,
        device_features,
        device,
        num_rooms,
        batch.row_env_idx.to(device=device, dtype=torch.int64),
    )
    return row_logits


def sorted_compact_wave_candidates(
    candidate_batch: WaveCandidateBatch,
    candidate_logits: torch.Tensor,
    env_count: int,
    num_rooms: int,
) -> tuple[torch.Tensor, torch.Tensor, Actions, torch.Tensor]:
    row_env_idx = candidate_batch.row_env_idx.to(dtype=torch.int64, device=torch.device("cpu"))
    candidate_count = candidate_batch.candidates.room_idx.shape[1]
    attempts_per_env = torch.bincount(row_env_idx, minlength=env_count) * candidate_count
    apply_env_idx = torch.nonzero(attempts_per_env > 0, as_tuple=False).flatten()
    max_attempts = int(attempts_per_env.max().item()) if attempts_per_env.numel() else 0
    if max_attempts == 0:
        return (
            apply_env_idx,
            torch.full((0, 0), -1, dtype=torch.int16),
            Actions(
                room_idx=torch.full((0, 0), num_rooms, dtype=torch.uint8),
                room_x=torch.zeros((0, 0), dtype=torch.int8),
                room_y=torch.zeros((0, 0), dtype=torch.int8),
            ),
            torch.zeros((0, 0), dtype=torch.int8),
        )
    apply_env_count = int(apply_env_idx.shape[0])
    sorted_frontier_idx = torch.full((apply_env_count, max_attempts), -1, dtype=torch.int16)
    sorted_actions = Actions(
        room_idx=torch.full(
            (apply_env_count, max_attempts),
            num_rooms,
            dtype=candidate_batch.candidates.room_idx.dtype,
        ),
        room_x=torch.zeros(
            (apply_env_count, max_attempts),
            dtype=candidate_batch.candidates.room_x.dtype,
        ),
        room_y=torch.zeros(
            (apply_env_count, max_attempts),
            dtype=candidate_batch.candidates.room_y.dtype,
        ),
    )
    env_to_apply_row = torch.full((env_count,), -1, dtype=torch.int64)
    env_to_apply_row[apply_env_idx] = torch.arange(apply_env_count, dtype=torch.int64)
    flat_logits = candidate_logits.to(torch.device("cpu")).flatten()
    flat_frontier_idx = candidate_batch.proposal_frontier_idx.flatten()
    flat_room_idx = candidate_batch.candidates.room_idx.flatten()
    flat_room_x = candidate_batch.candidates.room_x.flatten()
    flat_room_y = candidate_batch.candidates.room_y.flatten()
    candidate_slot_idx = torch.arange(candidate_count, dtype=torch.int64).expand(
        candidate_batch.candidates.room_idx.shape[0],
        candidate_count,
    )
    flat_candidate_clean = (
        candidate_slot_idx < candidate_batch.stats.clean_counts.unsqueeze(1)
    ).flatten()
    flat_env_idx = row_env_idx.repeat_interleave(candidate_count)
    sorted_candidate_clean = torch.zeros((apply_env_count, max_attempts), dtype=torch.int8)
    score_order = torch.argsort(flat_logits, descending=True, stable=True)
    env_order = torch.argsort(flat_env_idx[score_order], stable=True)
    attempt_idx = score_order[env_order]
    sorted_env_idx = flat_env_idx[attempt_idx]
    env_counts = attempts_per_env[apply_env_idx]
    env_start = torch.cumsum(env_counts, dim=0) - env_counts
    attempt_position = torch.arange(attempt_idx.shape[0], dtype=torch.int64) - torch.repeat_interleave(
        env_start,
        env_counts,
    )
    apply_row_idx = env_to_apply_row[sorted_env_idx]
    sorted_frontier_idx[apply_row_idx, attempt_position] = flat_frontier_idx[attempt_idx]
    sorted_actions.room_idx[apply_row_idx, attempt_position] = flat_room_idx[attempt_idx]
    sorted_actions.room_x[apply_row_idx, attempt_position] = flat_room_x[attempt_idx]
    sorted_actions.room_y[apply_row_idx, attempt_position] = flat_room_y[attempt_idx]
    sorted_candidate_clean[apply_row_idx, attempt_position] = flat_candidate_clean[
        attempt_idx
    ].to(torch.int8)
    return apply_env_idx, sorted_frontier_idx, sorted_actions, sorted_candidate_clean


def append_applied_wave_actions(
    actions: Actions,
    action_counts: torch.Tensor,
    applied_env_idx: torch.Tensor,
    applied_actions: Actions,
    applied_counts: torch.Tensor,
) -> None:
    if applied_counts.numel() == 0 or applied_actions.room_idx.shape[1] == 0:
        return
    max_count = applied_actions.room_idx.shape[1]
    slot_idx = torch.arange(max_count, dtype=torch.int64)
    remaining = actions.room_idx.shape[1] - action_counts[applied_env_idx]
    clipped_counts = torch.minimum(applied_counts, remaining)
    applied_mask = slot_idx.unsqueeze(0) < clipped_counts.unsqueeze(1)
    if not torch.any(applied_mask):
        return
    dst_env_idx = applied_env_idx.unsqueeze(1).expand(-1, max_count)[applied_mask]
    dst_action_idx = (
        action_counts[applied_env_idx].unsqueeze(1) + slot_idx.unsqueeze(0)
    )[applied_mask]
    actions.room_idx[dst_env_idx, dst_action_idx] = applied_actions.room_idx[applied_mask]
    actions.room_x[dst_env_idx, dst_action_idx] = applied_actions.room_x[applied_mask]
    actions.room_y[dst_env_idx, dst_action_idx] = applied_actions.room_y[applied_mask]
    action_counts[applied_env_idx] += clipped_counts


def wave_frame_data_from_prefixes(
    wave_frame_prefixes: list[torch.Tensor],
    env_count: int,
    device: torch.device,
) -> WaveFrameData:
    if not wave_frame_prefixes:
        return WaveFrameData(
            prefix_counts=torch.zeros((env_count, 0), dtype=torch.int64, device=device),
            frame_counts=torch.zeros(env_count, dtype=torch.int64, device=device),
        )
    prefixes = torch.stack(wave_frame_prefixes, dim=1)
    episode_prefixes = []
    frame_counts = []
    for env_idx in range(env_count):
        compacted = []
        previous_prefix = None
        for prefix in prefixes[env_idx].tolist():
            prefix = int(prefix)
            if previous_prefix != prefix:
                compacted.append(prefix)
                previous_prefix = prefix
        episode_prefixes.append(compacted)
        frame_counts.append(len(compacted))
    max_frames = max(frame_counts)
    prefix_counts = torch.zeros((env_count, max_frames), dtype=torch.int64, device=device)
    for env_idx, compacted in enumerate(episode_prefixes):
        if compacted:
            prefix_counts[env_idx, : len(compacted)] = torch.tensor(
                compacted,
                dtype=torch.int64,
                device=device,
            )
    return WaveFrameData(
        prefix_counts=prefix_counts,
        frame_counts=torch.tensor(frame_counts, dtype=torch.int64, device=device),
    )


def bootstrap_lookahead_outcomes(outcomes: StepOutcomes) -> StepOutcomes:
    return StepOutcomes(
        door_invalid=outcomes.door_invalid,
        connection_invalid=outcomes.connection_invalid,
        toilet_invalid=outcomes.toilet_invalid,
        phantoon_invalid=outcomes.phantoon_invalid,
        door_match=torch.full_like(outcomes.door_invalid, -1, dtype=torch.int16),
    )


def run_wave_generation_groups(
    envs: list[EnvironmentGroup],
    model,
    proposal_model,
    balance_model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool,
    profile: bool,
) -> tuple[
    EpisodeData,
    EpisodeOutcomes,
    DoorMatchCounts,
    WaveProposalData,
    WaveFrameData,
    GenerationStats,
    ProfileReport,
]:
    if not envs or len(envs) != len(configs):
        raise ValueError("wave generation groups require one config per environment group")
    profiler = GenerationProfiler(profile)
    num_rooms = len(envs[0].engine.rooms)
    groups = [
        GenerationGroup(
            env=env,
            config=config,
            feature_slot=FeatureSlot(env, pin_memory=device.type == "cuda"),
            candidate_slot=CandidateSlot(env, pin_memory=device.type == "cuda"),
            balance_preds=balance_model(config.generation_variable_floats),
            previous_lookahead_outcomes=None,
        )
        for env, config in zip(envs, configs)
    ]
    stat_totals = initial_generation_stats()
    results = []
    wave_prefix_idx = []
    wave_episode_idx = []
    wave_frontier_idx = []
    wave_door_variant_idx = []
    wave_target_logits = []
    with torch.no_grad():
        group_episode_offset = 0
        for group in groups:
            profile_time = profile_start(profile)
            group.env.clear()
            group.env.step_initial()
            initial_actions = group.env.get_actions(torch.device("cpu"))
            action_dtype = initial_actions.room_idx.dtype
            actions = Actions(
                room_idx=torch.full(
                    (group.env.num_envs, group.config.episode_length),
                    num_rooms,
                    dtype=action_dtype,
                    device=torch.device("cpu"),
                ),
                room_x=torch.zeros(
                    (group.env.num_envs, group.config.episode_length),
                    dtype=initial_actions.room_x.dtype,
                    device=torch.device("cpu"),
                ),
                room_y=torch.zeros(
                    (group.env.num_envs, group.config.episode_length),
                    dtype=initial_actions.room_y.dtype,
                    device=torch.device("cpu"),
                ),
            )
            actions.room_idx[:, 0] = initial_actions.room_idx[:, 0]
            actions.room_x[:, 0] = initial_actions.room_x[:, 0]
            actions.room_y[:, 0] = initial_actions.room_y[:, 0]
            action_counts = torch.ones(group.env.num_envs, dtype=torch.int64)
            active_env = action_counts < group.config.episode_length
            wave_frame_prefixes = [action_counts.clone()]
            group.previous_lookahead_outcomes = bootstrap_lookahead_outcomes(
                group.env.get_outcomes(
                    torch.device("cpu"),
                    verify_consistency=False,
                ).step_outcomes
            )
            profiler.add("python.wave.initialize_group", profile_time)
            stat_totals["wave_groups"] += 1.0

            while torch.any(active_env):
                profile_time = profile_start(profile)
                proposal_mask = mask_finished_wave_proposals(
                    group.env.get_all_proposal_candidate_masks(torch.device("cpu")),
                    active_env,
                )
                profiler.add("python.wave.prepare_proposal_mask", profile_time)
                proposal_active_env = torch.any(proposal_mask.valid_counts > 0, dim=1)
                active_env = active_env & proposal_active_env
                if not torch.any(active_env):
                    break
                proposal_mask = mask_finished_wave_proposals(proposal_mask, active_env)
                active_env_idx = torch.nonzero(active_env, as_tuple=False).flatten()
                active_proposal_mask = index_wave_proposal_mask(proposal_mask, active_env_idx)
                stat_totals["wave_iterations"] += 1.0
                stat_totals["wave_active_envs"] += float(active_env.sum().item())
                stat_totals["wave_env_slots"] += float(proposal_active_env.numel())
                stat_totals["wave_dense_frontier_slots"] += float(proposal_mask.valid_counts.numel())
                stat_totals["wave_active_frontier_slots"] += float(
                    active_proposal_mask.valid_counts.numel()
                )
                profile_time = profile_start(profile)
                proposal_inputs = prepare_wave_proposal_inputs(
                    group,
                    active_proposal_mask,
                    active_env_idx,
                )
                profiler.add("python.wave.prepare_proposal_features", profile_time)
                profile_time = profile_start(profile)
                proposal_scores = compute_wave_proposal_scores(
                    group,
                    proposal_model,
                    proposal_inputs.features,
                    proposal_inputs.mask,
                    device,
                    None,
                )
                sync_profile_device(device, profile)
                profiler.add("python.wave.score_proposals", profile_time)
                profile_time = profile_start(profile)
                sampled_frontier_idx, sampled_door_variant_idx = sample_wave_proposal_shortlist(
                    proposal_scores,
                    proposal_inputs.mask,
                    group.config,
                    group.config.proposal_temperature[active_env_idx],
                    device,
                )
                sync_profile_device(device, profile)
                profiler.add("python.wave.sample_proposals", profile_time)
                profile_time = profile_start(profile)
                valid_proposal_rows = proposal_inputs.mask.valid_counts.flatten() > 0
                stat_totals["wave_valid_proposal_rows"] += float(
                    valid_proposal_rows.sum().item()
                )
                flat_row_env_idx = active_env_idx.repeat_interleave(
                    proposal_inputs.mask.max_frontiers
                )
                compact_row_env_idx = flat_row_env_idx[valid_proposal_rows]
                compact_sampled_frontier_idx = sampled_frontier_idx.to(
                    torch.device("cpu")
                ).flatten(0, 1)[valid_proposal_rows]
                compact_sampled_door_variant_idx = sampled_door_variant_idx.to(
                    torch.device("cpu")
                ).flatten(0, 1)[valid_proposal_rows]
                compact_valid_counts = proposal_inputs.mask.valid_counts.flatten()[
                    valid_proposal_rows
                ]
                candidate_batch = get_compact_wave_candidate_batch(
                    group,
                    compact_row_env_idx,
                    compact_sampled_frontier_idx,
                    compact_sampled_door_variant_idx,
                )
                real_candidate_slots = candidate_batch.candidates.room_idx != num_rooms
                stat_totals["wave_candidate_rows"] += float(candidate_batch.row_env_idx.numel())
                stat_totals["wave_candidate_slots"] += float(
                    candidate_batch.candidates.room_idx.numel()
                )
                stat_totals["wave_real_candidate_slots"] += float(
                    real_candidate_slots.sum().item()
                )
                candidate_features = prepare_wave_candidate_features(group, candidate_batch)
                profiler.add("python.wave.prepare_candidates", profile_time)
                profile_time = profile_start(profile)
                logits = wave_candidate_logits(
                    group,
                    model,
                    candidate_batch,
                    candidate_features,
                    device,
                    num_rooms,
                )
                sync_profile_device(device, profile)
                profiler.add("python.wave.score_candidates", profile_time)

                target_logits = logits.to(torch.device("cpu"), dtype=torch.float32)
                row_has_target = torch.any(
                    (candidate_batch.proposal_door_variant_idx >= 0)
                    & torch.isfinite(target_logits),
                    dim=1,
                )
                if torch.any(row_has_target):
                    row_prefix_idx = action_counts[candidate_batch.row_env_idx]
                    row_episode_idx = group_episode_offset + candidate_batch.row_env_idx
                    wave_episode_idx.append(row_episode_idx[row_has_target])
                    wave_prefix_idx.append(row_prefix_idx[row_has_target])
                    wave_frontier_idx.append(
                        candidate_batch.proposal_frontier_idx[:, 0][row_has_target]
                    )
                    wave_door_variant_idx.append(
                        candidate_batch.proposal_door_variant_idx[row_has_target]
                    )
                    wave_target_logits.append(target_logits[row_has_target])

                add_compact_wave_candidate_stats(
                    stat_totals,
                    compact_valid_counts,
                    candidate_batch.stats,
                    group.config.shortlist_candidates,
                    group.config.recommended_candidates,
                )
                profile_time = profile_start(profile)
                sync_profile_device(device, profile)
                profiler.add("python.wave.sync_candidate_logits", profile_time)
                profile_time = profile_start(profile)
                (
                    apply_env_idx,
                    sorted_frontier_idx,
                    sorted_actions,
                    sorted_candidate_clean,
                ) = sorted_compact_wave_candidates(
                    candidate_batch,
                    logits,
                    group.env.num_envs,
                    num_rooms,
                )
                stat_totals["wave_apply_env_rows"] += float(apply_env_idx.numel())
                profiler.add("python.wave.sort_candidates", profile_time)
                profile_time = profile_start(profile)
                applied_actions, applied_counts = group.env.apply_compact_wave_candidates(
                    apply_env_idx,
                    sorted_frontier_idx,
                    sorted_actions,
                    sorted_candidate_clean,
                )
                stat_totals["wave_applied_env_rows"] += float((applied_counts > 0).sum().item())
                stat_totals["wave_applied_actions"] += float(applied_counts.sum().item())
                profiler.add("python.wave.apply_candidates", profile_time)
                profile_time = profile_start(profile)
                append_applied_wave_actions(
                    actions,
                    action_counts,
                    apply_env_idx,
                    applied_actions,
                    applied_counts,
                )
                profiler.add("python.wave.append_actions", profile_time)
                active_env = active_env & (action_counts < group.config.episode_length)
                if not torch.any(applied_counts > 0):
                    break
                wave_frame_prefixes.append(action_counts.clone())
                profile_time = profile_start(profile)
                group.previous_lookahead_outcomes = bootstrap_lookahead_outcomes(
                    group.env.get_outcomes(
                        torch.device("cpu"),
                        verify_consistency=verify_outcome_consistency,
                    ).step_outcomes
                )
                profiler.add("python.wave.refresh_outcomes", profile_time)

            profile_time = profile_start(profile)
            group.env.finish()
            episode_outcomes = group.env.get_outcomes(
                device,
                verify_consistency=verify_outcome_consistency,
            )
            door_match_counts = group.env.get_door_match_counts(device)
            wave_frame_data = wave_frame_data_from_prefixes(
                wave_frame_prefixes,
                group.env.num_envs,
                device,
            )
            results.append(
                (
                    EpisodeData(
                        actions=actions.to(device),
                        temperature=group.config.temperature,
                        recommended_candidates=torch.full_like(
                            group.config.temperature,
                            group.config.recommended_candidates,
                            dtype=torch.float32,
                        ),
                        generation_variable_floats=group.config.generation_variable_floats,
                    ),
                    episode_outcomes,
                    door_match_counts,
                    wave_frame_data,
                )
            )
            profiler.add("python.wave.finish_group", profile_time)
            group_episode_offset += group.env.num_envs

    episode_data = EpisodeData(
        actions=Actions(
            room_idx=torch.cat([episode.actions.room_idx for episode, _, _, _ in results]),
            room_x=torch.cat([episode.actions.room_x for episode, _, _, _ in results]),
            room_y=torch.cat([episode.actions.room_y for episode, _, _, _ in results]),
        ),
        temperature=torch.cat([episode.temperature for episode, _, _, _ in results]),
        recommended_candidates=torch.cat(
            [episode.recommended_candidates for episode, _, _, _ in results]
        ),
        generation_variable_floats=torch.cat(
            [episode.generation_variable_floats for episode, _, _, _ in results]
        ),
    )
    outcomes = EpisodeOutcomes(
        step_outcomes=StepOutcomes(
            door_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.door_invalid
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            connection_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.connection_invalid
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            toilet_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.toilet_invalid
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            phantoon_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.phantoon_invalid
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            door_match=torch.cat(
                [
                    episode_outcomes.step_outcomes.door_match
                    for _, episode_outcomes, _, _ in results
                ]
            ),
        ),
        end_outcomes=EndOutcomes(
            toilet_crossed_room_idx=torch.cat(
                [
                    episode_outcomes.end_outcomes.toilet_crossed_room_idx
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            avg_frontiers=torch.cat(
                [
                    episode_outcomes.end_outcomes.avg_frontiers
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            graph_diameter=torch.cat(
                [
                    episode_outcomes.end_outcomes.graph_diameter
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            active_room_part_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.active_room_part_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            save_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            save_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            save_to_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_to_room_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            save_to_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_to_room_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            save_from_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_from_room_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            save_from_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_from_room_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            refill_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            refill_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            refill_to_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_to_room_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            refill_to_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_to_room_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            refill_from_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_from_room_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            refill_from_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_from_room_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            missing_connect_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.missing_connect_distance
                    for _, episode_outcomes, _, _ in results
                ]
            ),
            missing_connect_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.missing_connect_distance_mask
                    for _, episode_outcomes, _, _ in results
                ]
            ),
        ),
    )
    door_match_counts = DoorMatchCounts(
        horizontal=torch.sum(
            torch.stack([counts.horizontal for _, _, counts, _ in results]), dim=0
        ),
        vertical=torch.sum(
            torch.stack([counts.vertical for _, _, counts, _ in results]), dim=0
        ),
    )
    wave_frame_data = merge_wave_frame_data([frame_data for _, _, _, frame_data in results])
    if wave_prefix_idx:
        proposal_data = WaveProposalData(
            episode_idx=torch.cat(wave_episode_idx).to(device),
            prefix_idx=torch.cat(wave_prefix_idx).to(device),
            frontier_idx=torch.cat(wave_frontier_idx).to(device),
            door_variant_idx=torch.cat(wave_door_variant_idx).to(device),
            target_logits=torch.cat(wave_target_logits).to(device),
        )
    else:
        proposal_data = empty_wave_proposal_data(configs[0].recommended_candidates, device)
    return (
        episode_data,
        outcomes,
        door_match_counts,
        proposal_data,
        wave_frame_data,
        finalize_generation_stats(stat_totals),
        profiler.report(),
    )
