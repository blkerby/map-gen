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
    StepOutcomes,
    ProposalCandidateMask,
    WaveProposalData,
    WaveProposalCandidateMask,
    FeatureRequirements,
    FeatureSlot,
    Features,
    extract_candidate_features,
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
    proposal_temperature = config.proposal_temperature.to(device).view(env_count, 1, 1)
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


def prepare_wave_proposal_inputs(group: GenerationGroup) -> WaveProposalInputs:
    proposal_mask = group.env.get_all_proposal_candidate_masks(torch.device("cpu"))
    if group.previous_lookahead_outcomes is None:
        raise ValueError("wave proposal features require previous lookahead outcomes")
    environment_count = group.config.temperature.shape[0]
    (
        log_temperature,
        log_recommended_candidates,
        generation_variable_floats,
    ) = state_log_inputs(group.config, environment_count)
    return WaveProposalInputs(
        features=group.env.extract_features(
            group.feature_slot,
            log_temperature,
            group.env.engine.features.temperature,
            log_recommended_candidates,
            group.env.engine.features.recommended_candidates,
            generation_variable_floats,
            group.env.engine.features.generation_variable_floats,
            group.previous_lookahead_outcomes,
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
    env_count = group.env.num_envs
    if candidates.room_idx.shape[0] % env_count != 0:
        raise ValueError("wave candidate rows must be divisible by environment count")
    candidate_count = candidates.room_idx.shape[1]
    max_frontiers = candidates.room_idx.shape[0] // env_count
    env_candidate_shape = torch.Size([env_count, max_frontiers * candidate_count])
    env_candidates = Actions(
        room_idx=candidates.room_idx.contiguous().view(env_candidate_shape),
        room_x=candidates.room_x.contiguous().view(env_candidate_shape),
        room_y=candidates.room_y.contiguous().view(env_candidate_shape),
    )
    (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_generation_variable_floats,
    ) = candidate_log_inputs(
        group.config,
        env_candidate_shape,
    )
    post_candidate_outcomes = reshape_wave_outcomes_for_env(
        candidate_batch.post_candidate_outcomes,
        env_count,
        max_frontiers,
        candidate_count,
    )
    return extract_candidate_features(
        group.env,
        env_candidates,
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
    actual_balance_score, actual_balance_score_mask = compute_step_balance_score_target_logits(
        group.balance_preds,
        post_candidate_door_match,
    )
    actual_balance_score = actual_balance_score[row_env_idx]
    actual_balance_score_mask = actual_balance_score_mask[row_env_idx]
    balance_score = torch.where(actual_balance_score_mask, actual_balance_score, balance_score)
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
        group.config,
    )
    temperature = group.config.temperature.to(device)
    temperature = temperature[row_env_idx]
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
    return {
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
) -> None:
    valid_counts = mask.valid_counts
    stat_totals["proposal_mask_rows"] += float(valid_counts.numel())
    stat_totals["proposal_valid_cells"] += float(valid_counts.sum().item())
    stat_totals["proposal_full_set_rows"] += float(
        (valid_counts <= shortlist_candidates).sum().item()
    )
    stat_totals["proposal_clean_candidates"] += float(stats.clean_counts.sum().item())
    stat_totals["proposal_evaluated_candidates"] += float(stats.evaluated_counts.sum().item())
    stat_totals["proposal_rejected_candidates"] += float(stats.rejected_counts.sum().item())
    stat_totals["proposal_exhausted_rows"] += float(
        (
            (stats.clean_counts < recommended_candidates)
            & (valid_counts.flatten() > shortlist_candidates)
        )
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
    env_count = group.env.num_envs
    if candidates.room_idx.shape[0] % env_count != 0:
        raise ValueError("wave candidate rows must be divisible by environment count")
    candidate_count = candidates.room_idx.shape[1]
    max_frontiers = candidates.room_idx.shape[0] // env_count
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
    env_candidates = reshape_wave_actions_for_env(
        candidates,
        env_count,
        max_frontiers,
        candidate_count,
    )
    env_reward_outcomes = reshape_wave_row_outcomes_for_env(
        batch.reward_outcomes,
        env_count,
        max_frontiers,
        candidate_count,
    )
    env_post_candidate_outcomes = reshape_wave_outcomes_for_env(
        batch.post_candidate_outcomes,
        env_count,
        max_frontiers,
        candidate_count,
    )
    row_env_idx = torch.arange(env_count, dtype=torch.int64, device=device)
    env_logits = score_candidate_logits(
        group,
        model,
        env_candidates,
        env_reward_outcomes,
        env_post_candidate_outcomes.door_match,
        device_features,
        device,
        num_rooms,
        row_env_idx,
    )
    return env_logits.view(env_count, max_frontiers, candidate_count).flatten(0, 1)


def sorted_wave_candidates(
    candidate_batch: WaveCandidateBatch,
    candidate_logits: torch.Tensor,
    env_count: int,
    max_frontiers: int,
    num_rooms: int,
    action_counts: torch.Tensor,
    episode_length: int,
) -> tuple[torch.Tensor, Actions]:
    candidate_count = candidate_logits.shape[1]
    sorted_idx = torch.argsort(
        candidate_logits.view(env_count, max_frontiers * candidate_count),
        dim=1,
        descending=True,
    ).to(torch.device("cpu"))

    def gather_rows(values: torch.Tensor) -> torch.Tensor:
        flat = values.view(env_count, max_frontiers * candidate_count)
        return torch.gather(flat, 1, sorted_idx.to(flat.device)).to(torch.device("cpu"))

    sorted_frontier_idx = gather_rows(candidate_batch.proposal_frontier_idx)
    sorted_actions = Actions(
        room_idx=gather_rows(candidate_batch.candidates.room_idx),
        room_x=gather_rows(candidate_batch.candidates.room_x),
        room_y=gather_rows(candidate_batch.candidates.room_y),
    )
    slot_idx = torch.arange(sorted_idx.shape[1], dtype=torch.int64).unsqueeze(0)
    remaining = (episode_length - action_counts).clamp_min(0).unsqueeze(1)
    over_limit = slot_idx >= remaining
    sorted_frontier_idx = torch.where(
        over_limit,
        torch.full_like(sorted_frontier_idx, -1),
        sorted_frontier_idx,
    )
    sorted_actions = Actions(
        room_idx=torch.where(
            over_limit,
            torch.full_like(sorted_actions.room_idx, num_rooms),
            sorted_actions.room_idx,
        ),
        room_x=sorted_actions.room_x,
        room_y=sorted_actions.room_y,
    )
    return sorted_frontier_idx, sorted_actions


def append_applied_wave_actions(
    actions: Actions,
    action_counts: torch.Tensor,
    applied_actions: Actions,
    applied_counts: torch.Tensor,
) -> None:
    for env_idx, applied_count in enumerate(applied_counts.tolist()):
        start = int(action_counts[env_idx].item())
        count = min(applied_count, actions.room_idx.shape[1] - start)
        if count <= 0:
            continue
        end = start + count
        actions.room_idx[env_idx, start:end] = applied_actions.room_idx[env_idx, :count]
        actions.room_x[env_idx, start:end] = applied_actions.room_x[env_idx, :count]
        actions.room_y[env_idx, start:end] = applied_actions.room_y[env_idx, :count]
        action_counts[env_idx] += count


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
    EpisodeData, EpisodeOutcomes, DoorMatchCounts, WaveProposalData, GenerationStats, ProfileReport
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
            group.previous_lookahead_outcomes = bootstrap_lookahead_outcomes(
                group.env.get_outcomes(
                    torch.device("cpu"),
                    verify_consistency=False,
                ).step_outcomes
            )
            profiler.add("python.wave.initialize_group", profile_time)

            while torch.any(action_counts < group.config.episode_length):
                profile_time = profile_start(profile)
                proposal_inputs = prepare_wave_proposal_inputs(group)
                profiler.add("python.wave.prepare_proposal", profile_time)
                if proposal_inputs.mask.valid_counts.sum().item() == 0:
                    break
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
                    device,
                )
                sync_profile_device(device, profile)
                profiler.add("python.wave.sample_proposals", profile_time)
                profile_time = profile_start(profile)
                candidate_batch = get_wave_candidate_batch(
                    group,
                    sampled_frontier_idx.to(torch.device("cpu")),
                    sampled_door_variant_idx.to(torch.device("cpu")),
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
                        proposal_inputs.mask.proposal_frontier_idx.flatten()[row_has_target]
                    )
                    wave_door_variant_idx.append(
                        candidate_batch.proposal_door_variant_idx[row_has_target]
                    )
                    wave_target_logits.append(target_logits[row_has_target])

                add_wave_candidate_stats(
                    stat_totals,
                    proposal_inputs.mask,
                    candidate_batch.stats,
                    group.config.shortlist_candidates,
                    group.config.recommended_candidates,
                )
                profile_time = profile_start(profile)
                sorted_frontier_idx, sorted_actions = sorted_wave_candidates(
                    candidate_batch,
                    logits,
                    group.env.num_envs,
                    proposal_inputs.mask.max_frontiers,
                    num_rooms,
                    action_counts,
                    group.config.episode_length,
                )
                applied_actions, applied_counts = group.env.apply_wave_candidates(
                    sorted_frontier_idx,
                    sorted_actions,
                )
                append_applied_wave_actions(
                    actions,
                    action_counts,
                    applied_actions,
                    applied_counts,
                )
                profiler.add("python.wave.apply_candidates", profile_time)
                if not torch.any(applied_counts > 0):
                    break
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
                )
            )
            profiler.add("python.wave.finish_group", profile_time)
            group_episode_offset += group.env.num_envs

    episode_data = EpisodeData(
        actions=Actions(
            room_idx=torch.cat([episode.actions.room_idx for episode, _, _ in results]),
            room_x=torch.cat([episode.actions.room_x for episode, _, _ in results]),
            room_y=torch.cat([episode.actions.room_y for episode, _, _ in results]),
        ),
        temperature=torch.cat([episode.temperature for episode, _, _ in results]),
        recommended_candidates=torch.cat(
            [episode.recommended_candidates for episode, _, _ in results]
        ),
        generation_variable_floats=torch.cat(
            [episode.generation_variable_floats for episode, _, _ in results]
        ),
    )
    outcomes = EpisodeOutcomes(
        step_outcomes=StepOutcomes(
            door_invalid=torch.cat(
                [episode_outcomes.step_outcomes.door_invalid for _, episode_outcomes, _ in results]
            ),
            connection_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.connection_invalid
                    for _, episode_outcomes, _ in results
                ]
            ),
            toilet_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.toilet_invalid
                    for _, episode_outcomes, _ in results
                ]
            ),
            phantoon_invalid=torch.cat(
                [
                    episode_outcomes.step_outcomes.phantoon_invalid
                    for _, episode_outcomes, _ in results
                ]
            ),
            door_match=torch.cat(
                [episode_outcomes.step_outcomes.door_match for _, episode_outcomes, _ in results]
            ),
        ),
        end_outcomes=EndOutcomes(
            toilet_crossed_room_idx=torch.cat(
                [
                    episode_outcomes.end_outcomes.toilet_crossed_room_idx
                    for _, episode_outcomes, _ in results
                ]
            ),
            avg_frontiers=torch.cat(
                [episode_outcomes.end_outcomes.avg_frontiers for _, episode_outcomes, _ in results]
            ),
            graph_diameter=torch.cat(
                [
                    episode_outcomes.end_outcomes.graph_diameter
                    for _, episode_outcomes, _ in results
                ]
            ),
            active_room_part_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.active_room_part_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            save_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            save_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            save_to_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_to_room_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            save_to_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_to_room_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            save_from_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_from_room_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            save_from_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.save_from_room_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            refill_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            refill_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            refill_to_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_to_room_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            refill_to_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_to_room_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            refill_from_room_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_from_room_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            refill_from_room_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.refill_from_room_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
            missing_connect_distance=torch.cat(
                [
                    episode_outcomes.end_outcomes.missing_connect_distance
                    for _, episode_outcomes, _ in results
                ]
            ),
            missing_connect_distance_mask=torch.cat(
                [
                    episode_outcomes.end_outcomes.missing_connect_distance_mask
                    for _, episode_outcomes, _ in results
                ]
            ),
        ),
    )
    door_match_counts = DoorMatchCounts(
        horizontal=torch.sum(torch.stack([counts.horizontal for _, _, counts in results]), dim=0),
        vertical=torch.sum(torch.stack([counts.vertical for _, _, counts in results]), dim=0),
    )
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
        finalize_generation_stats(stat_totals),
        profiler.report(),
    )
