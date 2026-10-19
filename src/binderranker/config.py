"""Immutable scientific configuration profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FROZEN_SOURCE_SHA256 = {
    "contact_field_v3.py": "96757a5b7062385fc4cd1a7bebc31ef05529ab6f799aeb16a2e0fc906c8be083",
    "rank_candidates_v4_layered.py": "9871434c132ceb3fb0f5e54ff8ada17db241525e7f03c2124b6c6e746558ddfe",
    "rank_candidates_v5_proto.py": "2d6ec85a02f2cf7186900bb535d5769196f14f2c1794d2c346312200fc939f18",
    "rank_candidates_v5_alpha_frozen.py": "b61190f90d3a1942ebab13a5484e1a4f1435822cc992016e658f3274ff657047",
}


@dataclass(frozen=True)
class GeometryConfig:
    """Parameters used by the historical contact_field_v3 extractor."""

    profile: str = "frozen-v1"
    shell_max: float = 10.0
    ideal_dist: float = 7.5
    density_sigma: float = 2.0
    clash_cutoff: float = 3.5
    graph_cutoff: float = 10.0
    contact_threshold: float = 0.1
    tangent_floor: float = 0.45
    front_dot: float = 0.25
    back_dot: float = -0.25
    dot_sigma: float = 0.20
    back_penalty: float = 0.25
    cb_far_delta: float = 0.8
    cb_delta_sigma: float = 0.5
    cb_near_cutoff: float = 10.0
    cb_support_cutoff: float = 12.0
    cb_sigma: float = 1.5
    micro_radius: float = 8.0
    micro_dist_sigma: float = 3.0
    max_microdomains: int = 80
    jump_cutoff: float = 15.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_GEOMETRY_V1 = GeometryConfig()

