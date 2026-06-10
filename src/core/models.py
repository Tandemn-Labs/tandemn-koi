from dataclasses import dataclass, field


@dataclass
class Node:
    node_id: str
    node_type: str
    description: str | None = None
    unit: str | None = None


@dataclass
class Edge:
    edge_id: str
    src: str
    dst: str
    src_type: str
    dst_type: str
    status: str = "active"


@dataclass
class EdgeMetadata:
    edge_id: str
    alpha: float = 1.0
    beta: float = 1.0
    visit_count: int = 0
    last_touched_tick: int | None = None
    q_histogram: dict[str, int] = field(
        default_factory=lambda: {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    )
    envs_seen: set[str] = field(default_factory=set)
    q3_frequency: float = 0.0


@dataclass
class Mechanism:
    edge_ids: list[str]
    scope: dict
    narrative: str
    status: str = "active"
    mechanism_id: str | None = None
    archived_reason: str | None = None


@dataclass
class MechanismMetadata:
    mechanism_id: str
    alpha: float = 1.0
    beta: float = 1.0
    visit_count: int = 0
    envs_seen: set[str] = field(default_factory=set)
    last_touched_tick: int | None = None
    q_histogram: dict[str, int] = field(
        default_factory=lambda: {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    )
    inspection_count: int = 0


@dataclass(frozen=True)  # we add frozen to make class immutable and hashable to use in dicts
class EdgeConfidenceRecord:
    edge: Edge
    metadata: EdgeMetadata


@dataclass(frozen=True)
class MechanismConfidenceRecord:
    mechanism: Mechanism
    metadata: MechanismMetadata
