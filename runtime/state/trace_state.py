from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpanRecord:
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class TraceState:
    trace_id: str = ""
    current_span_id: str = ""
    spans: list[SpanRecord] = field(default_factory=list)
