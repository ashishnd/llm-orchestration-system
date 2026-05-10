"""Tests for the shared context schema.

Focus: the hardenings — Span validator, SubTask conditional-required field,
PolicyViolation raw_preview truncation, and discriminated union dispatch.

We don't test every field of every model (that would just be testing Pydantic
itself); we test the rules and behaviors we encoded on top of Pydantic.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.context.schema import (
    AgentOutput,
    CompressionOutput,
    CritiqueOutput,
    DecompositionOutput,
    PolicyViolation,
    PreservedRef,
    RetrievalHop,
    RetrievalOutput,
    Span,
    StructuredPipelineOutput,
    SubTask,
    SynthesisOutput,
)

# --------------------------------------------------------------------------- #
# Span validator
# --------------------------------------------------------------------------- #


class TestSpan:
    def test_valid_span(self):
        s = Span(start=0, end=10, text="some text")
        assert s.start == 0 and s.end == 10

    def test_empty_span_is_allowed(self):
        # start == end is empty but valid (half-open intervals)
        Span(start=5, end=5)

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError, match="start must be >= 0"):
            Span(start=-1, end=5)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError, match="end .* must be >= .* start"):
            Span(start=10, end=5)

    def test_text_is_optional(self):
        Span(start=0, end=10)  # no text — fine


# --------------------------------------------------------------------------- #
# SubTask conditional-required field
# --------------------------------------------------------------------------- #


class TestSubTask:
    def test_known_type_does_not_require_detail(self):
        SubTask(description="lookup paper", task_type="retrieval")

    def test_other_type_requires_detail(self):
        with pytest.raises(ValidationError, match="task_type_detail is required"):
            SubTask(description="do the thing", task_type="other")

    def test_other_type_with_detail_is_valid(self):
        SubTask(
            description="do the thing",
            task_type="other",
            task_type_detail="custom: cluster topics across chunks",
        )

    def test_dependency_graph_defaults_to_empty(self):
        t = SubTask(description="x", task_type="retrieval")
        assert t.depends_on == []


# --------------------------------------------------------------------------- #
# PolicyViolation raw_preview truncation
# --------------------------------------------------------------------------- #


class TestPolicyViolation:
    def test_short_preview_passes_through(self):
        pv = PolicyViolation(
            agent_name="retrieval",
            violation_type="schema_violation",
            detail="malformed JSON",
            raw_preview="{'broken'",
        )
        assert pv.raw_preview == "{'broken'"

    def test_long_preview_is_truncated_with_marker(self):
        long_text = "x" * 1000
        pv = PolicyViolation(
            agent_name="retrieval",
            violation_type="schema_violation",
            detail="malformed JSON",
            raw_preview=long_text,
        )
        assert pv.raw_preview is not None
        assert len(pv.raw_preview) <= 600  # 500 + suffix
        assert pv.raw_preview.endswith("...[truncated]")

    def test_no_preview_is_fine(self):
        pv = PolicyViolation(
            agent_name="orchestrator",
            violation_type="budget_overflow",
            detail="too many tokens",
        )
        assert pv.raw_preview is None

    def test_violation_type_enum_enforced(self):
        with pytest.raises(ValidationError):
            PolicyViolation(
                agent_name="x",
                violation_type="not_a_real_violation",  # type: ignore[arg-type]
                detail="x",
            )


# --------------------------------------------------------------------------- #
# Discriminated union dispatch
# --------------------------------------------------------------------------- #


class TestStructuredPipelineOutputDispatch:
    """The discriminator routes a JSON dict to the right typed model based on
    the `type` field. This is the contract every agent's parsed output relies on."""

    def setup_method(self):
        self.adapter = TypeAdapter(StructuredPipelineOutput)

    def test_dispatches_to_decomposition(self):
        raw = {
            "type": "decomposition",
            "sub_tasks": [
                {"description": "lookup", "task_type": "retrieval"},
            ],
        }
        out = self.adapter.validate_python(raw)
        assert isinstance(out, DecompositionOutput)
        assert len(out.sub_tasks) == 1

    def test_dispatches_to_retrieval(self):
        raw = {
            "type": "retrieval",
            "answer_draft": "draft",
            "hops": [
                {"hop_index": 0, "query": "q1", "retrieved_chunk_ids": ["c1"]},
                {"hop_index": 1, "query": "q2", "retrieved_chunk_ids": ["c2"]},
            ],
            "chunk_links": [],
        }
        out = self.adapter.validate_python(raw)
        assert isinstance(out, RetrievalOutput)
        assert out.is_compliant() is True

    def test_dispatches_to_critique(self):
        raw = {
            "type": "critique",
            "target_output_id": "abc",
            "claim_scores": [],
            "disagreements": [],
        }
        out = self.adapter.validate_python(raw)
        assert isinstance(out, CritiqueOutput)

    def test_dispatches_to_synthesis(self):
        raw = {
            "type": "synthesis",
            "final_answer": "the answer",
            "provenance": [],
        }
        out = self.adapter.validate_python(raw)
        assert isinstance(out, SynthesisOutput)

    def test_dispatches_to_compression(self):
        raw = {
            "type": "compression",
            "lossy_summary": "...",
            "preserved_refs": [{"ref_type": "tool_invocation", "ref_id": "t1"}],
            "original_token_count": 1000,
            "compressed_token_count": 200,
        }
        out = self.adapter.validate_python(raw)
        assert isinstance(out, CompressionOutput)
        assert isinstance(out.preserved_refs[0], PreservedRef)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"type": "unknown", "x": 1})

    def test_missing_type_rejected(self):
        with pytest.raises(ValidationError):
            self.adapter.validate_python({"sub_tasks": []})


# --------------------------------------------------------------------------- #
# Retrieval compliance helper
# --------------------------------------------------------------------------- #


class TestRetrievalCompliance:
    def test_two_hops_is_compliant(self):
        out = RetrievalOutput(
            answer_draft="x",
            hops=[
                RetrievalHop(hop_index=0, query="a", retrieved_chunk_ids=["c1"]),
                RetrievalHop(hop_index=1, query="b", retrieved_chunk_ids=["c2"]),
            ],
            chunk_links=[],
        )
        assert out.is_compliant()

    def test_single_hop_is_NOT_compliant_but_still_validates(self):
        # The brief requires >=2 hops; we enforce as a logged violation at
        # orchestrator level, not as a parse error. Schema must accept it.
        out = RetrievalOutput(
            answer_draft="x",
            hops=[
                RetrievalHop(hop_index=0, query="a", retrieved_chunk_ids=["c1"]),
            ],
            chunk_links=[],
        )
        assert not out.is_compliant()

    def test_zero_hops_is_NOT_compliant_but_still_validates(self):
        out = RetrievalOutput(answer_draft="x", hops=[], chunk_links=[])
        assert not out.is_compliant()


# --------------------------------------------------------------------------- #
# AgentOutput envelope
# --------------------------------------------------------------------------- #


class TestAgentOutputEnvelope:
    def test_structured_output_can_be_none(self):
        a = AgentOutput(agent_name="orchestrator", content="routing decision")
        assert a.structured_output is None

    def test_structured_output_dispatches_correctly(self):
        raw = {
            "agent_name": "decomposition",
            "content": "...",
            "structured_output": {
                "type": "decomposition",
                "sub_tasks": [{"description": "x", "task_type": "retrieval"}],
            },
        }
        a = AgentOutput.model_validate(raw)
        assert isinstance(a.structured_output, DecompositionOutput)

    def test_confidence_clamps_to_unit_interval(self):
        with pytest.raises(ValidationError):
            AgentOutput(agent_name="x", content="y", confidence=1.5)
        with pytest.raises(ValidationError):
            AgentOutput(agent_name="x", content="y", confidence=-0.1)
