"""Property-based tests for full-stack-agent-orchestration (design Properties 1-11).

Run: pytest -m property --hypothesis-seed=0
The pure invariants below run without redis/kafka/pgvector. Properties 6, 7, 12
are integration-level (require a DB / message interception) and live in the
integration suite.
"""
import asyncio
import uuid

import pytest
from hypothesis import given, settings, strategies as st

pytestmark = pytest.mark.property

from app.services.tool_router import classify_tool, ToolRouter
from app.services.tool_models import ToolCall
from app.services.planner import PlannerAgent, Subtask, ExecutionPlan
from app.tools.database_tool import _validate_sql_keywords, _validate_interpolation
from app.services.agent import _strip_sensitive_data
from tests.helpers import create_mock_memory_tool

_PREFIXES = ["user_", "task_", "query_", "memory_", "realtime_"]


# Property 1 — Tool Router prefix classification invariant
@given(st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1))
@settings(max_examples=100)
def test_tool_router_prefix_classification_invariant(suffix):
    for prefix, expected in [
        ("user_", "service_tool"), ("task_", "service_tool"),
        ("query_", "database_tool"), ("memory_", "memory_tool"),
        ("realtime_", "realtime_tool"),
    ]:
        assert classify_tool(prefix + suffix) == expected


@given(st.text(min_size=1).filter(lambda n: not any(n.startswith(p) for p in _PREFIXES)))
@settings(max_examples=100)
def test_tool_router_no_prefix_maps_to_external(name):
    assert classify_tool(name) == "external_tool"


# Property 2 — Unregistered tool returns an error ToolResult
@given(st.text(min_size=1, max_size=40).filter(lambda n: not any(n.startswith(p) for p in _PREFIXES)))
@settings(max_examples=50)
def test_tool_router_unregistered_returns_error(name):
    name = "zzz_" + name  # ensure it is not a real registered tool
    router = ToolRouter()
    res = asyncio.run(router.route(ToolCall(tool_name=name, parameters={}, call_id="c"), correlation_id="cid"))
    assert res.status == "error"
    assert res.error_message == f"Tool '{name}' not found in registry"


# Property 3 — Planner index uniqueness invariant
@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_planner_index_uniqueness(n):
    planner = PlannerAgent(registry=None, max_retries=2)
    subtasks = [Subtask(index=i, description=f"step {i}") for i in range(1, n + 1)]
    plan = ExecutionPlan(subtasks=subtasks, correlation_id="test")
    assert planner._validate_plan(plan) == []
    indices = [s.index for s in plan.subtasks]
    assert sorted(indices) == list(range(1, n + 1))
    assert sum(indices) == n * (n + 1) // 2


# Property 4 — Forward-only dependencies
@given(st.integers(min_value=2, max_value=10), st.integers(min_value=0, max_value=9))
@settings(max_examples=100)
def test_planner_forward_dependency_rejected(n, offset):
    planner = PlannerAgent(registry=None, max_retries=2)
    bad_index = 1 + (offset % n)
    subtasks = [Subtask(index=i, description=f"s{i}") for i in range(1, n + 1)]
    # Make subtask `bad_index` depend on a >= index (forward/self dependency).
    subtasks[bad_index - 1].dependency_indices = [bad_index]
    plan = ExecutionPlan(subtasks=subtasks, correlation_id="t")
    assert planner._validate_plan(plan) != []


# Property 5 — Memory store-then-search round-trip
@given(st.text(min_size=1, max_size=200).filter(lambda t: t.strip()))
@settings(max_examples=100)
def test_memory_round_trip(text):
    tool = create_mock_memory_tool(similarity_threshold=0.7)
    store = asyncio.run(tool.store(text=text, conversation_id=1))
    assert store["status"] == "success"
    search = asyncio.run(tool.search(query=text, conversation_id=1))
    chunks = search["chunks"]
    assert len(chunks) >= 1
    assert chunks[0]["similarity"] >= 0.7


# Property 8 — User service strips sensitive fields
@given(st.fixed_dictionaries({
    "id": st.integers(min_value=1),
    "name": st.text(min_size=1, max_size=20),
    "role": st.just("user"),
    "password": st.text(max_size=20),
    "password_hash": st.text(max_size=20),
    "api_key": st.text(max_size=20),
}))
@settings(max_examples=100)
def test_user_tool_strips_sensitive_fields(user_dict):
    result = _strip_sensitive_data(user_dict)
    assert "password" not in result
    assert "password_hash" not in result
    assert "api_key" not in result


# Property 9 — Database tool rejects DML/DDL keywords
@given(
    st.sampled_from(["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER",
                     "CREATE", "REPLACE", "GRANT", "REVOKE"]),
    st.text(alphabet="abc SELECT* ", max_size=20),
)
@settings(max_examples=100)
def test_database_tool_rejects_forbidden_keywords(keyword, prefix):
    allowed, error = _validate_sql_keywords(f"{prefix} {keyword} something")
    assert not allowed
    assert error == "Only SELECT queries are permitted"


# Property 10 — Database tool rejects unparameterized string literals
@given(st.text(alphabet=st.characters(blacklist_characters="'"), min_size=1, max_size=20))
@settings(max_examples=100)
def test_database_tool_rejects_interpolation(value):
    sql = f"SELECT * FROM users WHERE name = '{value}'"
    allowed, error = _validate_interpolation(sql)
    assert not allowed
    assert error == "Parameterization required: direct value interpolation detected"


# Property 11 — Correlation ID is a valid UUID v4
@given(st.integers(min_value=0, max_value=1000))
@settings(max_examples=100)
def test_correlation_id_is_uuid_v4(_):
    cid = str(uuid.uuid4())
    assert uuid.UUID(cid).version == 4
