"""Integration tests for the FastAPI app."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests import default_battery, make_24_hours, make_scenario


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_optimize_energy_placeholder(client: TestClient):
    scenario = make_scenario()
    payload = {
        "scenario_id": scenario.scenario_id,
        "operator_notes": scenario.operator_notes,
        "hourly_data": [h.model_dump() for h in scenario.hourly_data],
        "battery": scenario.battery.model_dump(),
    }
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scenario_id"] == "test-001"
    assert len(body["hourly_plan"]) == 24
    assert body["total_grid_kwh"] >= 0
    assert body["total_cost_bdt"] >= 0


def test_optimize_energy_rejects_malformed(client: TestClient):
    # Missing hourly_data
    bad = {
        "scenario_id": "x",
        "operator_notes": ["a"],
        "battery": default_battery().model_dump(),
    }
    resp = client.post("/optimize-energy", json=bad)
    assert resp.status_code == 422


def test_optimize_energy_rejects_extra_field(client: TestClient):
    scenario = make_scenario()
    payload = {
        "scenario_id": scenario.scenario_id,
        "operator_notes": scenario.operator_notes,
        "hourly_data": [h.model_dump() for h in scenario.hourly_data],
        "battery": scenario.battery.model_dump(),
        "rogue": "field",
    }
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 422


def test_optimize_energy_rejects_invalid_hour_count(client: TestClient):
    scenario = make_scenario()
    payload = {
        "scenario_id": scenario.scenario_id,
        "operator_notes": scenario.operator_notes,
        "hourly_data": [h.model_dump() for h in make_24_hours()[:12]],
        "battery": scenario.battery.model_dump(),
    }
    resp = client.post("/optimize-energy", json=payload)
    assert resp.status_code == 422


def test_optimize_energy_handles_blank_body(client: TestClient):
    resp = client.post("/optimize-energy", json={})
    assert resp.status_code == 422


def test_optimize_energy_does_not_crash_on_garbage(client: TestClient):
    resp = client.post(
        "/optimize-energy",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code in (400, 422)
