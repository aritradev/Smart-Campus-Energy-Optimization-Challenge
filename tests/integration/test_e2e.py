"""End-to-end pipeline tests.

Send complete POST /optimize-energy requests and verify the response.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests import make_scenario


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _post(client, scenario):
    payload = {
        "scenario_id": scenario.scenario_id,
        "operator_notes": scenario.operator_notes,
        "hourly_data": [h.model_dump() for h in scenario.hourly_data],
        "battery": scenario.battery.model_dump(),
    }
    return client.post("/optimize-energy", json=payload)


# ---------------------------------------------------------------------------
# Happy-path scenarios
# ---------------------------------------------------------------------------


def test_e2e_no_notes_one_distractor(client):
    """A distractor note should produce a valid schedule with applies=false."""
    scenario = make_scenario(
        scenario_id="e2e-1",
        notes=["The cafeteria menu has changed today."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scenario_id"] == "e2e-1"
    assert len(body["directive_interpretation"]) == 1
    d = body["directive_interpretation"][0]
    assert d["applies"] is False
    assert d["directive_type"] == "no_op"
    assert d["structured_adjustment"] is None
    assert len(body["hourly_plan"]) == 24
    assert body["total_grid_kwh"] >= 0
    assert body["total_cost_bdt"] >= 0
    assert body["peak_grid_kwh"] >= 0


def test_e2e_solar_reduction(client):
    scenario = make_scenario(
        scenario_id="e2e-2",
        notes=["Reduce solar availability by 80% from 10 AM to 2 PM."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200, r.text
    body = r.json()
    d = body["directive_interpretation"][0]
    assert d["applies"] is True
    assert d["directive_type"] == "solar_reduction"
    assert d["structured_adjustment"]["factor"] == pytest.approx(0.2)
    assert d["structured_adjustment"]["hours"] == [10, 11, 12, 13]


def test_e2e_no_charge_window(client):
    scenario = make_scenario(
        scenario_id="e2e-3",
        notes=["Do not charge the battery between 2 PM and 4 PM."],
    )
    r = _post(client, scenario)
    assert r.status_code == 200, r.text
    body = r.json()
    d = body["directive_interpretation"][0]
    assert d["directive_type"] == "no_charge_window"
    assert d["structured_adjustment"]["hours"] == [14, 15]
    # Verify charge is zero in those hours
    for h in [14, 15]:
        entry = next(e for e in body["hourly_plan"] if e["hour"] == h)
        assert entry["battery_charge_kwh"] == pytest.approx(0.0, abs=0.01)


def test_e2e_three_notes_mixed(client):
    scenario = make_scenario(
        scenario_id="e2e-4",
        notes=[
            "Reduce solar availability by 80% from 10 AM to 2 PM.",
            "Do not charge the battery between 14:00 and 16:00.",
            "Cafeteria menu has changed.",
        ],
    )
    r = _post(client, scenario)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["directive_interpretation"]) == 3
    # Order preserved
    assert body["directive_interpretation"][0]["note_index"] == 0
    assert body["directive_interpretation"][1]["note_index"] == 1
    assert body["directive_interpretation"][2]["note_index"] == 2
    # Verify end-of-day neutrality (last hour)
    e_last = body["hourly_plan"][-1]
    assert e_last["battery_energy_after_kwh"] == pytest.approx(
        scenario.battery.initial_energy_kwh, abs=0.01
    )


def test_e2e_response_field_names_match_competition(client):
    scenario = make_scenario()
    r = _post(client, scenario)
    body = r.json()
    # Required fields per the spec
    assert "scenario_id" in body
    assert "directive_interpretation" in body
    assert "hourly_plan" in body
    assert "total_grid_kwh" in body
    assert "total_cost_bdt" in body
    assert "peak_grid_kwh" in body
    assert "plan_summary" in body


def test_e2e_hourly_plan_entry_field_names(client):
    scenario = make_scenario()
    r = _post(client, scenario)
    entry = r.json()["hourly_plan"][0]
    for k in (
        "hour", "grid_kwh", "solar_used_kwh", "battery_charge_kwh",
        "battery_discharge_kwh", "battery_energy_after_kwh",
        "demand_kwh", "solar_available_kwh", "tariff_bdt_per_kwh", "cost_bdt",
    ):
        assert k in entry, f"missing {k}"


def test_e2e_per_hour_energy_balance(client):
    scenario = make_scenario(
        scenario_id="e2e-balance",
        notes=[
            "Reduce solar availability by 50% from 10 AM to 12 PM.",
            "Cap grid usage at 6 kWh from 6 PM to 9 PM.",
        ],
    )
    r = _post(client, scenario)
    body = r.json()
    for e in body["hourly_plan"]:
        # grid + solar_used + discharge == demand + charge
        lhs = e["grid_kwh"] + e["solar_used_kwh"] + e["battery_discharge_kwh"]
        rhs = e["demand_kwh"] + e["battery_charge_kwh"]
        assert lhs == pytest.approx(rhs, abs=0.01)
    # Grid cap
    for e in body["hourly_plan"]:
        if 18 <= e["hour"] <= 20:
            assert e["grid_kwh"] <= 6.0 + 0.01


def test_e2e_battery_neutrality(client):
    scenario = make_scenario()
    r = _post(client, scenario)
    body = r.json()
    initial = scenario.battery.initial_energy_kwh
    assert body["hourly_plan"][-1]["battery_energy_after_kwh"] == pytest.approx(initial, abs=0.01)


def test_e2e_total_cost_matches_sum(client):
    scenario = make_scenario()
    r = _post(client, scenario)
    body = r.json()
    summed_cost = sum(e["cost_bdt"] for e in body["hourly_plan"])
    assert body["total_cost_bdt"] == pytest.approx(summed_cost, abs=0.01)
    summed_grid = sum(e["grid_kwh"] for e in body["hourly_plan"])
    assert body["total_grid_kwh"] == pytest.approx(summed_grid, abs=0.01)
    peak = max(e["grid_kwh"] for e in body["hourly_plan"])
    assert body["peak_grid_kwh"] == pytest.approx(peak, abs=0.01)


def test_e2e_performance_single_note(client):
    scenario = make_scenario(
        scenario_id="perf-1",
        notes=["Do not charge between 14:00 and 15:00."],
    )
    t0 = time.perf_counter()
    r = _post(client, scenario)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 30.0  # generous target


def test_e2e_performance_three_notes(client):
    scenario = make_scenario(
        scenario_id="perf-3",
        notes=[
            "Reduce solar availability by 80% from 10 AM to 2 PM.",
            "Do not charge the battery between 14:00 and 16:00.",
            "Cafeteria menu has changed.",
        ],
    )
    t0 = time.perf_counter()
    r = _post(client, scenario)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 30.0


def test_e2e_health_fast(client):
    t0 = time.perf_counter()
    r = client.get("/health")
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 1.0
