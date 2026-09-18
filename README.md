# GridWise LLM — Smart Campus Energy Optimization

**BUP CSE FEST 2026 — Smart Campus Energy Optimization Challenge**

GridWise LLM is a production-ready HTTP microservice that interprets
natural-language operator notes for a campus energy system, compiles them
into mathematical constraints, and solves a deterministic Linear Program
to produce the **minimum-cost 24-hour grid schedule**.

```
   operator notes  ──►  LLM interpreter  ──►  guardrails  ──►  LP solver  ──►  validator  ──►  response
   (1–3 free-form)       (JSON schema)       (Pydantic +     (PuLP CBC)    (independent      (JSON, fully
                                              semantic)                    replay)            validated)
```

The implementation strictly follows the six-phase engineering plan
described in the competition brief, with a hard separation between the LLM
(soft, paraphrastic) and the optimizer (hard, deterministic).

---

## 1. Architecture

| Phase | Module | Responsibility |
| --- | --- | --- |
| 1 | `app/main.py`, `app/api/routes.py`, `app/schemas/` | FastAPI HTTP boundary + Pydantic request/response models |
| 2 | `app/llm/` | Directive interpreter (deterministic + OpenAI-compatible backends) |
| 3 | `app/validation/guardrails.py`, `app/validation/compiler.py` | Semantic guardrails + directive-to-constraint compiler |
| 4 | `app/optimization/` | PuLP-based LP optimizer (energy balance, solar limit, battery dynamics, reserve, rates, end-of-day neutrality) |
| 5 | `app/validation/final_validator.py`, `app/services/optimization_service.py` | Independent validator + end-to-end pipeline |
| 6 | `Dockerfile`, `docker-compose.yml`, `README.md`, `.env.example` | Production hardening, deployment, documentation |

---

## 2. Pipeline (single request)

1. **HTTP parse** — `OptimizeEnergyRequest` validated by Pydantic.
2. **LLM interpret** — One call for all 1–3 operator notes → structured
   `LLMDirectiveList` (deterministic backend by default; OpenAI-compatible
   available via `LLM_PROVIDER=openai`).
3. **Guardrails** — Pydantic structural validation + semantic checks
   (note indices, capacity, hour bounds, factor in `[0,1]`).
4. **Directive compiler** — Deterministic composition of overlapping
   directives into `CompiledScenario`:
   - `solar_reduction`:        `min(factor_a, factor_b)` (most reduction wins)
   - `minimum_battery_reserve`: `max(a, b)` per hour (most restrictive)
   - `max_grid_window`:        `min(cap_a, cap_b)` per hour (most restrictive)
   - `no_charge_window` / `no_discharge_window`: union of hours
5. **LP solver** — PuLP/CBC builds and solves the LP in < 100 ms.
6. **Final validator** — Independently replays every constraint, recalculates
   totals, and verifies end-of-day neutrality.
7. **HTTP response** — Strict `OptimizeEnergyResponse` schema.

---

## 3. Physical energy model

```
grid + solar_used + battery_discharge  =  demand + battery_charge     (hourly)
solar_used  ≤  effective_solar                                        (no export)
energy_after[h]  =  energy_after[h-1] + charge[h] - discharge[h]     (battery state)
energy_after[0]   =  initial + charge[0] - discharge[0]
energy_after[23]  =  initial                                          (HARD)
battery_energy_after[h]  ∈  [effective_min_reserve[h], capacity]      (per hour)
charge[h]  ∈  [0, max_charge_kwh_per_hour]
discharge[h]  ∈  [0, max_discharge_kwh_per_hour]
```

**Percentage semantics** — `factor` is the REMAINING usable solar fraction.
> "Reduce solar availability by 80%" → `factor = 0.2` (only 20% of solar usable).

**Time conversion** — Ranges are START-inclusive / END-exclusive.
> "1 PM to 3 PM" → `[13, 14]`.

---

## 4. Supported directive types

| Type | Example | Effect |
| --- | --- | --- |
| `solar_reduction` | "Reduce solar by 80% from 10 AM to 2 PM." | `effective_solar[h] = solar[h] · factor` |
| `minimum_battery_reserve` | "Keep battery at min 20 kWh from 6 PM to 9 PM." | `energy_after[h] ≥ kWh` |
| `no_charge_window` | "Do not charge between 2 PM and 4 PM." | `charge[h] = 0` |
| `no_discharge_window` | "Do not draw energy from battery between 6 PM and 10 PM." | `discharge[h] = 0` |
| `max_grid_window` | "Cap grid usage at 5 kWh from 6 PM to 9 PM." | `grid[h] ≤ max_grid_kwh` |
| `no_op` | "Cafeteria menu changed today." | directive ignored |

---

## 5. Requirements

* Python ≥ 3.12
* PuLP (CBC solver)
* FastAPI / Uvicorn / Pydantic 2
* pytest (test only)

---

## 6. Environment variables

See `.env.example`. Key values:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | bind address |
| `PORT` | `8000` | bind port |
| `LOG_LEVEL` | `INFO` | structured logging level |
| `LLM_PROVIDER` | `deterministic` | `deterministic` or `openai` |
| `LLM_API_KEY` | _empty_ | required when `LLM_PROVIDER=openai` |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | chat model name |
| `LLM_TIMEOUT` | `20` | LLM HTTP timeout (seconds) |
| `SOLVER_TIME_LIMIT` | `15` | LP solver wall-clock cap |
| `NUMERICAL_TOLERANCE_KWH` | `0.01` | validator tolerance |
| `NUMERICAL_TOLERANCE_BDT` | `0.01` | validator tolerance |

---

## 7. Local setup

```bash
git clone <repo-url>
cd gridwise-llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run the test suite:

```bash
pytest -q
```

---

## 8. Run locally

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

(or `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`).

OpenAPI docs at <http://localhost:8000/docs>.

---

## 9. Docker

### Build

```bash
docker build -t gridwise-llm:latest .
```

### Run

```bash
docker run --rm -p 8000:8000 --name gridwise-llm gridwise-llm:latest
```

### Docker Compose

```bash
docker compose up --build
```

---

## 10. API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /optimize-energy`

**Request body**

```json
{
  "scenario_id": "campus-2026-09-18",
  "operator_notes": [
    "Reduce solar availability by 80% from 10 AM to 2 PM.",
    "Do not charge the battery between 2 PM and 4 PM.",
    "Cafeteria menu has changed today."
  ],
  "hourly_data": [
    {"hour": 0,  "demand_kwh": 8.0,  "solar_kwh": 0.0, "tariff_bdt_per_kwh": 5.0},
    {"hour": 1,  "demand_kwh": 8.0,  "solar_kwh": 0.0, "tariff_bdt_per_kwh": 5.0},
    "... 22 more hours ..."
  ],
  "battery": {
    "capacity_kwh": 50.0,
    "initial_energy_kwh": 25.0,
    "minimum_energy_kwh": 5.0,
    "max_charge_kwh_per_hour": 10.0,
    "max_discharge_kwh_per_hour": 10.0
  }
}
```

**Response body (abridged)**

```json
{
  "scenario_id": "campus-2026-09-18",
  "directive_interpretation": [
    {"note_index": 0, "applies": true,  "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [10,11,12,13], "factor": 0.2}, ...},
    {"note_index": 1, "applies": true,  "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]}, ...},
    {"note_index": 2, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null, ...}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 8.0, "solar_used_kwh": 0.0,
     "battery_charge_kwh": 0.0, "battery_discharge_kwh": 0.0,
     "battery_energy_after_kwh": 25.0, "demand_kwh": 8.0,
     "solar_available_kwh": 0.0, "tariff_bdt_per_kwh": 5.0, "cost_bdt": 40.0},
    "... 23 more hours ..."
  ],
  "total_grid_kwh": 192.0,
  "total_cost_bdt": 960.0,
  "peak_grid_kwh": 10.0,
  "plan_summary": "campus-2026-09-18: grid=192.000 kWh, cost=BDT 960.00, peak=10.000 kWh, battery end=25.000 kWh"
}
```

---

## 11. Testing

```bash
pytest                       # full suite
pytest tests/unit            # unit only
pytest tests/integration     # API only
pytest tests/optimization    # LP only
pytest tests/adversarial     # adversarial only
```

Performance is asserted in `tests/integration/test_e2e.py`.

---

## 12. Performance

* `/health` < 5 ms
* `POST /optimize-energy` (1–3 notes) < 250 ms locally
* LP solver < 100 ms typical
* LLM (deterministic) < 1 ms; (OpenAI-compatible) bounded by API latency

The default backend never makes external calls; the entire pipeline runs
inside the container.

---

## 13. Deployment

1. Push the image: `docker push <registry>/gridwise-llm:<tag>`.
2. Run on any Docker-compatible host (Render, Fly.io, Railway, ECS, K8s).
3. The service binds `0.0.0.0:8000` and exposes `/health` and `/optimize-energy`.

For the competition deployment, run behind a public reverse proxy (Caddy,
NGINX, Cloudflare). The service returns no sensitive data and never logs
operator notes by default.

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `422 invalid_request` | malformed JSON / wrong types / wrong hour count | see error `detail` |
| `500 plan_validation_failed` | LP returned an infeasible schedule; final validator rejected it | reduce grid cap / battery cap / increase battery capacity |
| `LLM error: ...` | the deterministic interpreter cannot classify the note | rephrase with explicit numbers / standard keywords |
| solver slow | infeasibility or near-degeneracy | tighten `SOLVER_TIME_LIMIT`; check input consistency |

---

## 15. Competition Hard Rules (all enforced)

* 24 hours exactly, 0–23, ascending.
* `1 PM–3 PM` → `[13, 14]` (start inclusive, end exclusive).
* Solar reduction factor = REMAINING fraction (`80% reduction` → `0.2`).
* No solar export.
* Battery cannot go below the effective minimum reserve.
* Battery cannot exceed capacity.
* Charge/discharge rates respected per hour.
* End-of-day battery energy equals initial.
* `no_charge_window` ⇒ `charge = 0`; `no_discharge_window` ⇒ `discharge = 0`.
* `max_grid_window` ⇒ `grid ≤ max_grid_kwh`.
* `no_op` ⇒ `applies = false`.
* Each note → exactly one directive; `note_index` preserved.
* Validator independently replays every constraint.
* Totals are recalculated from the validated hourly plan.

---

## 16. License

MIT (or as required by the competition rules).
