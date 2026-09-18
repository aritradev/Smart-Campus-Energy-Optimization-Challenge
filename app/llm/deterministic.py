"""Deterministic rule-based directive interpreter.

Recognises the six directive types using pattern matching, number parsing,
and time-range extraction. Designed to be paraphrase-robust without relying
on exact phrase matching.

Key edge-case handling:

* ``Solar drops BY 80%``     → factor = 0.2  (reduction magnitude).
* ``Solar drops TO 20%``     → factor = 0.2  (remaining fraction).
* ``noon`` / ``midnight``    → hour 12 / hour 0; participates in ranges.
* Wrap-around ranges (``22:00 to 02:00``) → ascending sorted set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

_AMP_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)\b")
_24H_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_24H_INT_RE = re.compile(r"\b(\d{1,2})\s*:\s*00\b")
_HOUR_INT_RE = re.compile(r"\b([01]?\d|2[0-3])\s*(?:h|hr|hrs|hour|hours)\b", re.IGNORECASE)
_DASH_RANGE = re.compile(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*[-–—]\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", re.IGNORECASE)
_TO_RANGE = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\s*(?:to|through|till|until|and|or|-|–|—)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\b",
    re.IGNORECASE,
)

# Keyword hour tokens. ``noon`` = 12, ``midnight`` = 0.
_KEYWORD_HOURS = {
    "noon": 12,
    "midnight": 0,
}

# Range prepositions that can sit between two numeric tokens.
_TO_RANGE_TOKENS = r"(?:to|through|till|until|and|or|-|–|—)"


def _parse_clock(token: str) -> Optional[int]:
    """Parse a clock-time token like ``10am``, ``14:00``, ``2 PM``, ``noon``,
    ``midnight`` to hour 0-23. Returns ``None`` if the token is unparseable.
    """
    t = token.strip().lower().replace(".", "")
    # Keywords first.
    if t in _KEYWORD_HOURS:
        return _KEYWORD_HOURS[t]
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", t)
    if not m:
        return None
    h = int(m.group(1))
    minutes = int(m.group(2) or "0")
    meridian = m.group(3)
    if meridian == "am":
        if h == 12:
            h = 0
    elif meridian == "pm":
        if h != 12:
            h += 12
    if h < 0 or h > 23:
        return None
    return h


def _expand_range(start: int, end: int) -> List[int]:
    """Expand [start, end) — i.e. start inclusive, end exclusive — into hours."""
    if end == start:
        return [start % 24]
    if end > start:
        return list(range(start, end))
    # Wrap-around (e.g. 22:00 to 2:00)
    return list(range(start, 24)) + list(range(0, end))


def _extract_hours(text: str) -> List[int]:
    """Backward-compatible wrapper around :func:`_extract_hours_with_warnings`.

    Returns only the parsed hours list. Use
    :func:`_extract_hours_with_warnings` directly when you also need to
    surface parse warnings (e.g. for operator feedback).
    """
    hours, _ = _extract_hours_with_warnings(text)
    return hours


def _extract_hours_with_warnings(text: str) -> Tuple[List[int], List[str]]:
    """Extract all hour-clock mentions from the text, returning parse warnings.

    Hours are always returned in ascending order (per the schema contract).
    Wrap-around ranges such as "22:00 to 02:00" produce a set of hours
    ``{22, 23, 0, 1}`` and are sorted ascending, giving ``[0, 1, 22, 23]``.

    Recognised keyword tokens: ``noon`` (12), ``midnight`` (0).

    Warnings are emitted for clock-time tokens that matched a time-specific
    regex but failed to parse because the hour value was out of the valid
    1-12 (12h) or 0-23 (24h) range — e.g. ``33 AM`` in
    ``"Reduce solar by 50% from 33 AM to 1 PM"``. The best-effort fallback
    is to use whatever end of the range *did* parse; the warning surfaces
    the typo so the operator can fix it.

    Returns:
        ``(hours, warnings)`` where ``hours`` is a sorted, deduplicated list
        of integer hours 0-23 and ``warnings`` is a list of human-readable
        warning strings.
    """
    hours: List[int] = []
    warnings: List[str] = []

    # First, try to find a "X to Y" range — these are the most informative.
    for m in _TO_RANGE.finditer(text):
        s_raw, e_raw = m.group(1), m.group(2)
        s = _parse_clock(s_raw)
        e = _parse_clock(e_raw)
        # Surface parse failures as warnings so operators see their typos.
        if s is None and re.search(r"\d", s_raw):
            warnings.append(
                f"unparseable time '{s_raw.strip()}' "
                f"(hour out of 0-23 range)"
            )
        if e is None and re.search(r"\d", e_raw):
            warnings.append(
                f"unparseable time '{e_raw.strip()}' "
                f"(hour out of 0-23 range)"
            )
        # If only one of the pair has AM/PM, propagate it.
        if s is not None and e is not None:
            if "pm" in e_raw.lower() and "am" not in s_raw.lower() and "pm" not in s_raw.lower():
                s12 = s
                if s12 != 12:
                    s = (s12 + 12) % 24
            if "am" in e_raw.lower() and "am" not in s_raw.lower() and "pm" not in s_raw.lower():
                s12 = s
                if s12 == 12:
                    s = 0
            if "am" in s_raw.lower() and "am" not in e_raw.lower() and "pm" not in e_raw.lower():
                e12 = e
                if e12 == 12:
                    e = 0
            if "pm" in s_raw.lower() and "am" not in e_raw.lower() and "pm" not in e_raw.lower():
                e12 = e
                if e12 != 12:
                    e = (e12 + 12) % 24
            for h in _expand_range(s, e):
                if h not in hours:
                    hours.append(h)

    # If we found a complete range, prefer it over scattered mentions.
    if hours:
        return sorted(set(hours)), warnings

    # Otherwise, collect single mentions (AM/PM, 24h, and noon/midnight).
    for kw, hr in _KEYWORD_HOURS.items():
        if kw in text and hr not in hours:
            hours.append(hr)
    for m in _AMP_RE.finditer(text):
        h = _parse_clock(m.group(0))
        if h is not None and h not in hours:
            hours.append(h)
    for m in _24H_INT_RE.finditer(text):
        h = int(m.group(1))
        if 0 <= h <= 23 and h not in hours:
            hours.append(h)
    for m in _HOUR_INT_RE.finditer(text):
        h = int(m.group(1))
        if 0 <= h <= 23 and h not in hours:
            hours.append(h)
    return sorted(set(hours)), warnings


# ---------------------------------------------------------------------------
# Percentage parsing
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_REDUCTION_RE = re.compile(
    r"reduce(?:d|s)?|reduc(?:e|tion)|cut|cut(?:ting)?|limit(?:ed|ing)?|"
    r"lower(?:ed|ing)?|drop(?:ped|ping)?|shrink(?:ed|ing)?",
    re.IGNORECASE,
)
# Prepositions that change semantic interpretation of the percentage value.
# Note: "to" is intentionally constrained to look like a percentage value
# (followed by %, "percent", or appearing in a clearly solar/factor context)
# so it does NOT greedily match "to 2 PM" or other time expressions.
_TO_PREP_RE = re.compile(
    r"\b(?:to|down\s+to|at)\s+(?:a\s+|an\s+|only\s+|just\s+)?"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\b",
    re.IGNORECASE,
)
_BY_PREP_RE = re.compile(
    r"\bby\s+(?:a\s+|an\s+|another\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)?\b",
    re.IGNORECASE,
)
_AVAILABLE_RE = re.compile(
    r"available|availability|production|generate(?:d|s)?|output|"
    r"supply|produces?|produce",
    re.IGNORECASE,
)
_SOLAR_RE = re.compile(r"solar|panel|panels|photovoltaic|PV|rooftop|renewable", re.IGNORECASE)


def _extract_pct(text: str) -> Optional[float]:
    """Return the raw numeric percentage value found in ``text`` (0–100 scale).

    This helper just extracts the number. Callers that care about semantics
    (reduction magnitude vs remaining fraction) should use
    :func:`_extract_pct_semantics` instead.
    """
    m = _PCT_RE.search(text)
    if not m:
        # "by 80" without % symbol
        m = re.search(r"by\s+(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if not m:
            return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_pct_semantics(text: str) -> Tuple[Optional[float], bool]:
    """Extract a percentage with its semantic interpretation.

    Returns ``(value_in_0_1, is_reduction_magnitude)`` where:

    * ``is_reduction_magnitude=True`` means the percentage is the *amount of
      reduction*, so the usable factor is ``1 - value/100``.
      Example: "Reduce solar by 80%" → ``(0.80, True)`` → factor 0.20.
    * ``is_reduction_magnitude=False`` means the percentage is the *remaining
      usable fraction directly*, so the usable factor is ``value/100``.
      Example: "Solar drops to 20%" → ``(0.20, False)`` → factor 0.20.

    Priority order:

    1. Explicit ``BY <X>%`` → reduction magnitude.
    2. Explicit ``TO <X>%`` / ``down to <X>%`` → remaining fraction.
    3. Bare ``<X>%`` → remaining fraction.

    ``BY`` is checked first so "Reduce solar by 80%" is not confused with
    any incidental "to 2 PM" range expression.
    """
    # Try "by <X>%" first — most specific.
    m_by = _BY_PREP_RE.search(text)
    if m_by:
        try:
            v = float(m_by.group(1))
            return max(0.0, min(1.0, v / 100.0)), True
        except ValueError:
            pass

    # Then try "to <X>%"/"down to <X>%".
    m_to = _TO_PREP_RE.search(text)
    if m_to:
        try:
            v = float(m_to.group(1))
            return max(0.0, min(1.0, v / 100.0)), False
        except ValueError:
            pass

    # Fall back to any % sign. Treat as remaining fraction by default.
    m = _PCT_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            return max(0.0, min(1.0, v / 100.0)), False
        except ValueError:
            return None, False
    return None, False


def _is_solar_related(text: str) -> bool:
    return bool(_SOLAR_RE.search(text))


def _is_reduction(text: str) -> bool:
    return bool(_REDUCTION_RE.search(text))


# ---------------------------------------------------------------------------
# Battery reserve parsing
# ---------------------------------------------------------------------------

_RESERVE_RE = re.compile(
    r"(?:keep|maintain|preserve|reserve|hold)(?:\s+(?:at|above|over|of))?\s+"
    r"(?:a\s+|the\s+)?(?:minimum\s+|battery\s+|reserve\s+|level\s+|charge\s+)*"
    r"(?:of\s+)?(\d+(?:\.\d+)?)\s*(kwh|kWh|KWH)?",
    re.IGNORECASE,
)
_MINIMUM_RESERVE_RE = re.compile(
    r"minimum\s+(?:battery\s+|reserve\s+|level\s+)?(?:of\s+|at\s+)?(\d+(?:\.\d+)?)\s*(kwh|kWh)?",
    re.IGNORECASE,
)
# Match "keep at least X% of battery capacity" / "50% of capacity".
_PCT_OF_CAPACITY_RE = re.compile(
    r"(?:at\s+least\s+|at\s+|above\s+|over\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)"
    r"\s+of\s+(?:the\s+|battery\s+|battery\s+)?(?:capacity|total)",
    re.IGNORECASE,
)
_RESERVE_HINT_RE = re.compile(
    r"\b(reserve|minimum|maintain|preserve|hold\s+(?:at|above)|capacity)\b", re.IGNORECASE
)


def _extract_reserve_kwh(text: str) -> Optional[float]:
    """Return the literal kWh reserve value, or None.

    This helper intentionally does NOT resolve ``X% of capacity`` — that
    needs the battery capacity which is not available here. Callers that
    care should also use :func:`_extract_pct_of_capacity` and apply the
    multiplication themselves.
    """
    m = _RESERVE_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m = _MINIMUM_RESERVE_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_pct_of_capacity(text: str) -> Optional[float]:
    """Return the percentage value when the note says ``X% of capacity``.

    Example: ``"Keep at least 50% of battery capacity"`` → 50.0
    """
    m = _PCT_OF_CAPACITY_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Grid cap parsing
# ---------------------------------------------------------------------------

_GRID_CAP_RE = re.compile(
    r"(?:grid\s+(?:cap|cap|usage|draw|consumption|limit|max|maximum)|"
    r"maximum\s+grid|grid\s+shouldn['']t\s+exceed|grid\s+below|grid\s+under|"
    r"grid\s+less\s+than|cap\s+grid)\s*(?:at|of|to|below|under|less\s+than)?\s*"
    r"(\d+(?:\.\d+)?)\s*(kwh|kWh|KWH)?",
    re.IGNORECASE,
)
_DRAW_RE = re.compile(
    r"draw\s+(?:no\s+more\s+than|less\s+than|at\s+most|up\s+to|only)\s+(\d+(?:\.\d+)?)\s*(?:kwh|kWh)?",
    re.IGNORECASE,
)


def _extract_grid_cap(text: str) -> Optional[float]:
    m = _GRID_CAP_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m = _DRAW_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Charge / discharge classification
# ---------------------------------------------------------------------------

_NO_CHARGE_HINT_RE = re.compile(
    r"\b(?:do\s+not|don['']t|never|don\u2019t)\s+(?:charge|be\s+charging)"
    r"|"
    r"\bno\s+(?:charging|charge)"
    r"|"
    r"(?:charging|charge)\s+(?:should|must|will|is|are|to\s+be)\s+(?:be\s+)?(?:disabled|stopped|prohibited|forbidden|restricted|prevented)"
    r"|"
    r"(?:disable|prevent|stop|restrict|forbid|prohibit|skip|hold\s+off)\b[^.]{0,40}?"
    r"(?:charging|charge)\b",
    re.IGNORECASE,
)
_NO_DISCHARGE_HINT_RE = re.compile(
    r"\b(?:do\s+not|don['']t|never|don\u2019t)\s+(?:use|draw|discharge|drain)"
    r"|"
    r"\bno\s+(?:discharging|discharge|draining|drawing(?:\s+energy)?|using(?:\s+the\s+battery)?)"
    r"|"
    r"(?:discharging|discharge|drawing(?:\s+(?:energy\s+)?from\s+(?:the\s+)?battery)?|use\s+of\s+(?:the\s+)?battery)\s+"
    r"(?:should|must|will|is|are|to\s+be)\s+(?:be\s+)?(?:disabled|stopped|prohibited|forbidden|restricted|prevented)"
    r"|"
    r"(?:disable|prevent|stop|restrict|forbid|prohibit|skip|hold\s+off)\b[^.]{0,60}?"
    r"(?:discharging|discharge|drain|"
    r"drawing(?:\s+(?:energy\s+)?from(?:\s+the)?\s*battery)?|use\s+of\s+(?:the\s+)?battery)",
    re.IGNORECASE,
)
_HOLD_CHARGE_RE = re.compile(
    r"\b(?:hold|keep|maintain|leave)(?:\s+the)?\s+battery(?:\s+unchanged|\s+idle|\s+as\s+is)?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Misc keyword maps
# ---------------------------------------------------------------------------

_DISTRACTOR_TOPICS = (
    "cafeteria", "canteen", "menu", "food", "lunch", "breakfast", "dinner",
    "security", "guard", "patrol",
    "class", "lecture", "schedule", "timetable", "exam", "examination",
    "cleaning", "janitor", "cleaner",
    "event", "ceremony", "celebration", "seminar", "workshop", "meeting",
    "maintenance", "plumber", "carpenter", "painting", "repair",
    "weather", "rain", "sunny", "cloudy",
    "parking", "traffic",
    "staff", "teacher", "student", "faculty",
    "holiday", "vacation",
    "birthday", "wedding",
    "library", "book",
    "notice", "announcement", "message",
)


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------


@dataclass
class _Signals:
    """Aggregated signals extracted from a single note."""
    text: str
    text_lower: str
    hours: List[int]
    hours_warnings: List[str]
    pct: Optional[float]
    pct_value: Optional[float]  # 0..1 fraction
    pct_is_reduction_magnitude: bool
    pct_of_capacity: Optional[float]  # raw percentage value, e.g. 50.0
    is_solar: bool
    is_reduction: bool
    reserve_kwh: Optional[float]
    grid_cap: Optional[float]
    no_charge_hint: bool
    no_discharge_hint: bool
    hold_charge_hint: bool
    is_distractor: bool


class DeterministicInterpreter:
    """Rule-based directive interpreter.

    For each operator note, this function returns a *dict* matching the
    ``LLMDirective`` JSON schema. Validation is performed by Pydantic.

    Optional context: pass ``battery_capacity_kwh=`` so that percentage-of-
    capacity phrases (e.g. "Keep at least 50% of battery capacity") can be
    resolved to an absolute kWh value.
    """

    def __init__(self, battery_capacity_kwh: Optional[float] = None) -> None:
        self.battery_capacity_kwh = battery_capacity_kwh

    def interpret(self, notes: List[str]) -> List[Dict[str, Any]]:
        return [self._interpret_one(i, n) for i, n in enumerate(notes)]

    # ----- per-note -----

    def _interpret_one(self, idx: int, note: str) -> Dict[str, Any]:
        s = self._signals(note)
        directive_type, adjustment = self._classify(s)
        applies = directive_type != "no_op"

        # Convert any Pydantic model adjustments back to plain dicts to keep
        # the wire format simple; the validation layer will rebuild the
        # correctly-typed models.
        if hasattr(adjustment, "model_dump"):
            adjustment = adjustment.model_dump()

        out: Dict[str, Any] = {
            "note_index": idx,
            "applies": applies,
            "directive_type": directive_type,
            "structured_adjustment": adjustment,
            "paraphrase": note.strip(),
            "warnings": list(s.hours_warnings),
        }
        return out

    def _signals(self, note: str) -> _Signals:
        text = note.strip()
        text_lower = text.lower()

        hours, hours_warnings = _extract_hours_with_warnings(text_lower)
        pct = _extract_pct(text_lower)
        pct_value, pct_is_reduction_magnitude = _extract_pct_semantics(text_lower)
        pct_of_capacity = _extract_pct_of_capacity(text_lower)
        is_solar = _is_solar_related(text_lower)
        is_reduction = _is_reduction(text_lower)
        reserve_kwh = _extract_reserve_kwh(text_lower)
        grid_cap = _extract_grid_cap(text_lower)
        no_charge_hint = bool(_NO_CHARGE_HINT_RE.search(text_lower))
        no_discharge_hint = bool(_NO_DISCHARGE_HINT_RE.search(text_lower))
        hold_charge_hint = bool(_HOLD_CHARGE_RE.search(text_lower))
        # Distractor: not energy related AND no energy signals
        is_distractor = not any([
            is_solar,
            is_reduction,
            reserve_kwh is not None,
            grid_cap is not None,
            pct_of_capacity is not None,
            no_charge_hint,
            no_discharge_hint,
            hold_charge_hint,
        ])
        # Strengthen distractor detection with keyword list
        if is_distractor and any(w in text_lower for w in _DISTRACTOR_TOPICS):
            is_distractor = True
        # Reject distractor if it has hours AND any keyword like "charge"/"discharge"/"solar"/"battery"
        if any(k in text_lower for k in (
            "battery", "charge", "discharge", "solar", "grid", "kwh", "kw",
            "reserve", "tariff", "energy", "demand", "capacity",
        )):
            is_distractor = False

        return _Signals(
            text=text,
            text_lower=text_lower,
            hours=hours,
            hours_warnings=hours_warnings,
            pct=pct,
            pct_value=pct_value,
            pct_is_reduction_magnitude=pct_is_reduction_magnitude,
            pct_of_capacity=pct_of_capacity,
            is_solar=is_solar,
            is_reduction=is_reduction,
            reserve_kwh=reserve_kwh,
            grid_cap=grid_cap,
            no_charge_hint=no_charge_hint,
            no_discharge_hint=no_discharge_hint,
            hold_charge_hint=hold_charge_hint,
            is_distractor=is_distractor,
        )

    def _classify(self, s: _Signals) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Distractor: only no_op
        if s.is_distractor:
            return "no_op", None

        # Most-specific first.
        # 1) minimum_battery_reserve
        if s.reserve_kwh is not None:
            hours = s.hours if s.hours else list(range(24))
            return "minimum_battery_reserve", {
                "hours": hours,
                "minimum_energy_kwh": s.reserve_kwh,
            }

        # 1b) "X% of battery capacity" — needs capacity context.
        if s.pct_of_capacity is not None and self.battery_capacity_kwh:
            hours = s.hours if s.hours else list(range(24))
            min_kwh = max(0.0, (s.pct_of_capacity / 100.0) * self.battery_capacity_kwh)
            return "minimum_battery_reserve", {
                "hours": hours,
                "minimum_energy_kwh": min_kwh,
            }

        # 2) max_grid_window
        if s.grid_cap is not None:
            hours = s.hours if s.hours else list(range(24))
            # If hours absent, this is technically global — but the schema
            # requires min 1 hour. Use full day.
            return "max_grid_window", {
                "hours": hours,
                "max_grid_kwh": s.grid_cap,
            }

        # 3) no_discharge_window (specific phrase before generic charge phrase)
        if s.no_discharge_hint:
            hours = s.hours if s.hours else list(range(24))
            return "no_discharge_window", {"hours": hours}

        # 4) no_charge_window
        if s.no_charge_hint or s.hold_charge_hint:
            hours = s.hours if s.hours else list(range(24))
            return "no_charge_window", {"hours": hours}

        # 5) solar_reduction (must have percentage AND solar/reduction context)
        if s.is_solar and s.is_reduction and s.pct_value is not None:
            if s.pct_is_reduction_magnitude:
                # "reduce BY 80%" / "drops BY 80%" → factor = 1 - value
                factor = max(0.0, min(1.0, 1.0 - s.pct_value))
            else:
                # "drops TO 20%" / "only 20% of normal" → factor = value directly
                factor = s.pct_value
            hours = s.hours if s.hours else list(range(24))
            return "solar_reduction", {
                "hours": hours,
                "factor": round(factor, 6),
            }

        # 6) Fallback: try percentage alone with solar context (no reduction verb).
        if s.is_solar and s.pct_value is not None and not s.is_reduction:
            # "Solar available will be only 20% of normal" → factor 0.2
            # "Solar output will be 80%" → factor 0.8
            factor = s.pct_value
            hours = s.hours if s.hours else list(range(24))
            return "solar_reduction", {
                "hours": hours,
                "factor": round(factor, 6),
            }

        # 7) Last-resort energy-text without any rule → no_op
        return "no_op", None
