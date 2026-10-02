from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable

from railforge.locomotive.tractive_effort import Locomotive, available_effort, required_effort
from railforge.maintenance.release import Evidence, ReleasePolicy, releasable

PRESERVE_FEASIBLE = "preserve_feasible"
ROLLBACK = "rollback"
STRATEGIES = (PRESERVE_FEASIBLE, ROLLBACK)


class RepairConflict(RuntimeError):
    """A repair insertion invalidated committed legs while the strategy was a
    full rollback, so nothing was committed."""

    def __init__(self, order_id: str, impacted: Iterable[str]):
        self.order_id = order_id
        self.impacted = tuple(impacted)
        super().__init__(f"repair {order_id} invalidates committed legs: {','.join(self.impacted)}")


@dataclass(frozen=True)
class CycleUnit:
    unit_id: str
    location: str
    available_at: datetime
    continuous_kn: float
    adhesion_kn: float
    due_hours: float


@dataclass(frozen=True)
class CycleLeg:
    leg_id: str
    train_id: str
    origin: str
    destination: str
    depart: datetime
    arrive: datetime
    tonnes: float
    gradient_permille: float
    curve_resistance_kn: float
    required_units: int = 1


@dataclass(frozen=True)
class UnitWindow:
    """Reservation over a unit: busy during [start, end) and left at `location`."""

    unit_id: str
    start: datetime
    end: datetime
    location: str


@dataclass(frozen=True)
class RepairOrder:
    """Unplanned repair insertion for one unit, plus the evidence that will
    certify its release back into the cycle."""

    order_id: str
    unit_id: str
    location: str
    start: datetime
    end: datetime
    required_evidence: frozenset[str]
    release_evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class CycleState:
    """Every fact validated inside one version boundary."""

    legs: tuple[CycleLeg, ...]
    units: tuple[CycleUnit, ...]
    turn_minutes: int
    adhesion_factor: float
    policy: ReleasePolicy
    evidence: dict[str, tuple[Evidence, ...]]
    reservations: tuple[UnitWindow, ...]
    repairs: tuple[RepairOrder, ...]
    assignments: dict[str, tuple[str, ...]]
    released: tuple[str, ...]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return _utc(a_start) < _utc(b_end) and _utc(b_start) < _utc(a_end)


def repair_window(order: RepairOrder) -> UnitWindow:
    return UnitWindow(order.unit_id, order.start, order.end, order.location)


def _unit_effort(unit: CycleUnit, adhesion_factor: float) -> float:
    return min(unit.continuous_kn, unit.adhesion_kn * adhesion_factor)


def _traction_ok(units, leg: CycleLeg, adhesion_factor: float) -> bool:
    rows = [Locomotive(u.unit_id, u.continuous_kn, u.adhesion_kn, True) for u in units]
    return available_effort(rows, adhesion_factor) >= required_effort(
        leg.tonnes, leg.gradient_permille, leg.curve_resistance_kn)


def _choose(candidates, leg: CycleLeg, adhesion_factor: float, by_id):
    if len(candidates) < leg.required_units:
        return None
    chosen = list(candidates[:leg.required_units])
    if _traction_ok([by_id[u] for u in chosen], leg, adhesion_factor):
        return tuple(sorted(chosen))
    for cand in candidates[leg.required_units:]:
        weakest = min(chosen, key=lambda u: (_unit_effort(by_id[u], adhesion_factor), u))
        if _unit_effort(by_id[cand], adhesion_factor) > _unit_effort(by_id[weakest], adhesion_factor):
            chosen[chosen.index(weakest)] = cand
        if _traction_ok([by_id[u] for u in chosen], leg, adhesion_factor):
            return tuple(sorted(chosen))
    return None


def plan_cycle(legs, units, turn_minutes, adhesion_factor, evidence, policy,
               reservations=(), fixed=None, skip=frozenset()):
    """Assign units to legs in departure order.

    One deterministic pass validates traction capacity, unit availability,
    maintenance evidence, round-trip turnover and every reservation window
    (existing train plans and repairs) against the same inputs.  `fixed`
    pins previously committed assignments, which are kept while still
    feasible.  Returns (assignments, released leg ids).
    """
    legs = tuple(legs)
    units = tuple(units)
    reservations = tuple(reservations)
    if turn_minutes < 0:
        raise ValueError("turn minutes")
    if len({l.leg_id for l in legs}) != len(legs):
        raise ValueError("duplicate leg")
    by_id = {u.unit_id: u for u in units}
    if len(by_id) != len(units):
        raise ValueError("duplicate unit")
    turn = timedelta(minutes=turn_minutes)
    evidence = {k: tuple(v) for k, v in dict(evidence or {}).items()}
    fixed = dict(fixed or {})
    state = {u.unit_id: [u.location, _utc(u.available_at), u.due_hours] for u in units}
    res_by_unit: dict[str, list[UnitWindow]] = {}
    for w in reservations:
        if w.unit_id in state:
            res_by_unit.setdefault(w.unit_id, []).append(w)
    for ws in res_by_unit.values():
        ws.sort(key=lambda w: (_utc(w.start), _utc(w.end)))
        for earlier, later in zip(ws, ws[1:]):
            if _utc(later.start) < _utc(earlier.end):
                raise ValueError("overlapping reservations")
    consumed = {uid: 0 for uid in res_by_unit}
    assignments: dict[str, tuple[str, ...]] = {}
    released: list[str] = []
    for leg in sorted(legs, key=lambda l: (_utc(l.depart), l.leg_id)):
        if leg.leg_id in skip:
            continue
        depart = _utc(leg.depart)
        arrive = _utc(leg.arrive)
        if arrive <= depart:
            raise ValueError("leg")
        busy_until = arrive + turn
        hours = (arrive - depart).total_seconds() / 3600.0
        for uid, ws in res_by_unit.items():
            index = consumed[uid]
            while index < len(ws) and _utc(ws[index].end) <= depart:
                window = ws[index]
                state[uid] = [window.location, max(state[uid][1], _utc(window.end)), state[uid][2]]
                index += 1
            consumed[uid] = index

        def eligible(uid):
            location, available, due = state[uid]
            if location != leg.origin or available > depart or due <= hours:
                return False
            if any(_overlaps(depart, busy_until, w.start, w.end) for w in res_by_unit.get(uid, ())):
                return False
            return releasable(list(evidence.get(uid, ())), policy, depart)

        chosen = None
        want = fixed.get(leg.leg_id)
        if (want is not None and len(want) == leg.required_units
                and all(u in by_id and eligible(u) for u in want)
                and _traction_ok([by_id[u] for u in want], leg, adhesion_factor)):
            chosen = tuple(sorted(want))
        else:
            candidates = [u for _, u in sorted((state[u][1], u) for u in state if eligible(u))]
            chosen = _choose(candidates, leg, adhesion_factor, by_id)
        if chosen is None:
            released.append(leg.leg_id)
            continue
        for uid in chosen:
            state[uid] = [leg.destination, busy_until, state[uid][2] - hours]
        assignments[leg.leg_id] = chosen
    return assignments, tuple(released)


def commit_state(legs, units, turn_minutes, adhesion_factor, evidence, policy,
                 reservations=()) -> CycleState:
    """Build the initial cycle state; every leg must be covered or the whole
    commit fails closed."""
    evidence_map = {k: tuple(v) for k, v in dict(evidence or {}).items()}
    assignments, released = plan_cycle(legs, units, turn_minutes, adhesion_factor,
                                       evidence_map, policy, reservations)
    if released:
        raise ValueError("infeasible legs: " + ",".join(released))
    return CycleState(tuple(legs), tuple(units), turn_minutes, adhesion_factor, policy,
                      evidence_map, tuple(reservations), (), dict(assignments), ())


def insert_repair(state: CycleState, order: RepairOrder, strategy: str,
                  external_reservations=()):
    """Insert an unplanned repair into a committed cycle.

    Committed legs are never invalidated silently: the impacted set is
    computed explicitly and the caller picks the outcome via `strategy`:

    - PRESERVE_FEASIBLE commits the repair, reassigning impacted legs where a
      cover exists and explicitly releasing the rest;
    - ROLLBACK raises RepairConflict and nothing changes.

    The repair window itself may not overlap the existing train plan, other
    committed cycles or other repairs; such orders are rejected outright.
    Re-inserting an already known order id is a no-op.
    """
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy")
    if order.order_id in {r.order_id for r in state.repairs}:
        return state, {"duplicate": True, "impacted": [], "reassigned": {}, "released": []}
    start = _utc(order.start)
    end = _utc(order.end)
    if end <= start:
        raise ValueError("repair window")
    if order.unit_id not in {u.unit_id for u in state.units}:
        raise ValueError("unknown unit")
    if not releasable(list(order.release_evidence), ReleasePolicy(order.required_evidence), end):
        raise ValueError("repair release evidence incomplete")
    blocked = (tuple(state.reservations) + tuple(external_reservations)
               + tuple(repair_window(r) for r in state.repairs))
    for w in blocked:
        if w.unit_id == order.unit_id and _overlaps(start, end, w.start, w.end):
            raise ValueError("repair window conflicts with existing plan")
    reservations = blocked + (repair_window(order),)
    assignments, released = plan_cycle(state.legs, state.units, state.turn_minutes,
                                       state.adhesion_factor, state.evidence, state.policy,
                                       reservations, fixed=state.assignments,
                                       skip=frozenset(state.released))
    impacted = sorted(lid for lid in state.assignments
                      if assignments.get(lid) != state.assignments[lid])
    if impacted and strategy == ROLLBACK:
        raise RepairConflict(order.order_id, impacted)
    reassigned = {lid: assignments[lid] for lid in impacted if lid in assignments}
    dropped = tuple(lid for lid in impacted if lid not in assignments)
    new_state = replace(state, repairs=state.repairs + (order,), assignments=assignments,
                        released=tuple(sorted(set(state.released) | set(dropped))))
    return new_state, {"duplicate": False, "impacted": impacted,
                       "reassigned": reassigned, "released": list(dropped)}


def windows_from_state(state: CycleState) -> tuple[UnitWindow, ...]:
    """Reservations this committed cycle imposes on any other plan."""
    legs = {l.leg_id: l for l in state.legs}
    turn = timedelta(minutes=state.turn_minutes)
    windows = []
    for leg_id, uids in state.assignments.items():
        leg = legs[leg_id]
        for uid in uids:
            windows.append(UnitWindow(uid, leg.depart, _utc(leg.arrive) + turn, leg.destination))
    windows.extend(repair_window(r) for r in state.repairs)
    return tuple(windows)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _parse(value: str) -> datetime:
    return _utc(datetime.fromisoformat(value))


def evidence_to_dict(e: Evidence) -> dict:
    return {"kind": e.kind, "completed_at": _iso(e.completed_at),
            "valid_for": e.valid_for.total_seconds(), "passed": e.passed}


def evidence_from_dict(d) -> Evidence:
    return Evidence(d["kind"], _parse(d["completed_at"]),
                    timedelta(seconds=float(d["valid_for"])), bool(d["passed"]))


def leg_to_dict(leg: CycleLeg) -> dict:
    return {"leg_id": leg.leg_id, "train_id": leg.train_id, "origin": leg.origin,
            "destination": leg.destination, "depart": _iso(leg.depart), "arrive": _iso(leg.arrive),
            "tonnes": leg.tonnes, "gradient_permille": leg.gradient_permille,
            "curve_resistance_kn": leg.curve_resistance_kn, "required_units": leg.required_units}


def leg_from_dict(d) -> CycleLeg:
    return CycleLeg(d["leg_id"], d["train_id"], d["origin"], d["destination"],
                    _parse(d["depart"]), _parse(d["arrive"]), float(d["tonnes"]),
                    float(d["gradient_permille"]), float(d["curve_resistance_kn"]),
                    int(d["required_units"]))


def unit_to_dict(unit: CycleUnit) -> dict:
    return {"unit_id": unit.unit_id, "location": unit.location,
            "available_at": _iso(unit.available_at), "continuous_kn": unit.continuous_kn,
            "adhesion_kn": unit.adhesion_kn, "due_hours": unit.due_hours}


def unit_from_dict(d) -> CycleUnit:
    return CycleUnit(d["unit_id"], d["location"], _parse(d["available_at"]),
                     float(d["continuous_kn"]), float(d["adhesion_kn"]), float(d["due_hours"]))


def window_to_dict(w: UnitWindow) -> dict:
    return {"unit_id": w.unit_id, "start": _iso(w.start), "end": _iso(w.end), "location": w.location}


def window_from_dict(d) -> UnitWindow:
    return UnitWindow(d["unit_id"], _parse(d["start"]), _parse(d["end"]), d["location"])


def order_to_dict(order: RepairOrder) -> dict:
    return {"order_id": order.order_id, "unit_id": order.unit_id, "location": order.location,
            "start": _iso(order.start), "end": _iso(order.end),
            "required_evidence": sorted(order.required_evidence),
            "release_evidence": [evidence_to_dict(e) for e in order.release_evidence]}


def order_from_dict(d) -> RepairOrder:
    return RepairOrder(d["order_id"], d["unit_id"], d["location"], _parse(d["start"]),
                       _parse(d["end"]), frozenset(d["required_evidence"]),
                       tuple(evidence_from_dict(e) for e in d["release_evidence"]))


def state_to_dict(state: CycleState) -> dict:
    return {
        "legs": [leg_to_dict(l) for l in state.legs],
        "units": [unit_to_dict(u) for u in state.units],
        "turn_minutes": state.turn_minutes,
        "adhesion_factor": state.adhesion_factor,
        "policy": sorted(state.policy.required),
        "evidence": {uid: [evidence_to_dict(e) for e in evs]
                     for uid, evs in sorted(state.evidence.items())},
        "reservations": [window_to_dict(w) for w in state.reservations],
        "repairs": [order_to_dict(r) for r in state.repairs],
        "assignments": {k: list(v) for k, v in sorted(state.assignments.items())},
        "released": list(state.released),
    }


def state_from_dict(d) -> CycleState:
    return CycleState(
        legs=tuple(leg_from_dict(x) for x in d["legs"]),
        units=tuple(unit_from_dict(x) for x in d["units"]),
        turn_minutes=int(d["turn_minutes"]),
        adhesion_factor=float(d["adhesion_factor"]),
        policy=ReleasePolicy(frozenset(d["policy"])),
        evidence={k: tuple(evidence_from_dict(e) for e in v) for k, v in d["evidence"].items()},
        reservations=tuple(window_from_dict(x) for x in d["reservations"]),
        repairs=tuple(order_from_dict(x) for x in d["repairs"]),
        assignments={k: tuple(v) for k, v in d["assignments"].items()},
        released=tuple(d["released"]),
    )
