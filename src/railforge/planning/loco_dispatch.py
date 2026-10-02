"""Locomotive cycle dispatch with unscheduled (running-repair) insertion.

Five planning factors are evaluated *together* by :func:`solve`, inside one
version boundary on the workflow layer:

* tractive capacity (continuous/adhesion effort against train resistance),
* locomotive availability (initial ready time plus post-move position),
* maintenance evidence (inspection/release evidence valid at departure),
* round-trip turnaround (minimum turn time between legs),
* existing train plans (fixed/locked consists that must be retained).

A running-repair order (:class:`RepairOrder`) grounds one locomotive for a
window. Insertion never silently invalidates prior cycles: callers choose an
explicit :data:`conflict policy <POLICY_PRESERVE_FEASIBLE>` — either keep every
prior train feasible by re-powering unlocked cycles (each change is reported),
or roll the whole insertion back and leave the board untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from datetime import datetime, timedelta

from railforge.core.time import utc
from railforge.locomotive.tractive_effort import required_effort
from railforge.maintenance.release import Evidence, ReleasePolicy, releasable

#: Keep every prior train feasible: unlocked cycles may be re-powered, and each
#: re-powering is returned as an explicit :class:`Change`. Locked trains cannot
#: be re-powered, so a conflict touching one rejects the insertion.
POLICY_PRESERVE_FEASIBLE = "preserve_feasible"

#: Refuse the repair order whenever it would touch any prior cycle at all.
POLICY_ROLLBACK_ALL = "rollback_all"

CONFLICT_POLICIES = frozenset({POLICY_PRESERVE_FEASIBLE, POLICY_ROLLBACK_ALL})

# Insertion outcomes (see InsertionOutcome.outcome).
OUTCOME_INSERTED = "inserted"
OUTCOME_ROLLED_BACK = "rolled_back"
OUTCOME_PRESERVE_INFEASIBLE = "preserve_infeasible"
OUTCOME_GROUNDS_INCOMPLETE = "grounds_incomplete"


@dataclass(frozen=True)
class Loco:
    loco_id: str
    location: str
    available_at: datetime
    continuous_kn: float
    adhesion_kn: float
    due_hours: float


@dataclass(frozen=True)
class TrainMove:
    """One train leg that needs ``required_units`` locomotives in consist."""
    move_id: str
    origin: str
    destination: str
    depart: datetime
    arrive: datetime
    train_tonnes: float
    gradient_permille: float
    curve_resistance_kn: float
    required_units: int
    #: A locked train has an externally published consist that must never change.
    locked: bool = False


@dataclass(frozen=True)
class RepairOrder:
    """Unscheduled repair that grounds ``loco_id`` at ``location`` for a window."""
    repair_id: str
    loco_id: str
    location: str
    start: datetime
    end: datetime
    #: Evidence kinds (e.g. defect report) required to authorise grounding.
    grounds_kinds: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Assignment:
    move_id: str
    loco_ids: tuple[str, ...]


@dataclass(frozen=True)
class Change:
    """An explicit re-powering of a prior cycle — never silent."""
    move_id: str
    before: tuple[str, ...]
    after: tuple[str, ...]


@dataclass(frozen=True)
class Conflict:
    code: str
    detail: str
    move_id: str | None = None
    loco_id: str | None = None
    repair_id: str | None = None


@dataclass(frozen=True)
class InsertionOutcome:
    outcome: str
    policy: str
    conflicts: tuple[Conflict, ...]
    assignments: tuple[Assignment, ...]
    changes: tuple[Change, ...]


class DispatchConflictError(RuntimeError):
    """Raised when a plan commit cannot satisfy all five planning factors."""

    def __init__(self, conflicts: list[Conflict] | tuple[Conflict, ...]):
        self.conflicts = tuple(conflicts)
        codes = ", ".join(sorted({c.code for c in self.conflicts}))
        super().__init__(f"dispatch conflicts: {codes}")


@dataclass
class _State:
    loco_id: str
    location: str
    ready_at: datetime
    due_hours: float
    continuous_kn: float
    adhesion_kn: float
    repairs: list[RepairOrder]
    moves: list[str]

    def advance(self, at: datetime) -> None:
        """Apply every repair window that has ended by ``at``."""
        while self.repairs and self.repairs[0].end <= at:
            r = self.repairs.pop(0)
            self.location = r.location
            if self.ready_at < r.end:
                self.ready_at = r.end


def _normalise_loco(l: Loco) -> Loco:
    if l.continuous_kn <= 0 or l.adhesion_kn <= 0 or l.due_hours <= 0:
        raise ValueError(f"invalid locomotive {l.loco_id}")
    return replace(l, available_at=utc(l.available_at))


def _normalise_move(m: TrainMove) -> TrainMove:
    depart, arrive = utc(m.depart), utc(m.arrive)
    if arrive <= depart:
        raise ValueError(f"invalid move interval {m.move_id}")
    if m.required_units < 1:
        raise ValueError(f"invalid required_units {m.move_id}")
    if m.train_tonnes <= 0 or m.curve_resistance_kn < 0:
        raise ValueError(f"invalid train {m.move_id}")
    return replace(m, depart=depart, arrive=arrive)


def _normalise_repair(r: RepairOrder) -> RepairOrder:
    start, end = utc(r.start), utc(r.end)
    if end <= start:
        raise ValueError(f"invalid repair window {r.repair_id}")
    return replace(r, start=start, end=end, grounds_kinds=frozenset(r.grounds_kinds))


def _blocked_by_repair(st: _State, depart: datetime, arrive: datetime) -> RepairOrder | None:
    for r in st.repairs:
        if r.start < arrive and r.end > depart:
            return r
    return None


def _effort(locos: list[_State] | tuple[_State, ...], adhesion_factor: float) -> float:
    return sum(min(s.continuous_kn, s.adhesion_kn * adhesion_factor) for s in locos)


def _individual_blockers(st: _State, m: TrainMove, evidence: dict[str, tuple[Evidence, ...]],
                         policy: ReleasePolicy, trip_hours: float) -> list[str]:
    st.advance(m.depart)
    blockers: list[str] = []
    if _blocked_by_repair(st, m.depart, m.arrive) is not None:
        blockers.append("repair_overlap")
    if st.location != m.origin:
        blockers.append("unavailable")
    elif st.ready_at > m.depart:
        blockers.append("turnaround" if st.moves else "unavailable")
    if st.due_hours <= trip_hours:
        blockers.append("due_hours")
    if not releasable(list(evidence.get(st.loco_id, ())), policy, m.depart):
        blockers.append("evidence")
    return blockers


def _repair_audit(locos: list[Loco], moves_by_id: dict[str, TrainMove],
                  assigned: dict[str, tuple[str, ...]],
                  repairs: list[RepairOrder]) -> list[Conflict]:
    """Flag repair windows that overlap committed moves or that the loco cannot
    physically reach in time. Runs after assignment so it also covers locos
    that never reach their shop window."""
    out: list[Conflict] = []
    by_loco: dict[str, list[TrainMove]] = {}
    for move_id, loco_ids in assigned.items():
        m = moves_by_id[move_id]
        for lid in loco_ids:
            by_loco.setdefault(lid, []).append(m)
    initial = {l.loco_id: l for l in locos}
    for r in sorted(repairs, key=lambda x: (x.start, x.repair_id)):
        rows = sorted(by_loco.get(r.loco_id, ()), key=lambda m: (m.depart, m.move_id))
        overlapping = [m for m in rows if m.depart < r.end and m.arrive > r.start]
        for m in overlapping:
            out.append(Conflict("repair_overlap",
                                f"loco {r.loco_id} grounded {r.start.isoformat()}.."
                                f"{r.end.isoformat()} during move {m.move_id}",
                                move_id=m.move_id, loco_id=r.loco_id, repair_id=r.repair_id))
        if overlapping:
            continue
        prior = [m for m in rows if m.arrive <= r.start]
        if prior:
            site_ok = prior[-1].destination == r.location
        else:
            init = initial[r.loco_id]
            site_ok = init.location == r.location and init.available_at <= r.start
        if not site_ok:
            out.append(Conflict("unreachable_shop",
                                f"loco {r.loco_id} cannot reach {r.location} by {r.start.isoformat()}",
                                loco_id=r.loco_id, repair_id=r.repair_id))
    return out


def solve(locos: list[Loco], moves: list[TrainMove],
          fixed: dict[str, tuple[str, ...]],
          repairs: list[RepairOrder] | None = None,
          evidence_by_loco: dict[str, tuple[Evidence, ...]] | None = None,
          required_kinds: frozenset[str] = frozenset(),
          adhesion_factor: float = 1.0,
          turn_minutes: int = 0) -> tuple[list[Assignment], list[Conflict]]:
    """Assign locomotives to every move and return ``(assignments, conflicts)``.

    ``fixed`` maps move ids to consists that must be kept verbatim (existing
    train plans / locked trains). Every other move is powered from scratch,
    respecting all five factors at once. Deterministic: identical inputs yield
    identical assignments, which is what makes event replay idempotent.
    """
    if not 0 < adhesion_factor <= 1:
        raise ValueError("adhesion factor")
    if turn_minutes < 0:
        raise ValueError("turn_minutes")
    turn = timedelta(minutes=turn_minutes)
    release = ReleasePolicy(frozenset(required_kinds))
    evidence = dict(evidence_by_loco or {})
    repairs = [_normalise_repair(r) for r in (repairs or [])]

    states: dict[str, _State] = {}
    for raw in locos:
        l = _normalise_loco(raw)
        states[l.loco_id] = _State(l.loco_id, l.location, l.available_at, l.due_hours,
                                   l.continuous_kn, l.adhesion_kn, [], [])
    for r in repairs:
        if r.loco_id not in states:
            raise ValueError(f"repair {r.repair_id} references unknown loco {r.loco_id}")
        states[r.loco_id].repairs.append(r)
    for st in states.values():
        st.repairs.sort(key=lambda r: (r.start, r.repair_id))

    ordered = sorted((_normalise_move(m) for m in moves), key=lambda m: (m.depart, m.move_id))
    fixed = dict(fixed)
    unknown = set(fixed) - {m.move_id for m in ordered}
    if unknown:
        raise ValueError(f"fixed assignment for unknown moves: {sorted(unknown)}")

    assignments: list[Assignment] = []
    conflicts: list[Conflict] = []
    moves_by_id: dict[str, TrainMove] = {}

    if required_kinds and ordered:
        first_depart = ordered[0].depart
        for lid in sorted(states):
            if not releasable(list(evidence.get(lid, ())), release, first_depart):
                # Evidence is a board-level gate: a loco that fails it must not
                # be silently hidden behind a larger pool of healthy locos.
                conflicts.append(Conflict("evidence", f"loco {lid} lacks valid evidence for "
                                          f"{sorted(required_kinds)}", loco_id=lid))

    for m in ordered:
        moves_by_id[m.move_id] = m
        trip_hours = (m.arrive - m.depart).total_seconds() / 3600
        needed_kn = required_effort(m.train_tonnes, m.gradient_permille, m.curve_resistance_kn)

        if m.move_id in fixed:
            chosen_ids = list(fixed[m.move_id])
            for lid in chosen_ids:
                st = states.get(lid)
                if st is None:
                    conflicts.append(Conflict("unavailable", f"unknown loco {lid}",
                                              move_id=m.move_id, loco_id=lid))
                    continue
                for code in _individual_blockers(st, m, evidence, release, trip_hours):
                    repair = _blocked_by_repair(st, m.depart, m.arrive)
                    conflicts.append(Conflict(code, f"fixed move {m.move_id} fails {code} on {lid}",
                                              move_id=m.move_id, loco_id=lid,
                                              repair_id=repair.repair_id if repair else None))
            consist = [states[lid] for lid in chosen_ids if lid in states]
            if consist and _effort(consist, adhesion_factor) < needed_kn:
                conflicts.append(Conflict("traction",
                                          f"fixed move {m.move_id} consist effort below resistance",
                                          move_id=m.move_id))
        else:
            eligible: list[_State] = []
            blocker_summary: dict[str, list[str]] = {}
            for st in states.values():
                blockers = _individual_blockers(st, m, evidence, release, trip_hours)
                if blockers:
                    blocker_summary[st.loco_id] = blockers
                else:
                    eligible.append(st)
            best: tuple[str, ...] | None = None
            best_key = None
            for combo in combinations(eligible, m.required_units):
                if _effort(combo, adhesion_factor) < needed_kn:
                    continue
                ids = tuple(sorted(s.loco_id for s in combo))
                key = (max(s.ready_at for s in combo), -sum(s.due_hours for s in combo), ids)
                if best_key is None or key < best_key:
                    best_key, best = key, ids
            if best is None:
                enough = len(eligible) >= m.required_units
                code = "traction" if enough else "locomotive_shortage"
                detail = f"{code} for move {m.move_id}; eligible={len(eligible)} required={m.required_units}"
                if blocker_summary:
                    flat = sorted({b for bs in blocker_summary.values() for b in bs})
                    detail += "; blockers=" + ",".join(flat)
                conflicts.append(Conflict(code, detail, move_id=m.move_id))
                continue
            chosen_ids = list(best)

        assignments.append(Assignment(m.move_id, tuple(chosen_ids)))
        for lid in chosen_ids:
            st = states[lid]
            st.advance(m.depart)
            st.location = m.destination
            st.ready_at = max(st.ready_at, m.arrive) + turn
            st.due_hours -= trip_hours
            st.moves.append(m.move_id)

    assigned_map = {a.move_id: a.loco_ids for a in assignments}
    conflicts.extend(_repair_audit(list(locos), moves_by_id, assigned_map, repairs))

    # A fixed-consist overlap is seen both while applying the fixed move and by
    # the post-audit; collapse to one conflict per (code, move, loco, repair).
    dedup: dict[tuple, Conflict] = {}
    for c in conflicts:
        key = (c.code, c.move_id, c.loco_id, c.repair_id)
        if key not in dedup:
            dedup[key] = c
    return assignments, [dedup[k] for k in sorted(dedup)]


def evaluate_insertion(locos: list[Loco], moves: list[TrainMove],
                       assignments: list[Assignment] | tuple[Assignment, ...],
                       repairs: list[RepairOrder] | tuple[RepairOrder, ...],
                       evidence_by_loco: dict[str, tuple[Evidence, ...]],
                       required_kinds: frozenset[str],
                       repair: RepairOrder,
                       grounds_evidence: list[Evidence] | tuple[Evidence, ...],
                       adhesion_factor: float,
                       turn_minutes: int,
                       policy: str) -> InsertionOutcome:
    """Evaluate a running-repair order against the committed dispatch board.

    Does not mutate anything. Returns the resulting full assignment list plus
    explicit :class:`Change` rows when prior cycles had to be re-powered.
    """
    if policy not in CONFLICT_POLICIES:
        raise ValueError(f"unknown conflict policy {policy}")
    repair = _normalise_repair(repair)
    current = [_normalise_repair(r) for r in repairs]
    if any(r.repair_id == repair.repair_id for r in current):
        raise ValueError(f"repair {repair.repair_id} already inserted")
    if not any(l.loco_id == repair.loco_id for l in locos):
        raise ValueError(f"repair {repair.repair_id} references unknown loco {repair.loco_id}")

    old = {a.move_id: tuple(sorted(a.loco_ids)) for a in assignments}
    no_changes: tuple[Change, ...] = ()
    old_assignments = tuple(assignments)

    # The defect/grounds evidence must itself be complete and valid.
    if not releasable(list(grounds_evidence), ReleasePolicy(repair.grounds_kinds), repair.start):
        return InsertionOutcome(OUTCOME_GROUNDS_INCOMPLETE, policy, (
            Conflict("repair_grounds",
                     f"grounds evidence incomplete for repair {repair.repair_id}",
                     loco_id=repair.loco_id, repair_id=repair.repair_id),
        ), old_assignments, no_changes)

    planned_repairs = current + [repair]

    if policy == POLICY_ROLLBACK_ALL:
        # Nothing existing may move: every current consist stays fixed.
        _, conflicts = solve(locos, moves, dict(old), planned_repairs,
                             evidence_by_loco, required_kinds, adhesion_factor, turn_minutes)
        if conflicts:
            return InsertionOutcome(OUTCOME_ROLLED_BACK, policy, tuple(conflicts),
                                    old_assignments, no_changes)
        return InsertionOutcome(OUTCOME_INSERTED, policy, (), old_assignments, no_changes)

    # PRESERVE_FEASIBLE: locked consists stay fixed; unlocked cycles may be
    # re-powered — but only explicitly, via the returned Change rows.
    locked_moves = [m for m in moves if m.locked and m.move_id in old]
    locked = {m.move_id: old[m.move_id] for m in locked_moves}
    unlocked_moves = [m for m in moves if not m.locked]
    new_assignments, conflicts = solve(
        locos, unlocked_moves + locked_moves, locked, planned_repairs,
        evidence_by_loco, required_kinds, adhesion_factor, turn_minutes)
    if conflicts:
        return InsertionOutcome(OUTCOME_PRESERVE_INFEASIBLE, policy, tuple(conflicts),
                                old_assignments, no_changes)

    new_map = {a.move_id: tuple(sorted(a.loco_ids)) for a in new_assignments}
    full = {**old, **new_map}
    changes = tuple(
        Change(move_id, old[move_id], full[move_id])
        for move_id in sorted(old) if full.get(move_id) != old[move_id]
    )
    ordered_ids = [m.move_id for m in sorted(moves, key=lambda m: (m.depart, m.move_id))]
    result = tuple(Assignment(mid, full[mid]) for mid in ordered_ids if mid in full)
    return InsertionOutcome(OUTCOME_INSERTED, policy, (), result, changes)
