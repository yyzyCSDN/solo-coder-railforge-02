"""Versioned locomotive-dispatch workflow: cycles + running-repair insertion.

All five planning factors (tractive capacity, locomotive availability,
maintenance evidence, round-trip turnaround, existing train plans) are read and
committed through *one* versioned board. Every state transition is:

1. computed purely against a board snapshot taken at an expected version,
2. written with a compare-and-set (so a concurrent publication raises
   ``VersionConflict`` instead of overwriting),
3. recorded as a decision event carrying the full post-state snapshot, and
4. appended to the hash-chained audit trail.

Running-repair insertion is explicit about failure: ``preserve_feasible``
re-powers only unlocked cycles and reports every :class:`Change`; a locked
conflict or any conflict under ``rollback_all`` rejects the insertion and
leaves the board version untouched. Decisions are themselves events, so the
whole history is replayable, and the same ``operation_id`` is idempotent.
"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from json import dumps
from threading import RLock
from typing import Any, Mapping

from railforge.core.versioning import VersionConflict
from railforge.events.stream import EventStream, StreamEvent
from railforge.audit.trail import Trail
from railforge.maintenance.release import Evidence, ReleasePolicy
from railforge.planning.loco_dispatch import (
    Assignment,
    Conflict,
    DispatchConflictError,
    Loco,
    RepairOrder,
    TrainMove,
    evaluate_insertion,
    solve,
)
from railforge.storage.ops_store import VersionedStore

TOPIC_BOARD_OPENED = "loco.dispatch.board.opened"
TOPIC_CYCLE_PLANNED = "loco.cycle.planned"
TOPIC_REPAIR_INSERTED = "loco.repair.inserted"
TOPIC_REPAIR_REJECTED = "loco.repair.rejected"

DISPATCH_TOPICS = frozenset({
    TOPIC_BOARD_OPENED, TOPIC_CYCLE_PLANNED,
    TOPIC_REPAIR_INSERTED, TOPIC_REPAIR_REJECTED,
})

STATUS_OPENED = "opened"
STATUS_COMMITTED = "committed"
STATUS_INSERTED = "inserted"
STATUS_REJECTED = "rejected"


@dataclass(frozen=True)
class DispatchBoard:
    locos: tuple[Loco, ...]
    moves: tuple[TrainMove, ...]
    assignments: tuple[Assignment, ...]
    repairs: tuple[RepairOrder, ...]
    evidence_by_loco: Mapping[str, tuple[Evidence, ...]]
    adhesion_factor: float
    turn_minutes: int
    required_kinds: frozenset[str]


@dataclass(frozen=True)
class LocoDispatchReceipt:
    operation_id: str
    topic: str
    subject: str
    status: str
    version: int
    result: Mapping[str, Any]
    event: StreamEvent
    audit_index: int


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _jsonable(value: Any) -> Any:
    """Deterministic JSON-safe encoding used for snapshots and idempotency keys."""
    if is_dataclass(value):
        kind = type(value).__name__
        return {"_kind": kind, **{k: _jsonable(v) for k, v in asdict(value).items()}}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, timedelta):
        return {"_kind": "timedelta", "seconds": value.total_seconds()}
    if isinstance(value, datetime):
        return {"_kind": "datetime", "iso": _utc(value).astimezone(timezone.utc).isoformat()}
    if isinstance(value, Decimal):
        return str(value)
    return value


def _fingerprint(parts: Mapping[str, Any]) -> str:
    return dumps(_jsonable(parts), sort_keys=True, separators=(",", ":"))


def _dt(data: Mapping[str, Any], key: str) -> datetime:
    value = data[key]
    if isinstance(value, Mapping):  # nested {"_kind": "datetime", "iso": ...}
        value = value["iso"]
    return datetime.fromisoformat(value)


def _loco(data: Mapping[str, Any]) -> Loco:
    return Loco(data["loco_id"], data["location"], _dt(data, "available_at"),
                data["continuous_kn"], data["adhesion_kn"], data["due_hours"])


def _move(data: Mapping[str, Any]) -> TrainMove:
    return TrainMove(data["move_id"], data["origin"], data["destination"],
                     _dt(data, "depart"), _dt(data, "arrive"), data["train_tonnes"],
                     data["gradient_permille"], data["curve_resistance_kn"],
                     data["required_units"], data["locked"])


def _repair(data: Mapping[str, Any]) -> RepairOrder:
    return RepairOrder(data["repair_id"], data["loco_id"], data["location"],
                       _dt(data, "start"), _dt(data, "end"), frozenset(data["grounds_kinds"]))


def _assignment(data: Mapping[str, Any]) -> Assignment:
    return Assignment(data["move_id"], tuple(data["loco_ids"]))


def _evidence(data: Mapping[str, Any]) -> Evidence:
    return Evidence(data["kind"], _dt(data, "completed_at"),
                    timedelta(seconds=data["valid_for"]["seconds"]), data["passed"])


def _board_from_data(data: Mapping[str, Any]) -> DispatchBoard:
    evidence = {
        lid: tuple(_evidence(e) for e in rows)
        for lid, rows in data["evidence_by_loco"].items()
    }
    return DispatchBoard(
        tuple(_loco(x) for x in data["locos"]),
        tuple(_move(x) for x in data["moves"]),
        tuple(_assignment(x) for x in data["assignments"]),
        tuple(_repair(x) for x in data["repairs"]),
        evidence,
        data["adhesion_factor"], data["turn_minutes"],
        frozenset(data["required_kinds"]),
    )


@dataclass(frozen=True)
class ReplayReport:
    applied: int
    decisions: tuple[StreamEvent, ...]
    board: DispatchBoard | None
    last_version: int


class LocoDispatchWorkflow:
    """Single-version-boundary workflow for loco cycles and repair insertion."""

    def __init__(self,
                 stream: EventStream | None = None,
                 trail: Trail | None = None,
                 store: VersionedStore | None = None,
                 subject: str = "loco-dispatch:default"):
        self.stream = stream or EventStream()
        self.trail = trail or Trail()
        self.store = store or VersionedStore()
        self.subject = subject
        self._board: DispatchBoard | None = None
        self._receipts: dict[str, LocoDispatchReceipt] = {}
        # Serialises the read-compute-commit transaction: the CAS happens while
        # holding this lock, so two planners cannot both read version N and then
        # both commit N+1.
        self._lock = RLock()

    # ------------------------------------------------------------------ state
    @property
    def board(self) -> DispatchBoard | None:
        return self._board

    @property
    def version(self) -> int:
        row = self.store.read(self.subject)
        return row.version if row else 0

    def receipts(self) -> Mapping[str, LocoDispatchReceipt]:
        return dict(self._receipts)

    def events_after(self, cursor: int, limit: int = 100):
        return self.stream.page(cursor, limit=limit, topics=DISPATCH_TOPICS)

    # --------------------------------------------------------------- internal
    def _require_open(self) -> DispatchBoard:
        if self._board is None:
            raise RuntimeError("dispatch board not opened")
        return self._board

    def _idempotency_gate(self, operation_id: str, fingerprint: str) -> LocoDispatchReceipt | None:
        prior = self._receipts.get(operation_id)
        if prior is None:
            return None
        if prior.event.payload.get("fingerprint") != fingerprint:
            raise ValueError("operation id reused with different arguments")
        return prior

    def _commit(self, operation_id: str, topic: str, status: str, at: datetime,
                fingerprint: str, expected_version: int,
                build) -> LocoDispatchReceipt:
        """CAS + decision event + audit, all or nothing from the caller's view."""
        result, board = build()
        # The store write is the version boundary: a concurrent planner wins and
        # we fail loudly instead of overwriting their dispatch.
        try:
            version = self.store.write(self.subject, _jsonable(board), expected_version)
        except RuntimeError as exc:  # VersionedStore raises a plain RuntimeError
            if str(exc) == "version conflict":
                raise VersionConflict(
                    f"{self.subject}: expected={expected_version} actual={self.version}") from exc
            raise
        payload = {
            "operation_id": operation_id,
            "status": status,
            "version": version,
            "fingerprint": fingerprint,
            "result": _jsonable(result),
            "board": _jsonable(board),
        }
        event = self.stream.publish(operation_id, _utc(at), topic, self.subject, payload)
        audit = self.trail.append("loco-dispatch", topic, self.subject, _utc(at))
        receipt = LocoDispatchReceipt(operation_id, topic, self.subject, status,
                                      version, result, event, audit.index)
        self._board = board
        self._receipts[operation_id] = receipt
        return receipt

    # -------------------------------------------------------------- commands
    def open_board(self, operation_id: str, locos: list[Loco], moves: list[TrainMove],
                   evidence_by_loco: dict[str, tuple[Evidence, ...]] | None = None,
                   adhesion_factor: float = 1.0, turn_minutes: int = 30,
                   required_kinds: frozenset[str] = frozenset(),
                   at: datetime | None = None,
                   expected_version: int | None = None) -> LocoDispatchReceipt:
        with self._lock:
            at = at or datetime.now(timezone.utc)
            if self._board is not None or self.version != 0:
                raise RuntimeError("dispatch board already opened")
            fp = _fingerprint({"op": "open", "locos": locos, "moves": moves,
                               "evidence": evidence_by_loco or {}, "adhesion_factor": adhesion_factor,
                               "turn_minutes": turn_minutes, "required_kinds": frozenset(required_kinds)})
            prior = self._idempotency_gate(operation_id, fp)
            if prior is not None:
                return prior

            board = DispatchBoard(
                tuple(locos), tuple(moves), (), (), dict(evidence_by_loco or {}),
                adhesion_factor, turn_minutes, frozenset(required_kinds),
            )
            result = {"status": STATUS_OPENED, "locos": len(board.locos),
                      "moves": len(board.moves), "version": 1}
            return self._commit(operation_id, TOPIC_BOARD_OPENED, STATUS_OPENED, at, fp,
                                expected_version or 0, lambda: (result, board))

    def plan_cycles(self, operation_id: str, at: datetime,
                    additional_moves: list[TrainMove] | None = None,
                    additional_locos: list[Loco] | None = None,
                    additional_evidence: dict[str, tuple[Evidence, ...]] | None = None,
                    expected_version: int | None = None) -> LocoDispatchReceipt:
        """Power every (newly added or still-unassigned) move inside one commit."""
        with self._lock:
            board = self._require_open()
            fp = _fingerprint({"op": "plan", "additional_moves": additional_moves or [],
                               "additional_locos": additional_locos or [],
                               "additional_evidence": additional_evidence or {}})
            prior = self._idempotency_gate(operation_id, fp)
            if prior is not None:
                return prior

            locos = tuple(board.locos) + tuple(additional_locos or ())
            evidence = {k: tuple(v) for k, v in board.evidence_by_loco.items()}
            for lid, rows in (additional_evidence or {}).items():
                evidence[lid] = tuple(evidence.get(lid, ())) + tuple(rows)
            moves = tuple(board.moves) + tuple(additional_moves or ())
            ids = {l.loco_id for l in locos}
            if len(ids) != len(locos):
                raise ValueError("duplicate locomotive id")
            mids = [m.move_id for m in moves]
            if len(set(mids)) != len(mids):
                raise ValueError("duplicate move id")
            fixed = {a.move_id: a.loco_ids for a in board.assignments}

            assignments, conflicts = solve(
                list(locos), list(moves), fixed, list(board.repairs), evidence,
                board.required_kinds, board.adhesion_factor, board.turn_minutes)
            if conflicts:
                raise DispatchConflictError(conflicts)

            new_board = DispatchBoard(locos, moves, tuple(assignments), board.repairs, evidence,
                                      board.adhesion_factor, board.turn_minutes,
                                      board.required_kinds)
            result = {"status": STATUS_COMMITTED,
                      "assignments": [{"move_id": a.move_id, "loco_ids": list(a.loco_ids)}
                                      for a in assignments],
                      "cycles_added": len(assignments) - len(board.assignments),
                      "version": 0}
            expected = self.version if expected_version is None else expected_version

            def build():
                result["version"] = expected + 1
                return result, new_board

            return self._commit(operation_id, TOPIC_CYCLE_PLANNED, STATUS_COMMITTED, at, fp,
                                expected, build)

    def insert_repair(self, operation_id: str, repair: RepairOrder,
                      grounds_evidence: list[Evidence] | tuple[Evidence, ...],
                      policy: str, at: datetime,
                      expected_version: int | None = None) -> LocoDispatchReceipt:
        """Insert a running-repair order; rejection is a committed decision too."""
        with self._lock:
            board = self._require_open()
            fp = _fingerprint({"op": "insert", "repair": repair,
                               "grounds_evidence": list(grounds_evidence), "policy": policy})
            prior = self._idempotency_gate(operation_id, fp)
            if prior is not None:
                return prior

            outcome = evaluate_insertion(
                list(board.locos), list(board.moves), board.assignments,
                list(board.repairs), dict(board.evidence_by_loco), board.required_kinds,
                repair, list(grounds_evidence), board.adhesion_factor,
                board.turn_minutes, policy)

            conflicts = [_jsonable(c) for c in outcome.conflicts]
            changes = [_jsonable(ch) for ch in outcome.changes]
            expected = self.version if expected_version is None else expected_version

            if outcome.outcome == "inserted":
                new_board = DispatchBoard(
                    board.locos, board.moves, tuple(outcome.assignments),
                    tuple(board.repairs) + (repair,), board.evidence_by_loco,
                    board.adhesion_factor, board.turn_minutes, board.required_kinds)
                result = {"status": STATUS_INSERTED, "outcome": outcome.outcome,
                          "repair_id": repair.repair_id, "policy": policy,
                          "changes": changes, "conflicts": [], "cycles_repowered": len(changes),
                          "version": 0}

                def build():
                    result["version"] = expected + 1
                    return result, new_board

                return self._commit(operation_id, TOPIC_REPAIR_INSERTED, STATUS_INSERTED, at,
                                    fp, expected, build)

            # Rejected: the decision must be replayable, but the board version and
            # assignments stay exactly where they were — an explicit rollback.
            result = {"status": STATUS_REJECTED, "outcome": outcome.outcome,
                      "repair_id": repair.repair_id, "policy": policy,
                      "changes": [], "conflicts": conflicts, "cycles_repowered": 0,
                      "version": expected}
            event = self.stream.publish(
                operation_id, _utc(at), TOPIC_REPAIR_REJECTED, self.subject, {
                    "operation_id": operation_id,
                    "status": STATUS_REJECTED,
                    "version": expected,
                    "fingerprint": fp,
                    "result": _jsonable(result),
                })
            audit = self.trail.append("loco-dispatch", TOPIC_REPAIR_REJECTED, self.subject, _utc(at))
            receipt = LocoDispatchReceipt(operation_id, TOPIC_REPAIR_REJECTED, self.subject,
                                          STATUS_REJECTED, expected, result, event, audit.index)
            self._receipts[operation_id] = receipt
            return receipt

    # ---------------------------------------------------------------- replay
    def replay(self, upto: int | None = None) -> ReplayReport:
        """Rebuild the board from the decision-event history (event sourcing)."""
        with self._lock:
            board: DispatchBoard | None = None
            decisions: list[StreamEvent] = []
            applied = 0
            cursor = 0
            last_version = 0
            while True:
                page = self.stream.page(cursor, limit=500, topics=DISPATCH_TOPICS)
                for event in page.events:
                    if upto is not None and event.sequence > upto:
                        continue
                    applied += 1
                    last_version = int(event.payload.get("version", last_version))
                    decisions.append(event)
                    if event.payload.get("status") != STATUS_REJECTED:
                        board = _board_from_data(event.payload["board"])
                    self._receipts[event.payload["operation_id"]] = LocoDispatchReceipt(
                        event.payload["operation_id"], event.topic, event.subject,
                        event.payload["status"], int(event.payload["version"]),
                        dict(event.payload.get("result", {})), event, -1)
                cursor = page.next_cursor
                if not page.events:
                    break
            self._board = board
            return ReplayReport(applied, tuple(decisions), board, last_version)
