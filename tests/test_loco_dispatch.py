import unittest
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.core.versioning import VersionConflict
from railforge.maintenance.release import Evidence
from railforge.planning.loco_dispatch import (
    Assignment,
    DispatchConflictError,
    Loco,
    POLICY_PRESERVE_FEASIBLE,
    POLICY_ROLLBACK_ALL,
    RepairOrder,
    TrainMove,
    solve,
)
from railforge.workflows.loco_dispatch import (
    LocoDispatchWorkflow,
    TOPIC_CYCLE_PLANNED,
    TOPIC_REPAIR_INSERTED,
    TOPIC_REPAIR_REJECTED,
)

U = timezone.utc
T0 = datetime(2026, 1, 1, 0, tzinfo=U)


def t(hour):
    return datetime(2026, 1, 1, hour, tzinfo=U)


# 2000 t @ 25 permille needs ~526 kN; each loco pulls 600 kN, so two units.
def locos():
    return [
        Loco("L1", "A", t(0), 600, 600, 100),
        Loco("L2", "A", t(0), 600, 600, 100),
        Loco("L3", "A", t(0), 600, 600, 100),
    ]


def cycle_moves():
    return [
        TrainMove("M1", "A", "B", t(8), t(10), 2000, 25, 0, 2),
        TrainMove("M2", "B", "A", t(12), t(14), 2000, 25, 0, 2),
        TrainMove("M3", "A", "B", t(16), t(18), 2000, 25, 0, 2),
    ]


def inspection(loco_id, at=T0):
    return {loco_id: (Evidence("inspection", at, timedelta(days=30), True),)}


def all_inspection():
    out = {}
    for l in locos():
        out.update(inspection(l.loco_id))
    return out


class DispatchEngineTests(unittest.TestCase):

    def test_happy_path_cycle_respects_all_five_factors(self):
        assignments, conflicts = solve(
            locos(), cycle_moves(), {}, adhesion_factor=1.0, turn_minutes=60)
        self.assertEqual(conflicts, [])
        by_id = {a.move_id: set(a.loco_ids) for a in assignments}
        self.assertEqual(len(assignments), 3)
        # Round trip: M1 pair reaches B at 10:00, turns 60 min, works M2 at 12:00.
        self.assertEqual(by_id["M1"], by_id["M2"])
        for a in assignments:
            self.assertEqual(len(a.loco_ids), 2)

    def test_traction_shortage_is_structural_conflict(self):
        # Only 200 kN per loco -> even two units (400 kN) cannot produce ~526 kN.
        weak = [Loco(x, "A", t(0), 200, 200, 100) for x in ("L1", "L2", "L3")]
        assignments, conflicts = solve(weak, cycle_moves(), {}, adhesion_factor=1.0, turn_minutes=60)
        codes = {c.code for c in conflicts}
        self.assertIn("traction", codes)
        self.assertEqual(assignments, [])

    def test_turnaround_violation_blocks_reuse(self):
        # M2 departs B at 10:30 while the pair only arrives at 10:00 -> the 60
        # minute turn is impossible and no spare locos exist at B.
        tight = [
            TrainMove("M1", "A", "B", t(8), t(10), 2000, 25, 0, 2),
            TrainMove("M2", "B", "A", t(10) + timedelta(minutes=30),
                      t(12) + timedelta(minutes=30), 2000, 25, 0, 2),
        ]
        _, conflicts = solve(locos(), tight, {},
                             adhesion_factor=1.0, turn_minutes=60)
        self.assertTrue(any(c.code == "locomotive_shortage" for c in conflicts))

    def test_maintenance_evidence_required_in_same_solution(self):
        evidence = all_inspection()
        # L2's inspection expired before the plan starts.
        evidence["L2"] = (Evidence("inspection", t(0) - timedelta(days=40),
                                   timedelta(days=1), True),)
        _, conflicts = solve(locos(), cycle_moves(), {}, evidence_by_loco=evidence,
                             required_kinds=frozenset({"inspection"}),
                             adhesion_factor=1.0, turn_minutes=60)
        self.assertTrue(any(c.code == "evidence" and c.loco_id == "L2" for c in conflicts))

    def test_existing_fixed_plan_is_kept_and_its_conflicts_are_reported(self):
        fixed = {"M1": ("L1", "L2")}
        assignments, conflicts = solve(locos(), cycle_moves(), fixed,
                                       adhesion_factor=1.0, turn_minutes=60)
        by_id = {a.move_id: a.loco_ids for a in assignments}
        self.assertEqual(by_id["M1"], ("L1", "L2"))
        self.assertEqual(conflicts, [])

    def test_locked_repair_conflict_forces_rollback(self):
        assignments, _ = solve(locos(), cycle_moves(), {}, adhesion_factor=1.0, turn_minutes=60)
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        from railforge.planning.loco_dispatch import evaluate_insertion
        outcome = evaluate_insertion(
            locos(), cycle_moves(), assignments, [], {}, frozenset(),
            repair, grounds, 1.0, 60, POLICY_ROLLBACK_ALL)
        self.assertEqual(outcome.outcome, "rolled_back")
        self.assertTrue(any(c.code == "repair_overlap" and c.move_id == "M3"
                            for c in outcome.conflicts))
        # Rollback keeps the prior consists verbatim.
        self.assertEqual(tuple(assignments), outcome.assignments)
        self.assertEqual(outcome.changes, ())

    def test_preserve_feasible_reports_every_repowering_explicitly(self):
        from railforge.planning.loco_dispatch import evaluate_insertion
        assignments, _ = solve(locos(), cycle_moves(), {}, adhesion_factor=1.0, turn_minutes=60)
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        outcome = evaluate_insertion(
            locos(), cycle_moves(), assignments, [], {}, frozenset(),
            repair, grounds, 1.0, 60, POLICY_PRESERVE_FEASIBLE)
        self.assertEqual(outcome.outcome, "inserted")
        # L1 is grounded over M3; M3 keeps running because it is re-powered, and
        # the change is surfaced rather than silently mutating the prior cycle.
        changed = {ch.move_id: (ch.before, ch.after) for ch in outcome.changes}
        self.assertIn("M3", changed)
        self.assertIn("L1", changed["M3"][0])
        self.assertNotIn("L1", changed["M3"][1])
        self.assertNotIn("L1", dict((a.move_id, a.loco_ids)
                                    for a in outcome.assignments)["M3"])

    def test_preserve_rejects_when_no_feasible_plan_exists(self):
        from railforge.planning.loco_dispatch import evaluate_insertion
        moves = [
            TrainMove("M1", "A", "B", t(8), t(10), 2000, 25, 0, 2, locked=True),
            TrainMove("M2", "B", "A", t(12), t(14), 2000, 25, 0, 2),
        ]
        assignments, _ = solve(locos(), moves, {"M1": ("L1", "L2")}, adhesion_factor=1.0, turn_minutes=60)
        # Repair grounds L1 while it is committed to the locked M1 consist.
        repair = RepairOrder("R1", "L1", "B", t(9), t(13), frozenset({"defect"}))
        grounds = [Evidence("defect", t(8), timedelta(days=1), True)]
        outcome = evaluate_insertion(
            locos(), moves, assignments, [], {}, frozenset(),
            repair, grounds, 1.0, 60, POLICY_PRESERVE_FEASIBLE)
        self.assertEqual(outcome.outcome, "preserve_infeasible")
        self.assertTrue(any(c.move_id == "M1" and c.code == "repair_overlap"
                            for c in outcome.conflicts))
        # Infeasibility must not have altered the prior cycles.
        self.assertEqual(tuple(assignments), outcome.assignments)


class DispatchWorkflowTests(unittest.TestCase):

    def setUp(self):
        self.workflow = LocoDispatchWorkflow()

    def _open(self, wf=None, evidence=None):
        wf = wf or self.workflow
        return wf.open_board(
            "op-open", locos(), cycle_moves(),
            evidence_by_loco=all_inspection() if evidence is None else evidence,
            adhesion_factor=1.0, turn_minutes=60,
            required_kinds=frozenset({"inspection"}), at=t(7))

    def test_full_workflow_version_boundary_and_events(self):
        self._open()
        receipt = self.workflow.plan_cycles("op-plan", t(7))
        self.assertEqual(receipt.status, "committed")
        self.assertEqual(receipt.topic, TOPIC_CYCLE_PLANNED)
        self.assertEqual(receipt.version, 2)
        self.assertTrue(self.workflow.stream.verify_chain())
        self.assertTrue(self.workflow.trail.verify())
        # The five factors live in one version snapshot.
        stored = self.workflow.store.read(self.workflow.subject)
        self.assertEqual(stored.version, 2)
        self.assertEqual(len(stored.value["assignments"]), 3)
        self.assertEqual(len(stored.value["evidence_by_loco"]), 3)

    def test_plan_conflict_raises_without_mutating_version(self):
        weak = [Loco(x, "A", t(0), 200, 200, 100) for x in ("L1", "L2", "L3")]
        self.workflow.open_board("op-open", weak, cycle_moves(), at=t(7))
        version_before = self.workflow.version
        with self.assertRaises(DispatchConflictError) as ctx:
            self.workflow.plan_cycles("op-plan", t(7))
        self.assertIn("traction", {c.code for c in ctx.exception.conflicts})
        self.assertEqual(self.workflow.version, version_before)

    def test_preserve_insertion_commits_and_changes_board(self):
        self._open()
        self.workflow.plan_cycles("op-plan", t(7))
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        receipt = self.workflow.insert_repair(
            "op-repair", repair, grounds, POLICY_PRESERVE_FEASIBLE, t(14))
        self.assertEqual(receipt.topic, TOPIC_REPAIR_INSERTED)
        self.assertEqual(receipt.status, "inserted")
        self.assertEqual(receipt.version, 3)
        self.assertEqual(receipt.result["cycles_repowered"], 1)
        after = {a.move_id: a.loco_ids for a in self.workflow.board.assignments}
        self.assertNotIn("L1", after["M3"])
        self.assertIn(repair, self.workflow.board.repairs)

    def test_rollback_insertion_leaves_board_version_untouched(self):
        self._open()
        self.workflow.plan_cycles("op-plan", t(7))
        version_before = self.workflow.version
        before = self.workflow.board.assignments
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        receipt = self.workflow.insert_repair(
            "op-rollback", repair, grounds, POLICY_ROLLBACK_ALL, t(14))
        self.assertEqual(receipt.topic, TOPIC_REPAIR_REJECTED)
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.result["outcome"], "rolled_back")
        # Explicit rollback: same version, same consists, prior cycles valid.
        self.assertEqual(self.workflow.version, version_before)
        self.assertEqual(self.workflow.board.assignments, before)
        self.assertEqual(self.workflow.board.repairs, ())
        # But the decision is itself on the record and replayable.
        self.assertTrue(self.workflow.stream.verify_chain())

    def test_incomplete_grounds_evidence_is_rejected(self):
        self._open()
        self.workflow.plan_cycles("op-plan", t(7))
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        stale = [Evidence("defect", t(10), timedelta(hours=4), True)]
        receipt = self.workflow.insert_repair(
            "op-bad-evidence", repair, stale, POLICY_PRESERVE_FEASIBLE, t(14))
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.result["outcome"], "grounds_incomplete")
        self.assertEqual(self.workflow.board.repairs, ())

    def test_idempotent_retry_returns_same_receipt(self):
        self._open()
        first = self.workflow.plan_cycles("dup", t(7))
        second = self.workflow.plan_cycles("dup", t(7))
        self.assertIs(first, second)
        self.assertEqual(self.workflow.version, 2)
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        r1 = self.workflow.insert_repair("dup-repair", repair, grounds,
                                         POLICY_ROLLBACK_ALL, t(14))
        r2 = self.workflow.insert_repair("dup-repair", repair, grounds,
                                         POLICY_ROLLBACK_ALL, t(14))
        self.assertIs(r1, r2)
        # Only one rejection decision event despite two calls.
        pages = self.workflow.events_after(0, limit=100)
        rejections = [e for e in pages.events if e.topic == TOPIC_REPAIR_REJECTED]
        self.assertEqual(len(rejections), 1)

    def test_same_operation_id_with_different_args_fails(self):
        self._open()
        self.workflow.plan_cycles("same", t(7))
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        self.workflow.insert_repair("same-repair", repair, grounds,
                                    POLICY_PRESERVE_FEASIBLE, t(14))
        other = RepairOrder("R9", "L2", "A", t(15), t(17), frozenset({"defect"}))
        with self.assertRaises(ValueError):
            self.workflow.insert_repair("same-repair", other, grounds,
                                        POLICY_PRESERVE_FEASIBLE, t(14))

    def test_stale_expected_version_conflicts(self):
        self._open()
        self.workflow.plan_cycles("op-a", t(7))
        with self.assertRaises(VersionConflict):
            self.workflow.plan_cycles("op-b", t(7), expected_version=1)

    def test_concurrent_committers_share_one_version_boundary(self):
        from threading import Barrier, Thread
        self._open()
        barrier = Barrier(2)
        outcomes = []

        def race(name):
            barrier.wait()
            try:
                outcomes.append(("ok", name, self.workflow.plan_cycles(
                    name, t(7), expected_version=1).version))
            except VersionConflict:
                outcomes.append(("vc", name, None))

        threads = [Thread(target=race, args=(f"op-{i}",)) for i in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # Exactly one planner wins the compare-and-set at version 1.
        self.assertEqual(sorted(o[0] for o in outcomes), ["ok", "vc"])
        self.assertEqual(self.workflow.version, 2)
        self.assertTrue(self.workflow.stream.verify_chain())

    def test_event_replay_rebuilds_board_and_decisions(self):
        self._open()
        self.workflow.plan_cycles("op-plan", t(7))
        repair = RepairOrder("R1", "L1", "A", t(15), t(17), frozenset({"defect"}))
        grounds = [Evidence("defect", t(14), timedelta(days=1), True)]
        self.workflow.insert_repair("op-keep", repair, grounds,
                                    POLICY_PRESERVE_FEASIBLE, t(14))
        rolled = RepairOrder("R2", "L2", "A", t(15), t(17), frozenset({"defect"}))
        self.workflow.insert_repair("op-roll", rolled, grounds,
                                    POLICY_ROLLBACK_ALL, t(14))

        fresh = LocoDispatchWorkflow(stream=self.workflow.stream,
                                     trail=self.workflow.trail,
                                     store=self.workflow.store)
        report = fresh.replay()
        self.assertIsNotNone(report.board)
        # Only R1 survived: R2 was rejected and never entered the snapshot.
        self.assertEqual([r.repair_id for r in report.board.repairs], ["R1"])
        current = {a.move_id: a.loco_ids for a in report.board.assignments}
        self.assertNotIn("L1", current["M3"])
        # The rejected decision is present in the replay history.
        topics = [e.topic for e in report.decisions]
        self.assertIn(TOPIC_REPAIR_INSERTED, topics)
        self.assertIn(TOPIC_REPAIR_REJECTED, topics)
        self.assertEqual(report.last_version, 3)

    def test_service_exposes_dispatch_on_shared_control_plane(self):
        service = RailForgeService()
        wf = service.loco_dispatch("region-north")
        self.assertIs(wf, service.loco_dispatch("region-north"))
        self.assertIsNotNone(service.loco_dispatch("region-south"))
        self.assertIs(wf.stream, service.workflow().stream)
        self._open(wf)
        wf.plan_cycles("op-plan", t(7))
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())


if __name__ == "__main__":
    unittest.main()
