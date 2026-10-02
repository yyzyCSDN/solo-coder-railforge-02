import unittest
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.maintenance.release import Evidence, ReleasePolicy
from railforge.planning.cycle_repair import (PRESERVE_FEASIBLE, ROLLBACK, CycleLeg, CycleUnit,
                                             RepairConflict, RepairOrder, UnitWindow)

U = timezone.utc
POLICY = ReleasePolicy(frozenset({"daily"}))


def dt(hour, minute=0):
    return datetime(2026, 3, 1, hour, minute, tzinfo=U)


def units(*ids):
    return [CycleUnit(uid, "A", dt(6), 400.0, 500.0, 200.0) for uid in ids]


def evidence(*ids):
    return {uid: [Evidence("daily", dt(5), timedelta(hours=72), True)] for uid in ids}


def legs():
    return [
        CycleLeg("L1", "T1", "A", "B", dt(8), dt(10), 1000.0, 0.0, 0.0),
        CycleLeg("L2", "T2", "B", "A", dt(11), dt(13), 1000.0, 0.0, 0.0),
        CycleLeg("L3", "T3", "A", "B", dt(14), dt(16), 1000.0, 0.0, 0.0),
    ]


def order(order_id, unit_id, start, end, location="A"):
    return RepairOrder(order_id, unit_id, location, start, end,
                       frozenset({"post"}),
                       (Evidence("post", start, timedelta(hours=72), True),))


class CycleRepairTests(unittest.TestCase):
    def setUp(self):
        self.service = RailForgeService()
        self.features = self.service.features

    def commit(self, op="op-cycle", cycle="C1", **overrides):
        args = dict(legs=legs(), units=units("U1", "U2"), turn_minutes=60,
                    adhesion_factor=0.5, evidence=evidence("U1", "U2"),
                    policy=POLICY, existing=(), at=dt(4))
        args.update(overrides)
        return self.features.commit_loco_cycle(op, cycle, **args)

    def test_commit_validates_everything_in_one_version(self):
        receipt = self.commit()
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.version, 1)
        self.assertEqual(receipt.result["assignments"],
                         {"L1": ["U1"], "L2": ["U1"], "L3": ["U2"]})
        self.assertTrue(self.service.workflow().stream.verify_chain())
        self.assertTrue(self.service.workflow().trail.verify())

    def test_commit_is_idempotent_per_operation(self):
        first = self.commit()
        head = self.service.workflow().stream.head
        again = self.commit()
        self.assertIs(first, again)
        self.assertEqual(self.service.workflow().stream.head, head)

    def test_traction_shortage_fails_closed(self):
        heavy = [CycleLeg("L1", "T1", "A", "B", dt(8), dt(10), 100000.0, 5.0, 0.0)]
        with self.assertRaises(ValueError):
            self.commit(legs=heavy)
        self.assertIsNone(self.service.control.store.read("C1"))

    def test_missing_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            self.commit(evidence={})
        self.assertIsNone(self.service.control.store.read("C1"))

    def test_existing_plan_windows_are_respected(self):
        receipt = self.commit(existing=[UnitWindow("U1", dt(7), dt(9), "A")])
        self.assertEqual(receipt.result["assignments"]["L1"], ["U2"])
        self.assertEqual(receipt.result["assignments"]["L2"], ["U2"])

    def test_recommit_is_rejected(self):
        self.commit()
        with self.assertRaises(ValueError):
            self.commit(op="op-cycle-2")

    def test_repair_without_conflict_commits(self):
        self.commit()
        receipt = self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(18), dt(20)), PRESERVE_FEASIBLE, dt(17))
        self.assertEqual(receipt.version, 2)
        self.assertEqual(receipt.result["impacted"], [])
        self.assertFalse(receipt.result["duplicate"])
        self.assertEqual(len(receipt.result["state"]["repairs"]), 1)

    def test_repair_conflict_preserve_feasible_reassigns(self):
        self.commit()
        receipt = self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(9), dt(12)), PRESERVE_FEASIBLE, dt(7))
        self.assertEqual(receipt.result["impacted"], ["L1", "L2"])
        self.assertEqual(receipt.result["reassigned"], {"L1": ["U2"], "L2": ["U2"]})
        self.assertEqual(receipt.result["released"], [])
        state = receipt.result["state"]
        self.assertEqual(state["assignments"]["L3"], ["U2"])

    def test_repair_conflict_preserve_releases_uncoverable_legs(self):
        self.commit(units=units("U1"), evidence=evidence("U1"))
        receipt = self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(9), dt(12)), PRESERVE_FEASIBLE, dt(7))
        self.assertEqual(receipt.result["impacted"], ["L1", "L2"])
        self.assertEqual(receipt.result["reassigned"], {})
        self.assertEqual(receipt.result["released"], ["L1", "L2"])
        self.assertEqual(sorted(receipt.result["state"]["released"]), ["L1", "L2"])

    def test_repair_conflict_rollback_leaves_state_untouched(self):
        committed = self.commit(units=units("U1"), evidence=evidence("U1"))
        head = self.service.workflow().stream.head
        with self.assertRaises(RepairConflict) as ctx:
            self.features.insert_unplanned_repair(
                "op-repair", "C1", order("R1", "U1", dt(9), dt(12)), ROLLBACK, dt(7))
        self.assertEqual(ctx.exception.impacted, ("L1", "L2"))
        row = self.service.control.store.read("C1")
        self.assertEqual(row.version, 1)
        self.assertEqual(self.service.workflow().stream.head, head)
        # a failed commit stores no receipt, so the same operation id may retry
        retry = self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(9), dt(12)), PRESERVE_FEASIBLE, dt(7))
        self.assertEqual(retry.result["released"], ["L1", "L2"])
        self.assertEqual(committed.version, 1)

    def test_duplicate_repair_order_is_a_noop(self):
        self.commit()
        self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(18), dt(20)), PRESERVE_FEASIBLE, dt(17))
        receipt = self.features.insert_unplanned_repair(
            "op-repair-2", "C1", order("R1", "U1", dt(18), dt(20)), PRESERVE_FEASIBLE, dt(17, 30))
        self.assertTrue(receipt.result["duplicate"])
        self.assertEqual(len(receipt.result["state"]["repairs"]), 1)

    def test_repair_may_not_overlap_existing_plan(self):
        self.commit(existing=[UnitWindow("U1", dt(9), dt(12), "A")])
        with self.assertRaises(ValueError):
            self.features.insert_unplanned_repair(
                "op-repair", "C1", order("R1", "U1", dt(10), dt(11)), PRESERVE_FEASIBLE, dt(7))

    def test_repair_requires_release_evidence(self):
        self.commit()
        bad = RepairOrder("R1", "U1", "A", dt(18), dt(20), frozenset({"post"}), ())
        with self.assertRaises(ValueError):
            self.features.insert_unplanned_repair("op-repair", "C1", bad, PRESERVE_FEASIBLE, dt(17))

    def test_repair_cascade_beyond_direct_overlap(self):
        cascade_legs = [
            CycleLeg("L1", "T1", "A", "B", dt(8), dt(9), 1000.0, 0.0, 0.0),
            CycleLeg("L2", "T2", "B", "A", dt(10), dt(11), 1000.0, 0.0, 0.0),
            CycleLeg("L3", "T3", "A", "B", dt(13), dt(14), 1000.0, 0.0, 0.0),
        ]
        self.commit(legs=cascade_legs, units=units("U1"), evidence=evidence("U1"),
                    turn_minutes=30)
        # no leg overlaps [11:30, 12:00), but the unit is left at C afterwards
        receipt = self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(11, 30), dt(12), location="C"),
            PRESERVE_FEASIBLE, dt(7))
        self.assertEqual(receipt.result["impacted"], ["L3"])
        self.assertEqual(receipt.result["released"], ["L3"])

    def test_cross_cycle_version_boundary(self):
        self.commit(legs=[legs()[0]], units=units("U1"), evidence=evidence("U1"))
        clash = [CycleLeg("M1", "T9", "B", "A", dt(9, 30), dt(11), 1000.0, 0.0, 0.0)]
        with self.assertRaises(ValueError):
            self.commit(op="op-c2", cycle="C2", legs=clash,
                        units=units("U1"), evidence=evidence("U1"))
        ok = [CycleLeg("M2", "T9", "B", "A", dt(11, 30), dt(12, 30), 1000.0, 0.0, 0.0)]
        receipt = self.commit(op="op-c2", cycle="C2", legs=ok,
                              units=units("U1"), evidence=evidence("U1"))
        self.assertEqual(receipt.result["assignments"], {"M2": ["U1"]})

    def test_replay_reconstructs_committed_state(self):
        self.commit()
        self.features.insert_unplanned_repair(
            "op-repair", "C1", order("R1", "U1", dt(9), dt(12)), PRESERVE_FEASIBLE, dt(7))
        replayed = self.features.replay_loco_cycle("C1")
        stored = self.service.control.store.read("C1").value["state"]
        self.assertEqual(replayed, stored)
        self.assertEqual(replayed["assignments"]["L1"], ["U2"])
        self.assertIsNone(self.features.replay_loco_cycle("unknown"))

    def test_unknown_cycle_repair_raises(self):
        with self.assertRaises(ValueError):
            self.features.insert_unplanned_repair(
                "op-repair", "missing", order("R1", "U1", dt(9), dt(10)), PRESERVE_FEASIBLE, dt(7))


if __name__ == "__main__":
    unittest.main()
