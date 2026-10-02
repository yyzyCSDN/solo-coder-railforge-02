from __future__ import annotations

from railforge.braking.brake_percentage import brake_percentage
from railforge.consist.integrity import validate
from railforge.routing.clearance import route_ok
from railforge.signaling.block_occupancy import conflicts
from railforge.workflows.control import RailForgeWorkflowControl
from railforge.workflows.features import RailForgeFeatureService
from railforge.workflows.loco_dispatch import LocoDispatchWorkflow
from railforge.capability_catalog import load_capabilities


class RailForgeService:
    def __init__(self, control: RailForgeWorkflowControl | None = None):
        self.control = control or RailForgeWorkflowControl()
        self.features = RailForgeFeatureService(self.control)
        self._dispatch_boards: dict[str, LocoDispatchWorkflow] = {}

    def loco_dispatch(self, subject: str = "default") -> LocoDispatchWorkflow:
        """Versioned loco-cycle / running-repair workflow on the shared plane."""
        board = self._dispatch_boards.get(subject)
        if board is None:
            board = LocoDispatchWorkflow(
                stream=self.control.stream, trail=self.control.trail,
                store=self.control.store, subject=f"loco-dispatch:{subject}")
            self._dispatch_boards[subject] = board
        return board

    def health(self):
        return {
            "service": "railforge",
            "status": "ok",
            "runtime_dependencies": 0,
            "capabilities": load_capabilities(),
            "workflow": {"events": self.control.stream.head, "audit_valid": self.control.trail.verify()},
        }

    def block_conflicts(self, candidate, existing, clearance):
        return conflicts(candidate, existing, clearance)

    def validate_consist(self, rows):
        return validate(rows)

    def brake_percentage(self, rows, gradient):
        return brake_percentage(rows, gradient)

    def route_clearance(self, profile, segments):
        return route_ok(profile, segments)

    def workflow(self):
        return self.control

