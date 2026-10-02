# RailForge Public Benchmark

A dependency-free freight railway operations benchmark with a public smoke suite. Private task prompts, hidden regressions and broken snapshots are deliberately excluded from this package.

## Locomotive dispatch workflow

`RailForgeService().loco_dispatch(region)` exposes a versioned board for
locomotive cycles and unscheduled running-repair insertion. Each commit is one
compare-and-set version boundary (`core.versioning.VersionConflict` on stale
captures), recorded as a hash-chained decision event plus an audit record.

A repair order never silently invalidates prior cycles. Choose the conflict
policy explicitly:

- `preserve_feasible` — locked train consists stay fixed; unlocked cycles may be
  re-powered, and every re-powering is returned as an explicit `Change`. If no
  feasible board exists the insertion is rejected and the prior board is kept.
- `rollback_all` — any conflict rejects the order outright and leaves version
  and assignments untouched.

Rejections are committed decision events too, so the full history replays via
`workflow.replay()`, and retrying an `operation_id` with identical arguments is
idempotent (different arguments with the same id raise).
