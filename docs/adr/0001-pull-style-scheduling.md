# ADR-0001: Pull-style scheduling via Cloud Scheduler

**Status**: Accepted

## Context

Scheduled weekly/yearly group reminders were originally driven by APScheduler
inside the app process. On Cloud Run the instance is hibernated when idle and
scales to zero, so an in-process timer simply never fires. Reminders were
silently missed whenever no traffic happened to keep an instance warm.

## Decision

Scheduling is **pull-style**. `SchedulerService` holds the job definitions and
exposes `dispatch_due_jobs()`. An external Cloud Scheduler job hits
`POST /internal/cron` once per minute; that request wakes the instance and
dispatches whatever matches the current minute in `Asia/Taipei`.

`start()` / `shutdown()` remain as no-ops so existing callers and tests keep
working.

## Consequences

- Jobs are still registered at startup, but only so `/health` can list them.
- `/internal/cron` is authenticated with a shared secret in `X-Cron-Token`
  (`INTERNAL_CRON_SECRET`), compared with `hmac.compare_digest`. Empty secret
  means the endpoint refuses all traffic.
- Matching uses the `X-CloudScheduler-ScheduleTime` header rather than wall
  clock, so a slow cold start still dispatches the correct minute.
- Dedupe (`_recent_fires`) is **process-local** and cannot dedupe across
  instances. This is acceptable only because the Cloud Scheduler job is
  configured with `--max-retry-attempts=0`. Changing that setting without
  adding shared-state dedupe will cause duplicate sends.
- `gcloud scheduler jobs create` takes `--headers`; `update` requires
  `--update-headers`. Both are used in `scripts/deploy_cloud_run.sh`.
