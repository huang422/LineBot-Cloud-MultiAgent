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

- Jobs are still registered at startup, but only so `/health/detail` can list
  their ids and next runs (never the message text).
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
- The same request also carries time-based maintenance, since the app has no
  timers of its own (ADR-0006). After dispatching scheduled messages,
  `/internal/cron` calls `run_cron_tick` (`src/pipeline/maintenance.py`),
  which flushes passive group buffers whose oldest message is older than
  `PASSIVE_GROUP_FLUSH_DELAY_SECONDS`, queues the daily model end-of-life
  check (an `eol` task, ADR-0007) and runs the Colab controller's tick
  (ADR-0010). It is bounded (50 chats; no new chat
  is started after 30 s, and a flush is never cancelled half-way) and runs
  even when scheduled messages are disabled, which is why the job exists
  whenever `INTERNAL_CRON_SECRET` is set. Its summary (`passive_flushed`,
  `maintenance_errors`) is merged into the JSON response.
