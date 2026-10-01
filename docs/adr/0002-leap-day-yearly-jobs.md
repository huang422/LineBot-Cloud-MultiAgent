# ADR-0002: Yearly 02-29 jobs clamp to 02-28

**Status**: Accepted

## Context

`ScheduledYearlyMessage` accepts any `day` from 1–31 for any month, so a
`02-29` birthday reminder is a legal and entirely ordinary configuration.

`_YearlyJob.next_run` handled the leap-day rollover by retrying at
`ref.year + 1`, but never checked whether that year was itself a leap year.
For most reference dates the retry landed on another non-leap year and raised
`ValueError` **outside** the handler.

The blast radius was a full outage, not a cosmetic error: `next_run` is reached
from `list_jobs()` → `get_stats()`, which is called both by `/health` and by
`lifespan` during startup. A single `02-29` job therefore prevented the
application from booting, and — because the deploy script gates on `/health` —
blocked deployment too. Roughly three years in four.

## Decision

Resolve the target day per candidate year and clamp it to that year's real
calendar (`_effective_day`). A `02-29` job resolves to `02-29` in leap years
and `02-28` otherwise.

The clamp applies to **both** `next_run` and `matches`. Clamping only
`next_run` would stop the crash but leave the reminder silently never firing in
non-leap years, which is the same bug wearing a quieter hat.

## Alternatives rejected

- **Fire only in leap years.** Literal-minded, but a birthday reminder that
  appears every fourth year reads as broken.
- **Roll forward to 03-01.** Used by some calendar systems, but unusual for
  birthday-style reminders.
- **Reject 02-29 at config validation.** Prevents the crash without delivering
  the feature.

## Consequences

- Non-leap years fire one day early rather than not at all.
- `matches` is year-sensitive: the same job matches `02-28` in 2027 and
  `02-29` in 2028.
- Covered by leap-cycle tests in `tests/test_api_integration.py`.
