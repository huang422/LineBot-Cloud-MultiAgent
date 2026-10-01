# ADR-0008: Secrets in one dotenv secret, least-privilege identities

**Status**: Accepted

## Context

Until v3 every deploy wrote all of `.env` into a generated env-vars file in
the repository root, which `gcloud builds submit` uploaded with the source.
`gs://<project>_cloudbuild/source` (US multi-region, 7-day soft delete)
collected 58 such tarballs; the newest also held
`.claude/settings.local.json`. The deploy then set every secret as a plain
Cloud Run environment variable, readable by anyone who can view the service.

One identity did everything: the Compute Engine default service account
(`<number>-compute@developer.gserviceaccount.com`) is the Cloud Run runtime
account, the project's default Cloud Build account and the stopped VM's
account, and it holds `roles/editor`. The default VPC still had
`default-allow-ssh`, `-rdp`, `-icmp`, `-http` and `-https` open to
`0.0.0.0/0`.

Phase 1 stopped the leak itself (the env file moved to `$TMPDIR`, and an
upload guard refuses secret files). This ADR is about where secrets live and
who can read them.

## Decision

### One dotenv secret, mounted as a file

- `.env` minus the deploy-only keys (`scripts/envfile.py DEPLOY_ONLY_KEYS`)
  becomes **one** Secret Manager secret, `linebot-dotenv`, with automatic
  replication (billed as one location; Cloud Run cannot use regional
  secrets). `python3 scripts/envfile.py to-dotenv` writes it: every value
  single-quoted, backslashes and quotes escaped, so python-dotenv reads back
  exactly what `.env` holds (multi-line JSON, `#`, quotes). It refuses, by key
  name, a value python-dotenv would change: one with `${` (python-dotenv
  expands it), a trailing backslash, or a carriage return, a quoted value
  python-dotenv cannot parse at all (a quote that never closes, or text after
  the closing quote: python-dotenv skips that key), and a blob over the
  64 KiB version limit. `envfile.py` reads `.env` with python-dotenv's own
  quoting rules, so a broken quote cannot swallow the keys after it, and its
  line rules: only CR LF, LF and CR end a line (U+2028 and the other breaks
  `str.splitlines()` knows stay in the value), and a leading byte order mark
  is dropped.
- Cloud Run mounts it at `/secrets/app/app.env`
  (`--set-secrets=/secrets/app/app.env=linebot-dotenv:N`), pinned to a
  version number so a revision rolls back with its own settings.
  `Settings.model_config["env_file"]` is `(".env", "/secrets/app/app.env")`:
  a later file wins, a missing file is skipped, and a real environment
  variable beats both. The deploy sets exactly one plain variable,
  `--set-env-vars=APP_VERSION=<tag>`. `--set-env-vars` replaces every plain
  variable, so the plain-text secrets of older revisions are gone and cannot
  shadow the file; it is never combined with `--env-vars-file` or
  `--clear-env-vars`.
- The deploy compares the SHA-256 of the local blob with the payload of
  `versions access latest` and adds a version only when `.env` changed.
  Neither the payload nor a hash is printed or logged: gcloud copies
  everything it prints into `~/.config/gcloud/logs` (kept about 30 days),
  so that read runs with `CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true`, like
  every call that reads or sets the Cloud Scheduler token. The blob only
  ever travels through a pipe, never a file. After a smoke check that
  passed, the deploy destroys every version that no remaining revision of
  any service in any region pins (it lists the regions with `gcloud run services
  list`, then reads each region's revisions) and that is not among the newest
  max(2, `DEPLOY_KEEP_REVISIONS`). Counting alone is not enough: a deploy
  that adds a version and then fails before Cloud Run creates a revision
  leaves a version nothing uses, and after the next change to `.env` the
  newest two would no longer include the previous revision's.
- Secret Manager's Always Free tier is **6 active versions** and 10,000
  accesses a month per billing account. Active means enabled *or disabled*:
  a disabled version is billed like an enabled one, and only a destroyed
  version is free. So old versions are destroyed, never disabled, and no
  secret is created with `--version-destroy-ttl` (delayed destruction keeps
  the version disabled, and billed, until the TTL ends). Budget: 3 live
  versions of `linebot-dotenv` (2, plus 1 during a deploy or after a deploy
  that failed before its revision or whose revision never became Ready; two
  such failures in a row reach 4, and the deploy warns above 3) and at
  most 3 of `linebot-colab-adc`, the Colab tier's token (ADR-0010: mounted
  only while `COLAB_ENABLED=true`, its unpinned versions destroyed the same
  way). The file is read at startup (Settings
  and the stale-key check), a couple of accesses per cold start.
- One blob rather than one secret per key: the bot has more than six
  secrets, so per-key secrets would exceed the free versions before the
  first rotation.

### Least-privilege identities

`scripts/provision_gcp.sh` (dry run by default) creates them:

| Identity | Roles |
| --- | --- |
| `linebot-runtime` (Cloud Run) | `roles/datastore.user` (project); `roles/storage.objectUser` on the media bucket only; `roles/iam.serviceAccountTokenCreator` on itself (V4 signed URLs through IAM Credentials); `roles/cloudtasks.enqueuer` on the queue; `roles/secretmanager.secretAccessor` on each secret it reads |
| `linebot-builder` (Cloud Build) | `roles/artifactregistry.writer` on the repository; `roles/logging.logWriter` (a user-specified build account must log somewhere; `cloudbuild.yaml` uses `CLOUD_LOGGING_ONLY`); `roles/storage.objectViewer` on the staging bucket |

- No logging or monitoring role for the runtime: stdout goes to Cloud
  Logging without one.
- The deploy itself still runs on the operator's machine, as the operator:
  the builder never deploys and never reads a secret.
- Builds stage their source in `gs://<project>-cb-src-usw1` (us-west1,
  Standard, uniform access, public access prevention, no soft delete,
  objects deleted after 3 days) instead of the multi-region default bucket.
  Bucket names are global and this one is predictable, so both scripts use
  it only when the project itself lists it (gcloud checks ownership of the
  default bucket, not of a `--gcs-source-staging-dir`); provisioning refuses
  a same-named bucket in another project rather than granting or adopting it.
  `purge-cloudbuild` clears soft delete on the old bucket *before* deleting
  `source/**` (clearing it does not affect objects already soft-deleted, so
  the other order would keep the secrets restorable for 7 days), then adds a
  1-day lifecycle rule. It needs `--yes-purge`.
- Until `linebot-builder`, the staging bucket, `linebot-runtime` and the
  secret exist, the deploy falls back to the old path and warns: build as
  the default account, deploy with a temporary env-vars file. Only a
  NOT_FOUND answer counts as "does not exist": any other lookup error
  (permission denied, API disabled, expired login, network) stops the deploy
  and the provisioning steps, so a failed read never downgrades a deploy or
  reports a rule as gone. And a service that already mounts `linebot-dotenv`
  is never moved back to plain variables, even when the secret or the
  account is really gone, unless the operator passes `--allow-env-fallback`.
- Removing `roles/editor` from the compute default account waits until
  nothing uses it (Phase 9, after a Policy Simulator check): the
  `drop-editor` step of `scripts/provision_gcp.sh`, run on its own, refuses
  while a replacement account is missing or anything still runs as the
  default account. That account is never deleted.

### Cloud Tasks keep a shared-secret header, not OIDC

The service must allow `allUsers` (LINE calls the webhook without Google
credentials), so Cloud Run would not verify an OIDC token on
`/internal/tasks/*` either; the app would have to verify it itself. OIDC
would also need `roles/iam.serviceAccountUser` on the runtime account for
both the runtime account (actAs) and the Cloud Tasks service agent. The
`X-Tasks-Token` header (`INTERNAL_TASKS_SECRET`, in the blob) needs neither
and is checked with `hmac.compare_digest` (ADR-0006).

### Network

Provisioning has no network step: the bot runs nothing on a VM, and the
VPC firewall does not sit in front of Cloud Run. The default VPC's
`default-allow-*` rules are left as they are; a VM added later must bring
its own rules. (Until 2026-10-01 a `firewall-pre` step replaced them with
SSH through IAP for the self-hosted search VM; it went with that VM, see
ADR-0009.)

## Consequences

- A secret no longer reaches Cloud Build, the image, or a plain environment
  variable; it exists in `.env`, in Secret Manager and in the running
  container. The secrets that sat in the old tarballs must still be rotated.
- Changing a setting means editing `.env` and deploying; the deploy adds the
  version. A rollback (`gcloud run services update-traffic
  --to-revisions=<previous>=100`) brings back that revision's own pinned
  version, which the deploy keeps. The Cloud Scheduler job's X-Cron-Token
  is not part of a revision: it is the latest deploy's
  INTERNAL_CRON_SECRET, so a rollback past a change of that secret needs
  the job's header reset (DEPLOY.md, 回滾).
- Secret Manager and the new service accounts cost nothing within the free
  tier; the budget alert (about 30 TWD, alerts only, no spend cap) is set
  in the console.
- Every kept revision keeps its secret version, so `DEPLOY_KEEP_REVISIONS`
  above 2 raises the number of billed versions toward (and past) the six
  free ones shared by all secrets.
- Once `ar-cleanup-enforce` has run, the Artifact Registry policy keeps only
  the 2 newest images and deletes the rest after a day, whatever revisions
  use them; Cloud Run keeps its own copy of an image only while a revision
  serves traffic. So rollback reaches back 2 images: raising
  `DEPLOY_KEEP_REVISIONS` or `DEPLOY_KEEP_IMAGES` above 2 does not add
  rollback room, and an image from a failed deploy or a build-only
  submission takes one of the two slots. The deploy warns when a kept
  revision's image falls outside them.
