# sgl-project/sglang — mass branch deletion + PR closure sweep, 2026-09-14

Audit record for the bulk branch deletion on `sgl-project/sglang` and the restore
performed in response. All timestamps UTC.

## What happened

Between **04:41:06Z and 04:46:31Z** on 2026-09-14 — a single window of 5m25s —
`Jiminator` (Jimmy Shong):

- deleted **1,596 branches**, leaving 7 alive in the whole repo
- closed **316 pull requests** unmerged, in the same window (04:41:08Z → 04:46:24Z)

The two actions interleave minute-for-minute, so this was one combined operation:
close the PR, delete its head branch.

## Why this does not look like routine stale-branch cleanup

| signal | value |
|---|---|
| PRs closed that were opened the same day | 4 |
| PRs closed that were opened within 7 days | 65 |
| PRs closed that were opened within 30 days | 125 |
| release branches deleted | `release/v0.5.11`, `release/v0.5.12` |
| closing comment / explanation left | none found |

Two examples, both authored by the project lead and both closed mid-CI:

- **#39347** "Allow CUDA VMM feature transport with the Rust frontend" —
  opened 02:55:54Z, closed 04:42:22Z (1h46m later). Author had just triggered CI.
- **#39328** "Fix first-token metadata and reused attention-layer indexing" —
  opened 00:34:37Z, closed 04:42:21Z. Same pattern.

316 closures spread over 90+ distinct authors, no explanation on any of them.

## Data sources

The GitHub **events** API (`/repos/:r/events`) caps at 300 records and reports only
176 of the deletions — it is not a reliable source here. The **activity** API
(`/repos/:r/activity?activity_type=branch_deletion`) is uncapped, returns 11,720
records back to 2024-01-08, and carries a `before` field holding each branch's
pre-deletion SHA. That field is what made restoration possible.

Two independent full paginations of the activity API returned identical results
(same record count, same oldest/newest, zero-line diff on the 1,596 name→SHA set).

## The restore

All 1,596 branches were recreated at their exact pre-deletion SHAs via
`POST /git/refs`, which fails rather than moves if a ref already exists — so
branches other people had re-created since the sweep were left untouched.

```
1,591  created
    5  already existed (re-created by others post-sweep)
    0  failures
```

Throughput: the first pass shelled out to `gh` per branch and managed 0.33/s
(3.0s per branch, almost entirely process spawn). Rewritten against the HTTPS API
with a 6-worker pool and a shared token bucket at 2.9/s — GitHub's secondary limit
is 900 points/min for REST and a POST costs 5 points, so 3.0/s is the ceiling —
it sustained 2.89/s with zero throttle events. 8.6x.

**The PRs were not reopened.** Only branches were restored, so the 316 closed PRs
remain closed.

## Contested during restore

`ch-wan` deleted 7 of the restored branches between 05:41:22Z and 05:48:33Z, each
within minutes of its recreation, mostly their own `cheng/*` refs. Six are gone
again; `cheng/refactor/attn-01-rename-static-buffers` is live only because it was
recreated after that deletion. Not re-restored — see the log for exact names.

## Files

| file | contents |
|---|---|
| `restore.tsv` | manifest: 1,596 `branch-name<TAB>pre-deletion-SHA` |
| `restore_log.jsonl` | per-branch outcome, timestamped — the undo list |
| `restore.py` | original `gh`-subprocess restore (slow, kept for provenance) |
| `restore_fast.py` | pooled-HTTPS restore, resumable, rate-limit aware |
| `pr_closers.py` | pages closed PRs and attributes each close to an actor |
| `pr_closers.json` | 363 PRs closed in the 24h window with closer attribution |

To roll the restore back, every branch to remove is a `"status":"created"` row in
`restore_log.jsonl`.
