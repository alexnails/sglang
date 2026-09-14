#!/usr/bin/env python3
"""Recreate branches deleted by Jiminator in the 24h window, at their pre-deletion SHAs.

Safety properties:
  - Never overwrites: uses POST /git/refs, which 422s if the ref already exists.
    A branch someone re-created after the sweep is left alone.
  - Resumable: every outcome is appended to restore_log.jsonl; a re-run skips
    anything already settled.
  - Paced: base delay + backoff that honors GitHub's secondary rate limit.
"""
import json, os, subprocess, sys, time

REPO = "sgl-project/sglang"
SP = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(SP, "restore.tsv")
LOG = os.path.join(SP, "restore_log.jsonl")

DRY = "--dry-run" in sys.argv
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
DELAY = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--delay=")), 1.0)

SETTLED = {"created", "exists", "object-missing"}


def load_done():
    done = {}
    if os.path.exists(LOG):
        with open(LOG) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("status") in SETTLED:
                    done[r["name"]] = r["status"]
    return done


def record(entry):
    with open(LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def create_ref(name, sha):
    payload = json.dumps({"ref": "refs/heads/" + name, "sha": sha})
    p = subprocess.run(
        ["gh", "api", "-X", "POST", f"/repos/{REPO}/git/refs", "--input", "-"],
        input=payload, capture_output=True, text=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode == 0:
        return "created", ""
    low = out.lower()
    if "already exists" in low:
        return "exists", out.strip()[:200]
    if "not a valid" in low or "no such object" in low or "object does not exist" in low:
        return "object-missing", out.strip()[:200]
    if "rate limit" in low or "secondary" in low or "abuse" in low or "429" in low:
        return "ratelimited", out.strip()[:200]
    return "error", out.strip()[:200]


def main():
    rows = []
    with open(MANIFEST) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            name, sha = line.split("\t", 1)
            rows.append((name, sha))

    done = load_done()
    todo = [(n, s) for n, s in rows if n not in done]
    if LIMIT:
        todo = todo[:LIMIT]

    print(f"manifest={len(rows)} already_settled={len(done)} todo={len(todo)} dry_run={DRY}", flush=True)
    if DRY:
        for n, s in todo[:10]:
            print(f"  WOULD CREATE refs/heads/{n} -> {s}", flush=True)
        print(f"  ... ({len(todo)} total)", flush=True)
        return

    tally = {}
    backoff = 30.0
    i = 0
    while i < len(todo):
        name, sha = todo[i]
        status, detail = create_ref(name, sha)

        if status == "ratelimited":
            print(f"[{i}/{len(todo)}] rate limited; sleeping {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 900)
            continue  # retry same entry

        backoff = 30.0
        tally[status] = tally.get(status, 0) + 1
        record({"name": name, "sha": sha, "status": status, "detail": detail,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        i += 1
        if i % 25 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {tally}", flush=True)
        time.sleep(DELAY)

    print(f"DONE {tally}", flush=True)


if __name__ == "__main__":
    main()
