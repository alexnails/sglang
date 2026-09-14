#!/usr/bin/env python3
"""Fast branch restore: direct HTTPS + worker pool + shared token-bucket limiter.

Replaces the `gh` subprocess-per-branch approach (~3.0s/branch) with pooled
connections paced to GitHub's secondary limit (900 points/min REST; POST = 5
points => 180 POST/min => 3.0/sec).

Same safety properties as restore.py:
  - POST /git/refs never moves an existing ref (422 -> logged 'exists')
  - resumable via restore_log.jsonl
  - shares the log with restore.py, so the 240 already settled are skipped
"""
import json, os, subprocess, sys, threading, time
import urllib.request, urllib.error

REPO = "sgl-project/sglang"
SP = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(SP, "restore.tsv")
LOG = os.path.join(SP, "restore_log.jsonl")

WORKERS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--workers=")), 6))
RATE = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rate=")), 2.9))
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)

SETTLED = {"created", "exists", "object-missing"}
TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
if not TOKEN:
    sys.exit("could not read gh auth token")


class Limiter:
    """Token bucket shared by all workers; rate is adjustable at runtime."""

    def __init__(self, rate):
        self.rate = rate
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            if self.next_slot < now:
                self.next_slot = now
            wait = self.next_slot - now
            self.next_slot += 1.0 / self.rate
        if wait > 0:
            time.sleep(wait)

    def throttle(self, factor=0.5):
        with self.lock:
            self.rate = max(0.4, self.rate * factor)
            return self.rate


limiter = Limiter(RATE)
log_lock = threading.Lock()
tally_lock = threading.Lock()
tally = {}
counter = {"n": 0}
start = time.time()


def record(entry):
    with log_lock:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def post_ref(name, sha):
    body = json.dumps({"ref": "refs/heads/" + name, "sha": sha}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/git/refs", data=body, method="POST",
        headers={"Authorization": "Bearer " + TOKEN,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "Content-Type": "application/json",
                 "User-Agent": "branch-restore"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
            return "created", "", 0
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")[:300]
        low = txt.lower()
        retry_after = int(e.headers.get("Retry-After") or 0)
        if e.code == 422 and "already exists" in low:
            return "exists", txt, 0
        if e.code == 422:
            return "object-missing", txt, 0
        if e.code in (403, 429):
            return "ratelimited", txt, retry_after
        return "error", f"HTTP {e.code}: {txt}", 0
    except Exception as e:  # network hiccup -> retry
        return "ratelimited", f"{type(e).__name__}: {e}", 2


def work(item):
    name, sha = item
    for _ in range(8):
        limiter.acquire()
        status, detail, retry_after = post_ref(name, sha)
        if status == "ratelimited":
            newrate = limiter.throttle()
            sleep_s = retry_after if retry_after else 20
            print(f"  throttled; rate -> {newrate:.2f}/s, sleeping {sleep_s}s", flush=True)
            time.sleep(sleep_s)
            continue
        record({"name": name, "sha": sha, "status": status, "detail": detail,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        with tally_lock:
            tally[status] = tally.get(status, 0) + 1
            counter["n"] += 1
            n = counter["n"]
        if n % 100 == 0:
            el = time.time() - start
            print(f"[{n}] {dict(tally)}  {n/el:.2f}/s  elapsed={el:.0f}s", flush=True)
        return
    record({"name": name, "sha": sha, "status": "gave-up", "detail": "retries exhausted",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def main():
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("status") in SETTLED:
                    done.add(r["name"])

    rows = []
    with open(MANIFEST) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                n, s = line.split("\t", 1)
                rows.append((n, s))

    todo = [r for r in rows if r[0] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"manifest={len(rows)} settled={len(done)} todo={len(todo)} "
          f"workers={WORKERS} rate={RATE}/s eta={len(todo)/RATE/60:.1f}min", flush=True)

    threads = []
    queue = list(todo)
    qlock = threading.Lock()

    def runner():
        while True:
            with qlock:
                if not queue:
                    return
                item = queue.pop()
            work(item)

    for _ in range(WORKERS):
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    el = time.time() - start
    print(f"DONE {dict(tally)} in {el:.0f}s ({counter['n']/max(el,1):.2f}/s)", flush=True)


if __name__ == "__main__":
    main()
