#!/usr/bin/env python3
"""Page through PRs closed in the last 24h and attribute each close to an actor."""
import json, subprocess, sys, os

SP = os.path.dirname(os.path.abspath(__file__))
SINCE = "2026-09-13T05:37:00Z"

Q = """
query($q:String!, $after:String) {
  search(query:$q, type:ISSUE, first:100, after:$after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title state merged createdAt closedAt
      author { login }
      timelineItems(itemTypes:[CLOSED_EVENT, REOPENED_EVENT], last:10) {
        nodes {
          __typename
          ... on ClosedEvent   { createdAt actor { login } }
          ... on ReopenedEvent { createdAt actor { login } }
        }
      }
    } }
  }
}
"""

def run(q, after=None):
    cmd = ["gh", "api", "graphql", "-f", "query=" + Q, "-F", "q=" + q]
    if after:
        cmd += ["-F", "after=" + after]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("graphql failed: " + p.stderr[:500])
    return json.loads(p.stdout)["data"]["search"]

rows = []
after = None
query = f"repo:sgl-project/sglang is:pr closed:>={SINCE}"
while True:
    d = run(query, after)
    for n in d["nodes"]:
        if not n:
            continue
        closers = [t for t in n["timelineItems"]["nodes"] if t and t["__typename"] == "ClosedEvent"]
        last = closers[-1] if closers else None
        rows.append({
            "number": n["number"], "title": n["title"], "state": n["state"],
            "merged": n["merged"], "author": (n["author"] or {}).get("login"),
            "createdAt": n["createdAt"], "closedAt": n["closedAt"],
            "closedBy": ((last or {}).get("actor") or {}).get("login"),
            "closedEventAt": (last or {}).get("createdAt"),
        })
    if not d["pageInfo"]["hasNextPage"]:
        break
    after = d["pageInfo"]["endCursor"]
    print(f"  ...{len(rows)}/{d['issueCount']}", flush=True)

with open(os.path.join(SP, "pr_closers.json"), "w") as fh:
    json.dump(rows, fh, indent=1)

print(f"total PRs closed in window: {len(rows)}")
by = {}
for r in rows:
    k = (r["closedBy"], "merged" if r["merged"] else "closed-unmerged")
    by[k] = by.get(k, 0) + 1
print("\nactor                 outcome            count")
for (actor, kind), c in sorted(by.items(), key=lambda x: -x[1]):
    print(f"{str(actor):22s}{kind:19s}{c}")
