import json
import sys

# acc lives on the point as acc_reported; utilisation and queue depth are
# arm-level (scraped from the server log), not per point.
for path in sys.argv[1:]:
    print("===", path)
    try:
        d = json.load(open(path))
    except Exception as e:
        print("  unreadable:", e)
        continue
    for name, a in d.get("arms", {}).items():
        print(
            "  %s  dataset=%s  peak_util=%s  peak_queue=%s  pool=%s GB  max_tok=%s"
            % (
                name,
                a.get("dataset"),
                a.get("peak_token_usage"),
                a.get("peak_queue_req"),
                a.get("kv_pool_gb"),
                a.get("max_total_num_tokens"),
            )
        )
        for p in a.get("points", []):
            print(
                "    c=%-5s out=%8.1f ttft50=%9.1f itl50=%6.3f acc=%.4f ok=%s"
                % (
                    p.get("concurrency"),
                    p.get("output_throughput", 0),
                    p.get("median_ttft_ms", 0),
                    p.get("median_itl_ms", 0),
                    p.get("acc_reported") or 0,
                    p.get("acc_within_tol"),
                )
            )
        if a.get("error"):
            print("    ERROR:", str(a["error"])[:200])
