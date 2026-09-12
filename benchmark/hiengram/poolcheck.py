import re
import sys

WA = re.compile(r"Load weight end\..*?avail mem=([\d.]+) GB")
POOL = re.compile(r"Memory pool end\. avail mem=([\d.]+) GB")
MAXTOK = re.compile(r"max_total_num_tokens=(\d+)")
CACHE = re.compile(
    r"engram hbm cache layer (\d+): \d+ groups, \d+ rows \(([\d.]+) GiB\)"
)

for path in sys.argv[1:]:
    t = open(path, errors="replace").read()
    wa = [float(x) for x in WA.findall(t)]
    pool = [float(x) for x in POOL.findall(t)]
    tok = [int(x) for x in MAXTOK.findall(t)]
    cache = sum(float(g) for _, g in CACHE.findall(t))
    print("===", path.split("/")[-1])
    print(
        "  avail_after_weights: n=%d min=%s max=%s"
        % (len(wa), min(wa) if wa else None, max(wa) if wa else None)
    )
    print(
        "  avail_after_pool   : n=%d min=%s max=%s"
        % (len(pool), min(pool) if pool else None, max(pool) if pool else None)
    )
    print("  engram hbm cache GiB:", round(cache, 2))
    if wa and pool and tok:
        delta = min(wa) - min(pool)
        m = tok[-1] / 1e6
        print(
            "  delta(GB)=%.2f  max_tok=%.2fM  GB per M tok=%.4f" % (delta, m, delta / m)
        )
        print(
            "  delta minus cache=%.2f  -> GB per M tok=%.4f"
            % (delta - cache, (delta - cache) / m)
        )
    # does the cache get allocated before or after the pool line?
    ci = t.find("engram hbm cache layer")
    pi = t.find("Memory pool end.")
    wi = t.find("Load weight end.")
    print(
        "  order: weight_end@%d cache@%d pool_end@%d -> cache %s pool"
        % (wi, ci, pi, "BEFORE" if 0 <= ci < pi else ("AFTER" if ci > pi else "absent"))
    )
