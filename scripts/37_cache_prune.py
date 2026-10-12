#!/usr/bin/env python
"""
Prune the py21cmfast cache in data/cache -- dry run by default.

Only the TOP LEVEL *.h5 files of --cache are considered. Sub-directories
(convergence_boxsize_stitched, convergence_resolution_stitched, ...) and all
*.pkl / *.npz are never touched.

Three ways to select what to drop (combine as needed):
  --kinds BrightnessTemp          only consider these kinds
  --seeds 101-106                 only consider these random seeds (a-b ranges ok)
  --keep-log FILE [--keep-log F]  keep every file listed in these logs (from
                                  scripts/36_run_logging_h5.py); everything else
                                  in scope is dropped
Without --keep-log you MUST give --kinds or --seeds (so a bare run can never
select the whole cache).

Safety: with --keep-log, aborts unless at least --min-keep logged files exist
inside --cache (default 50), i.e. unless the log really looks like a full run.

Examples:
  python scripts/37_cache_prune.py --cache data/cache --kinds BrightnessTemp
  python scripts/37_cache_prune.py --cache data/cache --seeds 101-106
  python scripts/37_cache_prune.py --cache data/cache --keep-log a.txt --keep-log b.txt
Add --delete to actually remove (a manifest of removed files is written first).
"""
import argparse
import collections
import datetime
import os
import re
import sys

PAT = re.compile(r"^([A-Za-z]+)_([0-9a-f]{32})_r(\d+)\.h5$")


def parse_seeds(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def gb(x):
    return f"{x / 1e9:9.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--kinds", default=None, help="comma list, e.g. BrightnessTemp,InitialConditions")
    ap.add_argument("--seeds", default=None, help="e.g. 101-106 or 37,201")
    ap.add_argument("--keep-log", action="append", default=[])
    ap.add_argument("--min-keep", type=int, default=50)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()

    cache = os.path.realpath(a.cache)
    if not os.path.isdir(cache):
        sys.exit(f"not a directory: {cache}")
    if not a.keep_log and not (a.kinds or a.seeds):
        sys.exit("refusing: give --keep-log, or restrict with --kinds/--seeds")

    keep = set()
    for lf in a.keep_log:
        with open(lf) as fh:
            keep |= {os.path.realpath(l.strip()) for l in fh if l.strip()}
    keep_here = {p for p in keep if os.path.dirname(p) == cache and os.path.exists(p)}
    if a.keep_log:
        print(f"keep-logs: {len(keep)} files logged, {len(keep_here)} of them exist in {cache}")
        if len(keep_here) < a.min_keep:
            sys.exit(f"ABORT: only {len(keep_here)} logged files found in {cache} "
                     f"(< --min-keep {a.min_keep}). Wrong log, or the run did not finish.")

    kinds = set(a.kinds.split(",")) if a.kinds else None
    seeds = parse_seeds(a.seeds) if a.seeds else None

    drop, kept, skipped = [], [], []
    for name in sorted(os.listdir(cache)):
        p = os.path.join(cache, name)
        if not (name.endswith(".h5") and os.path.isfile(p)):
            continue
        m = PAT.match(name)
        if not m:
            skipped.append(name)
            continue
        kind, seed = m.group(1), int(m.group(3))
        if (kinds and kind not in kinds) or (seeds and seed not in seeds):
            kept.append((kind, seed, p)); continue
        if os.path.realpath(p) in keep:
            kept.append((kind, seed, p)); continue
        drop.append((kind, seed, p))

    tab = collections.defaultdict(lambda: [0, 0, 0, 0])
    for lst, i in ((kept, 0), (drop, 2)):
        for kind, seed, p in lst:
            t = tab[(kind, seed)]
            t[i] += 1
            t[i + 1] += os.path.getsize(p)
    print(f"\n{'kind':<18}{'seed':>5}{'keep n':>8}{'keep size':>14}{'DROP n':>8}{'DROP size':>14}")
    for (kind, seed), t in sorted(tab.items()):
        print(f"{kind:<18}{seed:>5}{t[0]:>8}{gb(t[1]):>14}{t[2]:>8}{gb(t[3]):>14}")
    tk = sum(os.path.getsize(p) for _, _, p in kept)
    td = sum(os.path.getsize(p) for _, _, p in drop)
    print(f"\nKEEP {len(kept)} files {gb(tk)}   DROP {len(drop)} files {gb(td)}"
          f"   (unrecognised names left alone: {len(skipped)})")

    if not a.delete:
        print("\nDRY RUN - nothing deleted. Re-run with --delete to remove the DROP files.")
        return
    man = os.path.join(cache, f"prune_manifest_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    with open(man, "w") as fh:
        fh.write("\n".join(p for _, _, p in drop) + "\n")
    print(f"manifest: {man}")
    n = 0
    for _, _, p in drop:
        try:
            os.remove(p); n += 1
        except OSError as e:
            print(f"  could not remove {p}: {e}")
    print(f"removed {n} files, freed ~{gb(td).strip()}")


if __name__ == "__main__":
    main()
