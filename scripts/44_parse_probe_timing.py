"""
scripts/44_parse_probe_timing.py

Reads a probe run's output/log.txt (AMBER's own timing lines, format
"<ctime> <DDd HHh MMm SSs> : <stage>" from src/timing.f90) and:
  - sums per-shell wall time (MESH density/velocity field [+ interlace]
    + CMB power spectrum + CMB angular power spectrum -- the four stages
    inside cmbreion.f90's per-shell loop; map-making stages are excluded
    since --mapmake no was used)
  - extrapolates to a full config (default 28 shells, z=4.5-18.5 at
    czdel=0.5, AMBER's own default range) and to N_CONFIGS configs run
    ONE AT A TIME, so you can compare against a walltime budget whether
    run serially or (more realistically) as N_CONFIGS separate qsub jobs
    in parallel (in which case only the per-config number matters, not
    the total).

Usage:
  python scripts/44_parse_probe_timing.py runs/probe_2gpc_2048/output/log.txt \\
      --full-shells 28 --n-configs 16 --walltime-hours 72
"""
import argparse
import re

STAGE_RE = re.compile(
    r'(\d{2})d(\d{2})h(\d{2})m(\d{2})s\s*:\s*(.+?)\s*$')
SHELL_START_RE = re.compile(r'^\s*iz\s*:\s*(\d+)')

PER_SHELL_STAGES = {
    'MESH density field', 'MESH density interlace',
    'MESH velocity field', 'MESH velocity interlace',
    'CMB power spectrum', 'CMB angular power spectrum',
}


def parse(path):
    shells = []          # list of {stage: seconds}
    cur = None
    with open(path) as f:
        for line in f:
            m = SHELL_START_RE.match(line)
            if m:
                if cur:
                    shells.append(cur)
                cur = {}
                continue
            m = STAGE_RE.search(line)
            if m and cur is not None:
                d, h, mi, s, stage = m.groups()
                secs = int(d) * 86400 + int(h) * 3600 + int(mi) * 60 + int(s)
                if stage in PER_SHELL_STAGES:
                    cur[stage] = cur.get(stage, 0) + secs
    if cur:
        shells.append(cur)
    return shells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logfile')
    ap.add_argument('--full-shells', type=int, default=28,
                    help='shells in the real config (AMBER default '
                         'czmin=4.5,czmax=18.5,czdel=0.5 -> 28)')
    ap.add_argument('--n-configs', type=int, default=16)
    ap.add_argument('--walltime-hours', type=float, default=72.0)
    a = ap.parse_args()

    shells = parse(a.logfile)
    if not shells:
        raise SystemExit(f"no per-shell timing lines found in {a.logfile} -- "
                         "did amber.x actually start? check the log directly")

    print(f"parsed {len(shells)} completed shell(s) from {a.logfile}\n")
    totals = []
    for i, sh in enumerate(shells):
        t = sum(sh.values())
        totals.append(t)
        stagestr = ', '.join(f"{k}={v}s" for k, v in sh.items())
        print(f"  shell {i}: {t/60:6.1f} min   ({stagestr})")

    if len(shells) < a.full_shells:
        print(f"\n(only {len(shells)}/{a.full_shells} shells ran in this "
              f"probe -- that's expected, --czmax was narrowed on purpose)")

    avg = sum(totals) / len(totals)
    worst = max(totals)
    print(f"\nper-shell wall time: mean {avg/60:.1f} min, worst {worst/60:.1f} min "
          f"(N=2048 resolution)")

    for label, per_shell in [('mean', avg), ('worst-shell', worst)]:
        per_config_h = per_shell * a.full_shells / 3600
        total_h = per_config_h * a.n_configs
        print(f"\n[{label} per-shell] "
              f"1 config x {a.full_shells} shells ~ {per_config_h:.1f} h   "
              f"|   {a.n_configs} configs, ONE AT A TIME ~ {total_h:.1f} h")
        if per_config_h > a.walltime_hours:
            print(f"  -> EXCEEDS the {a.walltime_hours:.0f}h walltime budget "
                  f"for a SINGLE config -- would not finish even run alone")
        elif total_h > a.walltime_hours:
            print(f"  -> a single config fits in {a.walltime_hours:.0f}h, but "
                  f"all {a.n_configs} would need to run as separate concurrent "
                  f"jobs (not one after another) to meet the deadline")
        else:
            print(f"  -> fits within {a.walltime_hours:.0f}h even run serially")

    print("\n(this does not include field-generation stages that run once "
         "per config before the shell loop -- grf/lpt/esf/reion init -- "
         "check output/log.txt's 'MESH init'/'GRF'/'LPT'/'ESF'/'REION' "
         "timing lines for that one-time cost and add it to the per-config "
         "estimate above)")


if __name__ == '__main__':
    main()
