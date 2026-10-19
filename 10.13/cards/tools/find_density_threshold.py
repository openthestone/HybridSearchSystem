"""Find the DENSITY_THRESHOLD that shrinks the index by a target fraction.

The threshold cannot be computed from an index's K histogram. Its buckets are powers of two and
on the corpora seen so far both the index mass and every posting the queries touch sit inside one
of them, so interpolating within it is doing all the work exactly where it has no resolution --
measured 3x wrong on frnew_syn_v2 (-24.9% predicted against -7.9% built, -36.4% against -18.8%).
tools/recommend_density_threshold.py ranks the bucket edges; this places a point between them, by
building and measuring.

Secant rather than bisection: index size against threshold is close to linear over a range this
narrow (50-60 MB per 1e-6 measured), so two points aim the third within a point or two of target.
Indexes are 25-37GB, so every candidate that is not the answer is deleted again.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

MB = 1024 * 1024
SPARSE_PACKED = 4   # PostingLayout::SPARSE_PACKED, the index into [BITSTAT-INDEX]'s byType


def exact_from_hitcum(log_path, seg, base_mb, bytes_per_unit=None):
    """Index size at every reachable threshold, read off [BITSTAT-INDEX]'s hitCum.

    SelectLayout compares hitCount/docNumPerSegment against the threshold and hitCount is an
    integer, so a threshold only matters through floor(t * docNumPerSegment): the reachable index
    sizes are one per integer h, and a target between two of them does not exist. hitCum[h] is the
    number of postings with hitCount <= h, which is exactly the number a threshold just above
    h/docNumPerSegment compresses.

    Each compressed posting stops costing a docNumPerSegment/8 bitmap and starts costing
    8 + bytes_per_unit * K, where K is its own occupied-unit count. K is not quite the posting's
    hit count -- two hits can land in one 16-doc unit -- but it is never larger, and at the small
    hit counts a 20-30% target lives at the gap is a few bytes against a 16KB saving.

    Charge each posting ITS OWN hit count, not the threshold's. Charging every posting in the
    bucket 8 + bytes_per_unit * h prices a 1-hit posting at 5248 bytes when the threshold is 1310,
    and the error grows with h exactly as fast as the answer does: measured +6.2% residual at
    h<=131 against +167.6% at h<=1310 on the 10M corpus.

    Returns (rows, cum_all, bytes_per_unit).
    """
    cums, by_type = [], [0] * 5
    with open(log_path, "r", errors="replace") as fh:
        for row in fh:
            if row.startswith("[BITSTAT-INDEX]") and "hitCum=[" in row:
                cum = [int(x) for x in re.search(r"hitCum=\[([0-9,]*)\]", row).group(1).split(",") if x]
                if cum:
                    cums.append(cum)
                # byType entry 4 is SPARSE_PACKED (4 bytes per unit), entry 1 SPARSE_BITMAP (6).
                bt = re.search(r"byType=\[([0-9,]*)\]", row)
                if bt:
                    for i, v in enumerate(bt.group(1).split(",")[:5]):
                        by_type[i] += int(v or 0)
    if not cums:
        return None
    # Sum every [BITSTAT-INDEX] line (one per PostingFieldData) and divide saved bytes by the
    # measured baseline -- one line's own all-dense total is not the index. hitCum is truncated at
    # the last non-zero bin, so a short line is extended with its own last value, not dropped.
    width = max(len(c) for c in cums)
    cum_all = [sum(c[min(h, len(c) - 1)] for c in cums) for h in range(width)]
    if bytes_per_unit is None:
        bytes_per_unit = 6
        for t, n in enumerate(by_type):
            if t == int(SPARSE_PACKED) and n > 0:
                bytes_per_unit = 4
    bitmap = seg / 8.0
    rows = []
    saved, prev = 0.0, 0
    for h, n in enumerate(cum_all):
        saved += (n - prev) * (bitmap - (8 + bytes_per_unit * h))
        prev = n
        rows.append((h, base_mb - saved / MB))
    return rows, cum_all, bytes_per_unit


def du_mb(path):
    if not os.path.isdir(path):
        return None
    out = subprocess.run(["du", "-sm", path], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return int(out.stdout.split()[0])


def free_mb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize // MB


def scan_existing(runs_dir, seg, quiet=False):
    """Thresholds already built under runs_dir, as free data points.

    Directory names carry the threshold: den0.01, k0.00021, seg131072_den0. A name that pins a
    different DOC_NUM_PER_SEGMENT is skipped -- those indexes are not comparable.
    """
    found = {}
    if not os.path.isdir(runs_dir):
        return found
    for name in sorted(os.listdir(runs_dir)):
        seg_tag = re.search(r"seg(\d+)", name)
        if seg_tag and int(seg_tag.group(1)) != seg:
            continue
        # Exponent form too: run.sh puts 1.9e-05 straight into the directory name, and matching
        # only the mantissa would read den1.90735e-05 as t=1.90735.
        m = re.search(r"(?:den|k)(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", name)
        if not m:
            continue
        idx = os.path.join(runs_dir, name, "work", "index")
        mb = du_mb(idx)
        if mb is None:
            continue
        t = float(m.group(1))
        if t not in found:
            found[t] = (mb, idx)
        elif found[t] is not None and found[t][0] != mb:
            # Same threshold, two sizes -> one of them is not what its name says. None marks it.
            found[t] = None
    for t, v in list(found.items()):
        if v is None:
            if not quiet:
                print("  SKIP  t=%-10g -- built twice at different sizes under this runs dir" % t)
            del found[t]
    return found


def build(args, t, work_dir):
    env = dict(os.environ)
    env.update({
        "RAPIDJSON_INCLUDE_DIR": "/usr/include",
        "DATASET_FILE": args.dataset,
        "DEVICE_IDS": args.devices,
        "DOCS": str(args.docs),
        "DENSITY_THRESHOLD": repr(t),
        "DOC_NUM_PER_SEGMENT": str(args.seg),
        "WORK_DIR": work_dir,
        "RESULT_DIR": work_dir + "_result",
        "QUERY_OUT_DIR": args.query_dir,
        "NUM_QUERIES": "0",
        "SHARD_GROUP_SIZE": str(args.shard_group_size),
        "PARALLEL_LOAD": "1",
    })
    cmd = ["./run.sh", "--compile", "0", "--convert-query", "0",
           "--convert-data", "1", "--build-index", "1", "--search", "0"]
    print("    DENSITY_THRESHOLD=%r %s" % (t, " ".join(cmd)), flush=True)
    log = os.path.join(args.runs_dir, "find_t_%g.log" % t)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=args.root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print("    build failed (rc=%d), see %s" % (rc, log), file=sys.stderr)
        return None
    shutil.rmtree(os.path.join(work_dir, "builder_input"), ignore_errors=True)
    return du_mb(os.path.join(work_dir, "index"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="checkout holding run.sh (only needed to build)")
    ap.add_argument("--dataset", help="only needed to build")
    ap.add_argument("--query-dir", help="only needed to build")
    ap.add_argument("--runs-dir", help="where candidate WORK_DIRs go (only needed to build)")
    ap.add_argument("--baseline-index", help="the DENSITY_THRESHOLD=0 index dir, measured with du")
    ap.add_argument("--baseline-mb", type=int, help="use instead of --baseline-index")
    ap.add_argument("--target", type=float, default=0.20, help="smallest acceptable index reduction")
    ap.add_argument("--target-max", type=float,
                    help="largest acceptable reduction. Reachable reductions are discrete -- one per"
                         " integer hit count -- so a band is usually the honest way to ask, and every"
                         " point inside it is listed.")
    ap.add_argument("--tolerance", type=float, default=0.01, help="accept target +- this, in fraction")
    ap.add_argument("--start", type=float, help="first candidate (default: from the scan, else 0.0002)")
    ap.add_argument("--second", type=float, help="second candidate, to aim the secant")
    ap.add_argument("--max-builds", type=int, default=4)
    ap.add_argument("--docs", type=int, default=10000000)
    ap.add_argument("--seg", type=int, default=131072)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--shard-group-size", type=int, default=2)
    ap.add_argument("--bitstat-log",
                    help="a log holding [BITSTAT-INDEX] with hitCum (BITLIST_STATS=1). With it the "
                         "answer is computed and nothing is built.")
    ap.add_argument("--bytes-per-unit", type=int,
                    help="sparse posting bytes per occupied unit: 4 for SPARSE_PACKED, 6 for "
                         "SPARSE_BITMAP. Default: read off byType in the log.")
    ap.add_argument("--no-scan", action="store_true",
                    help="ignore indexes already under --runs-dir instead of using them as points")
    ap.add_argument("--keep-losers", action="store_true", help="do not delete candidates that miss")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show every self-check row and every index the scan skipped")
    args = ap.parse_args()

    if args.baseline_mb:
        base = args.baseline_mb
    elif args.baseline_index:
        base = du_mb(args.baseline_index)
        if base is None:
            sys.exit("cannot measure %s" % args.baseline_index)
    else:
        sys.exit("need --baseline-index or --baseline-mb")
    tmax = args.target_max if args.target_max is not None else args.target + 0.10
    if tmax < args.target:
        sys.exit("--target-max is below --target")
    target_mb = base * (1.0 - args.target)
    lo_mb = base * (1.0 - args.target - args.tolerance)
    hi_mb = base * (1.0 - args.target + args.tolerance)
    if args.bitstat_log:
        print("baseline %d MB at DENSITY_THRESHOLD=0, want -%.1f%%..-%.1f%%\n"
              % (base, args.target * 100, tmax * 100))
    else:
        print("baseline (DENSITY_THRESHOLD=0): %d MB" % base)
        print("target: %.1f%% -> %d MB, accept %d..%d MB\n" % (args.target * 100, target_mb, lo_mb, hi_mb))

    if not args.bitstat_log:
        missing = [n for n in ("root", "dataset", "query_dir", "runs_dir") if getattr(args, n) is None]
        if missing:
            sys.exit("building needs --%s (or pass --bitstat-log to compute instead)"
                     % ", --".join(m.replace("_", "-") for m in missing))
    if args.bitstat_log:
        got = exact_from_hitcum(args.bitstat_log, args.seg, base, args.bytes_per_unit)
        if got is None:
            rows = None
        else:
            rows, cum_all, bpu = got
            print("sparse postings priced at %d bytes per unit (%s)\n"
                  % (bpu, "given" if args.bytes_per_unit else "from byType"))
        if rows is None:
            sys.exit("no [BITSTAT-INDEX] with hitCum in %s -- rerun the search with BITLIST_STATS=1 "
                     "on a build that has it" % args.bitstat_log)
        # K stands in for hitCount in the size term, close at small h and not at large h, so only
        # indexes near the band are used to calibrate.
        measured, factor = {}, 1.0
        if args.runs_dir:
            disk = []
            for t, v in sorted(scan_existing(args.runs_dir, args.seg, quiet=not args.verbose).items()):
                if v is None or t <= 0:
                    continue
                hb = int(t * args.seg)
                if hb >= len(rows) or rows[hb][1] <= 0:
                    continue
                disk.append((hb, v[0]))
            # A higher threshold compresses a superset, so it cannot give a bigger index. A pair
            # that says otherwise was not built the same way.
            keep, prev = [], None
            for hb, mb in disk:
                if prev is not None and mb > prev[1]:
                    print("  SKIP  h<=%-6d %7d MB -- bigger than h<=%d (%d MB); different layout"
                          " or an unfinished build" % (hb, mb, prev[0], prev[1]))
                    continue
                keep.append((hb, mb))
                prev = (hb, mb)
            near = []
            for hb, mb in keep:
                measured[hb] = mb
                if args.target - 0.10 <= 1 - mb / float(base) <= tmax + 0.10:
                    near.append(mb / rows[hb][1])
            if near:
                near.sort()
                factor = near[len(near) // 2]
                if abs(factor - 1.0) > 0.001:
                    print("calibrated x%.4f from %d built index%s near the band\n"
                          % (factor, len(near), "" if len(near) == 1 else "es"))
        model = [mb for _h, mb in rows]
        rows = [(h, mb * factor) for h, mb in rows]
        for h in measured:
            rows[h] = (h, float(measured[h]))   # where it is built, use the disk
        print("  %-4s %-13s %-13s %9s %8s" % ("h <=", "t >", "t <=", "index MB", "vs den=0"))
        pts = [(h, mb, 1 - mb / base) for h, mb in rows if 1 - mb / base > 0]
        inside = [p for p in pts if args.target <= p[2] <= tmax]
        show = [p for p in pts if args.target - 0.04 <= p[2] <= tmax + 0.04]
        if len(show) > 14:
            # One row per 0.5 point, plus the band's first row, which is the answer.
            keep, last = [], None
            for p in show:
                if last is None or abs(p[2] - last) >= 0.005 or (inside and p[0] == inside[0][0]):
                    keep.append(p)
                    last = p[2]
            print("  (%d reachable rows, thinned to one per 0.5 point)" % len(show))
            show = keep
        for h, mb, red in show:
            if inside and h == inside[0][0]:
                tag = "   <- take this: least index saved, least latency paid"
            elif args.target <= red <= tmax:
                tag = ""
            else:
                tag = "   (outside)"
            print("  %-4d %-13.8f %-13.8f %9.0f %7.1f%%%s"
                  % (h, h / args.seg, (h + 1) / args.seg, mb, red * 100, tag))
        if not inside:
            sys.exit("\nnothing reachable between %.1f%% and %.1f%%. Reductions come one per integer"
                     " hit count, so widen the band." % (args.target * 100, tmax * 100))
        if measured:
            far = [h for h in sorted(measured)
                   if not (args.target - 0.10 <= 1 - measured[h] / float(base) <= tmax + 0.10)]
            print("\n%d built index%s used: %d near the band%s"
                  % (len(measured), "" if len(measured) == 1 else "es",
                     len(measured) - len(far),
                     ", %d far ones ignored (K stands in for hitCount and drifts there)" % len(far)
                     if far else ""))
            print("    %-8s %9s %9s %9s" % ("h <=", "on disk", "computed", "residual"))
            for h in sorted(measured):
                pred = model[h] * factor
                print("    %-8d %8d MB %8d MB %+8.1f%%"
                      % (h, measured[h], pred, 100.0 * (pred / measured[h] - 1.0)))
            # The size difference between two built indexes prices exactly the postings between
            # their thresholds, so the mean K it implies cannot exceed the upper threshold. This
            # caught the 10M corpus being encoded at 6 bytes per unit while this priced it at 4.
            bad, prev = [], None
            for h in sorted(measured):
                if prev is not None:
                    dn = cum_all[min(h, len(cum_all) - 1)] - cum_all[min(prev, len(cum_all) - 1)]
                    dbytes = (measured[prev] - measured[h]) * float(MB)
                    if dn > 0:
                        implied = (seg / 8.0 - 8 - dbytes / dn) / bpu
                        if implied > h or implied < 0:
                            bad.append((prev, h, implied))
                prev = h
            for lo, hi, implied in bad:
                print("    IMPOSSIBLE  h in (%d, %d] implies %.1f units per posting, ceiling %d"
                      % (lo, hi, implied, hi))
            if bad:
                print("    the cost model is wrong, not the histogram. Check byType, or set"
                      " --bytes-per-unit.")
        h, mb, red = inside[0]
        print("\nDENSITY_THRESHOLD=%.6g   (any t in (%.8f, %.8f] gives the same index)"
              % ((h + 0.5) / args.seg, h / args.seg, (h + 1) / args.seg))
        print("  index %d MB, -%.1f%%" % (mb, red * 100))
        print("  reachable reductions are one per integer hit count, so a target between two")
        print("  rows above cannot be built. Changing it needs --convert-data 1.")
        return

    # (threshold, index MB), cheapest first: anything already on disk costs nothing to measure.
    pts = []
    if not args.no_scan:
        prev = None
        for t, (mb, path) in sorted(scan_existing(args.runs_dir, args.seg).items()):
            if t <= 0:
                continue
            # A violation means these were not built the same way, so drop it and say so.
            if prev is not None and mb >= prev[1]:
                print("  SKIP  t=%-10g %7d MB  -- not smaller than t=%g (%d MB); different layout"
                      " or an unfinished build" % (t, mb, prev[0], prev[1]))
                continue
            pts.append((t, mb))
            prev = (t, mb)
            print("  found built: t=%-10g %7d MB  (%.1f%%)  %s" % (t, mb, 100 * (1 - mb / base), path))
        if pts:
            print()

    def hit(mb):
        return lo_mb <= mb <= hi_mb

    for t, mb in pts:
        if hit(mb):
            report(t, mb, base, "already built")
            return

    seeds = [x for x in (args.start, args.second) if x is not None]
    builds = 0
    while builds < args.max_builds:
        t = next_threshold(pts, target_mb, seeds)
        if t is None:
            print("no candidate left to try", file=sys.stderr)
            break
        # Only floor(t * seg) matters, so snap to the middle of that integer's band -- otherwise
        # the secant proposes a band it already has and rebuilds a byte-identical index.
        band = int(t * args.seg)
        t = round((band + 0.5) / args.seg, 12)
        if any(int(p[0] * args.seg) == band for p in pts):
            print("candidate lands on h <= %d, which is already built; the reachable reductions are"
                  " %.1f points apart here and the target falls between two of them"
                  % (band, 100.0 / max(1, band) * 0.8), file=sys.stderr)
            break
        work = os.path.join(args.runs_dir, "k%g" % t, "work")
        need = int(max((p[1] for p in pts), default=base) * 1.3)
        avail = free_mb(args.runs_dir)
        print("build %d: t=%g   (free %d MB, want >= %d)" % (builds + 1, t, avail, need))
        if avail < need and not args.dry_run:
            sys.exit("not enough free space; delete an index and retry")
        if args.dry_run:
            print("\n(dry run: stopping after the first candidate. Each build is a full reconvert,")
            print(" so the next one is only decided once this one has been measured.)")
            return
        mb = build(args, t, work)
        builds += 1
        if mb is None:
            break
        print("    -> %d MB (%.1f%%)\n" % (mb, 100 * (1 - mb / base)))
        if hit(mb):
            report(t, mb, base, "built")
            return
        pts.append((t, mb))
        pts.sort()
        if not args.keep_losers:
            shutil.rmtree(os.path.dirname(work), ignore_errors=True)
    if not args.dry_run:
        print("\ngave up after %d builds. Closest points:" % builds)
        for t, mb in sorted(pts, key=lambda p: abs(p[1] - target_mb))[:3]:
            print("  t=%-10g %7d MB  (%.1f%%)" % (t, mb, 100 * (1 - mb / base)))


def next_threshold(pts, target_mb, seeds):
    """Secant through the two points bracketing the target, or the two nearest to it."""
    if seeds:
        return seeds.pop(0)
    if not pts:
        return 0.0002
    if len(pts) == 1:
        t, mb = pts[0]
        return max(1e-6, t + (mb - target_mb) / 60.0 * 1e-6)
    below = [p for p in pts if p[1] > target_mb]      # too big an index -> threshold too low
    above = [p for p in pts if p[1] < target_mb]
    if below and above:
        p1, p2 = max(below), min(above)
    else:
        p1, p2 = sorted(pts, key=lambda p: abs(p[1] - target_mb))[:2]
    if p1[1] == p2[1]:
        return None
    t = p2[0] + (target_mb - p2[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
    # The monotonicity guard only catches a neighbour contradiction, so an index built in another
    # layout can still survive the scan. Check these two before trusting the answer.
    print("  interpolating between t=%g (%d MB) and t=%g (%d MB)" % (p1[0], p1[1], p2[0], p2[1]))
    return round(t, 9) if t > 0 else None


def report(t, mb, base, how):
    print("=" * 60)
    print("DENSITY_THRESHOLD=%g" % t)
    print("  index %d MB, %.1f%% under DENSITY_THRESHOLD=0 (%s)" % (mb, 100 * (1 - mb / base), how))
    print("=" * 60)
    print("It is an fr_converter argument, so reproducing this index needs --convert-data 1;")
    print("--build-index alone re-reads whatever builder_input already holds.")


if __name__ == "__main__":
    main()
