#!/usr/bin/env python3
"""
perf_compare.py — Compare experiment results against baseline in Performance.md

Parses the most recent subsection for a given experiment from Performance.md,
compares it row-by-row with a new report.tsv, and prints a markdown block
ready to append (subsection header + TSV table + verdict).

Exit codes:
  0  all metrics within thresholds
  1  at least one FAIL detected
  2  at least one WARN (but no FAIL)
"""

import argparse
import re
import sys
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────
QUERY_TIME_WARN_PCT = 5.0     # % increase → WARN
QUERY_TIME_FAIL_PCT = 10.0    # % increase → FAIL
ACCURACY_TOLERANCE = 0.0      # any change → FAIL
MEMORY_TOLERANCE_PCT = 0.0    # any change → WARN


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_tsv_rows(lines):
    """Parse TSV data lines into a dict keyed by subsection name.

    Columns are positional:
      [0] Subsection  [1] Query Time  [2] Accuracy  [-2] Memory  [-1] Building Time
    This works whether or not an extra metric column is present.
    """
    data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            data[name] = {
                'query_time': int(parts[1]),
                'accuracy':   float(parts[2]),
                'memory':     int(parts[-2]),
                'build_time': int(parts[-1]),
            }
        except (ValueError, IndexError):
            continue
    return data


def parse_performance_md(filepath, experiment):
    """Return (subsection_name, data_dict) for the *last* subsection of `experiment`."""
    text = Path(filepath).read_text()

    # Split on H2 headers, keeping the header text as the first line of each chunk.
    chunks = re.split(r'^## ', text, flags=re.MULTILINE)

    target = None
    for chunk in chunks:
        heading = chunk.split('\n', 1)[0].strip()
        if heading == experiment or heading == experiment.removesuffix('.toml'):
            target = chunk
            break

    if target is None:
        print(f"ERROR: experiment '{experiment}' not found in {filepath}", file=sys.stderr)
        sys.exit(1)

    # Split the section on H3 headers to get individual runs.
    subsections = re.split(r'^### ', target, flags=re.MULTILINE)
    if len(subsections) < 2:
        print(f"ERROR: no subsections (###) found for '{experiment}'", file=sys.stderr)
        sys.exit(1)

    last = subsections[-1]
    sub_name = last.split('\n', 1)[0].strip()

    # Collect data lines (skip header row, stop at non-data lines).
    data_lines = []
    header_seen = False
    for line in last.split('\n')[1:]:
        if line.startswith('Subsection'):
            header_seen = True
            continue
        if header_seen and line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[0].startswith('efs_'):
                data_lines.append(line)
            else:
                break

    return sub_name, parse_tsv_rows(data_lines)


def parse_report_tsv(filepath):
    """Return (raw_lines_including_header, data_dict) from a report.tsv."""
    raw = Path(filepath).read_text().strip().split('\n')
    return raw, parse_tsv_rows(raw[1:])   # first line is header


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(baseline, new):
    """Compare baseline vs new, return list of (subsection, verdict_string)."""
    verdicts = []

    for name, nv in new.items():
        issues = []

        if name not in baseline:
            verdicts.append((name, "NEW — no baseline to compare"))
            continue

        ov = baseline[name]

        # Query Time
        if ov['query_time'] > 0:
            qt_pct = ((nv['query_time'] - ov['query_time']) / ov['query_time']) * 100
        else:
            qt_pct = 0.0

        if qt_pct > QUERY_TIME_FAIL_PCT:
            issues.append(f"FAIL Query Time +{qt_pct:.1f}%")
        elif qt_pct > QUERY_TIME_WARN_PCT:
            issues.append(f"WARN Query Time +{qt_pct:.1f}%")
        elif qt_pct < -QUERY_TIME_WARN_PCT:
            issues.append(f"IMPROVED Query Time {qt_pct:.1f}%")

        # Accuracy
        acc_diff = nv['accuracy'] - ov['accuracy']
        if abs(acc_diff) > ACCURACY_TOLERANCE:
            issues.append(f"FAIL Accuracy {acc_diff:+.3f}")

        # Memory
        mem_diff = nv['memory'] - ov['memory']
        if mem_diff != 0:
            if ov['memory'] > 0:
                mem_pct = (mem_diff / ov['memory']) * 100
                issues.append(f"WARN Memory {mem_diff:+d} B ({mem_pct:+.2f}%)")
            else:
                issues.append(f"WARN Memory {mem_diff:+d} B")

        verdicts.append((name, "; ".join(issues) if issues else "OK"))

    return verdicts


# ── Output ────────────────────────────────────────────────────────────────────

def format_markdown(commit, date, report_lines, verdicts):
    """Build the markdown block to append to Performance.md."""
    out = []
    out.append(f"### {commit} ({date})")

    # Raw TSV (header + data) — preserves the original format exactly.
    for line in report_lines:
        out.append(line.rstrip())

    out.append("")

    # Verdict summary
    has_fail = any("FAIL" in v for _, v in verdicts)
    has_warn = any("WARN" in v for _, v in verdicts)
    has_improved = any("IMPROVED" in v for _, v in verdicts)

    if not has_fail and not has_warn and not has_improved:
        out.append("All metrics within acceptable thresholds.")
    else:
        for name, verdict in verdicts:
            if verdict != "OK":
                out.append(f"- **{name}**: {verdict}")

    out.append("")
    return "\n".join(out), has_fail, has_warn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results against Performance.md baseline"
    )
    parser.add_argument("--baseline",   required=True, help="Path to Performance.md")
    parser.add_argument("--new-report", required=True, help="Path to new report.tsv")
    parser.add_argument("--experiment", required=True,
                        help="Section name in Performance.md (e.g. 'dense_sift1m.toml')")
    parser.add_argument("--commit",     required=True, help="Short commit hash")
    parser.add_argument("--date",       required=True, help="Date (YYYY-MM-DD)")
    args = parser.parse_args()

    baseline_name, baseline_data = parse_performance_md(args.baseline, args.experiment)
    print(f"Baseline: '{baseline_name}' ({len(baseline_data)} rows)", file=sys.stderr)

    report_lines, new_data = parse_report_tsv(args.new_report)
    print(f"New report: {len(new_data)} rows", file=sys.stderr)

    verdicts = compare(baseline_data, new_data)

    markdown, has_fail, has_warn = format_markdown(
        args.commit, args.date, report_lines, verdicts
    )
    print(markdown)

    if has_fail:
        sys.exit(1)
    if has_warn:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
