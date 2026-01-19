#!/usr/bin/env python3
"""Extract best (fastest) grid-search configurations per metric cut.

For each metric cut (user-specified or auto-generated), selects the configuration
with the minimal query time among those achieving metric value >= cut.

Reads: grid_search_summary.tsv with columns:
    k_candidates, alpha, beta, ef_search, query_time_us, metric_value

Writes: TSV to stdout or file with best configuration per metric cut.

Usage:
    python scripts/extract_best_results.py [input_file]
    python scripts/extract_best_results.py --cuts 0.38 0.39 0.40
    python scripts/extract_best_results.py -o best_results.tsv
"""

import argparse
import csv
import sys
from typing import Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract fastest configuration per metric cut from grid search summary",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "input", 
        nargs="?", 
        default="grid_search_results/grid_search_summary.tsv",
        help="Input TSV file path (default: grid_search_results/grid_search_summary.tsv)"
    )
    p.add_argument(
        "--cuts", 
        nargs="*", 
        type=float,
        help="Metric cut values (e.g., 0.38 0.39 0.40). If omitted, auto-generates based on data range"
    )
    p.add_argument(
        "--output", "-o",
        help="Output TSV file path (default: stdout)"
    )
    p.add_argument(
        "--metric-col",
        default="metric_value",
        help="Name of the metric column (default: metric_value)"
    )
    p.add_argument(
        "--time-col",
        default="query_time_us",
        help="Name of the time column (default: query_time_us)"
    )
    return p.parse_args()


def read_tsv(filepath: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read TSV file and return header and rows."""
    try:
        with open(filepath, "r", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = reader.fieldnames or []
            rows = [row for row in reader]
        return header, rows
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        print("\nMake sure you've run the grid search first:", file=sys.stderr)
        print("  ./scripts/run_grid_search.sh -c <config.toml>", file=sys.stderr)
        sys.exit(1)


def to_float_safe(s: str) -> float:
    """Convert string to float, return nan on error."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def select_best_for_cut(
    rows: list[dict[str, str]], 
    metric_col: str, 
    time_col: str,
    cut: float
) -> Optional[dict[str, str]]:
    """Select row with minimal time among those with metric >= cut."""
    # Filter rows with metric >= cut
    filtered = [
        r for r in rows 
        if to_float_safe(r.get(metric_col, "nan")) >= cut
    ]
    
    if not filtered:
        return None
    
    # Select row with minimal time
    best = min(filtered, key=lambda r: to_float_safe(r.get(time_col, "inf")))
    return best


def auto_generate_cuts(rows: list[dict[str, str]], metric_col: str) -> list[float]:
    """Generate metric cuts based on data range."""
    # Get all metric values
    values = [to_float_safe(r.get(metric_col, "nan")) for r in rows]
    values = [v for v in values if not (v != v)]  # Filter out NaNs
    
    if not values:
        print("Warning: No valid metric values found, using default cuts", file=sys.stderr)
        return [0.38, 0.39, 0.40]
    
    min_val = min(values)
    max_val = max(values)
    
    # Generate cuts with step 0.001 from min to max
    step = 0.001
    num_steps = int((max_val - min_val) / step) + 1
    cuts = [round(min_val + i * step, 6) for i in range(num_steps)]
    
    return cuts


def main():
    args = parse_args()
    
    # Read input
    header, rows = read_tsv(args.input)
    
    if not rows:
        print(f"Error: No data found in {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Validate columns
    if args.metric_col not in header:
        print(f"Error: Column '{args.metric_col}' not found in input", file=sys.stderr)
        print(f"Available columns: {', '.join(header)}", file=sys.stderr)
        sys.exit(2)
    
    if args.time_col not in header:
        print(f"Error: Column '{args.time_col}' not found in input", file=sys.stderr)
        print(f"Available columns: {', '.join(header)}", file=sys.stderr)
        sys.exit(2)
    
    # Determine cuts
    if args.cuts:
        cuts = sorted(set(args.cuts))
        print(f"Using {len(cuts)} user-specified metric cuts", file=sys.stderr)
    else:
        cuts = auto_generate_cuts(rows, args.metric_col)
        print(f"Auto-generated {len(cuts)} metric cuts from data range", file=sys.stderr)
    
    # Print summary
    metric_values = [to_float_safe(r.get(args.metric_col, "nan")) for r in rows]
    metric_values = [v for v in metric_values if not (v != v)]
    
    if metric_values:
        print(f"Data range: {min(metric_values):.6f} - {max(metric_values):.6f}", file=sys.stderr)
        print(f"Cut range: {cuts[0]:.6f} - {cuts[-1]:.6f}", file=sys.stderr)
    print(f"Total configurations: {len(rows)}\n", file=sys.stderr)
    
    # Find best config for each cut
    results: list[dict[str, str]] = []
    
    for cut in cuts:
        best = select_best_for_cut(rows, args.metric_col, args.time_col, cut)
        
        if best is None:
            # No configuration achieves this cut - write placeholder
            empty_row = {k: "-" for k in header}
            empty_row["metric_cut"] = f"{cut:.6f}"
            empty_row[args.time_col] = "-"
            results.append(empty_row)
        else:
            # Add the best configuration
            out = dict(best)
            out["metric_cut"] = f"{cut:.6f}"
            results.append(out)
    
    # Output header: metric_cut + original columns
    out_header = ["metric_cut"] + header
    
    # Write output
    if args.output:
        out_f = open(args.output, "w", newline="")
        print(f"Writing results to: {args.output}", file=sys.stderr)
    else:
        out_f = sys.stdout
    
    try:
        writer = csv.DictWriter(out_f, fieldnames=out_header, delimiter="\t")
        writer.writeheader()
        
        for r in results:
            # Ensure all fields exist
            row = {k: r.get(k, "") for k in out_header}
            writer.writerow(row)
        
        if args.output:
            print(f"✓ Wrote {len(results)} results", file=sys.stderr)
    
    finally:
        if args.output:
            out_f.close()


if __name__ == "__main__":
    main()