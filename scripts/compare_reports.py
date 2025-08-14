import os
import sys
import pandas as pd

def extract_and_compare_reports(prefix):
    # Find all directories starting with the given prefix
    folders = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith(prefix)
    ]

    if not folders:
        print(f"No folders found with prefix '{prefix}'")
        return

    # Create list of (suffix, folder_name) and sort by suffix lexicographically
    suffix_to_folder = sorted([
        (d[len(prefix):], d) for d in folders
    ], key=lambda x: x[0])

    print("Found folders (sorted by suffix):")
    for suffix, folder in suffix_to_folder:
        print(f"  {suffix} -> {folder}")

    query_time_dfs = []
    recall_dfs = []
    subsection_column = None
    suffixes = []

    for suffix, folder in suffix_to_folder:
        report_path = os.path.join(folder, "report.tsv")
        if not os.path.exists(report_path):
            print(f"Warning: report.tsv not found in {folder}")
            continue

        df = pd.read_csv(report_path, sep='\t')

        if subsection_column is None:
            subsection_column = df['Subsection']

        query_col_name = f"Query Time_{suffix}"
        query_time_dfs.append(
            df[['Query Time (microsecs)']].rename(
                columns={'Query Time (microsecs)': query_col_name}
            )
        )

        recall_dfs.append((suffix, df['Recall']))
        suffixes.append(suffix)

    if subsection_column is None or not query_time_dfs:
        print("No valid data found.")
        return

    # Check Recall consistency
    print("\nChecking Recall consistency across reports...\n")
    base_suffix, base_recall = recall_dfs[0]
    recall_ok = True

    for suffix, recall in recall_dfs[1:]:
        diffs = base_recall != recall
        if diffs.any():
            recall_ok = False
            print(f"⚠️ Recall mismatch between {base_suffix} and {suffix}:")
            for idx in recall.index[diffs]:
                print(f"  Row {idx} ({subsection_column[idx]}): {base_suffix}={base_recall[idx]}, {suffix}={recall[idx]}")

    if recall_ok:
        print("✅ All Recall values are consistent across files.\n")

    # Assemble base dataframe
    result_df = pd.concat([subsection_column] + query_time_dfs, axis=1)

    # Compute percentage increases
    qt_columns = [f"Query Time_{s}" for s in suffixes]
    query_times = result_df[qt_columns]
    row_mins = query_times.min(axis=1)

    delta_columns = []
    for suffix in suffixes:
        qt_col = f"Query Time_{suffix}"
        delta_col = f"{qt_col} Δ%"
        result_df[delta_col] = ((result_df[qt_col] - row_mins) / row_mins * 100).round(1)
        delta_columns.append(delta_col)

    # Reorder columns: Subsection, then alternate QT and Δ%
    ordered_columns = ['Subsection']
    for qt, delta in zip(qt_columns, delta_columns):
        ordered_columns.extend([qt, delta])

    result_df = result_df[ordered_columns]

    # Print result
    print("Final Query Time Comparison Table with Δ%:\n")
    print(result_df.to_string(index=False))

    # Save to CSV
    output_filename = f"query_time_comparison_{prefix}.csv"
    result_df.to_csv(output_filename, index=False)
    print(f"\nTable saved to: {output_filename}")
    print(result_df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_report.py <folder_prefix>")
        sys.exit(1)

    prefix = sys.argv[1]
    extract_and_compare_reports(prefix)