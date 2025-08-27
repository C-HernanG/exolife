#!/usr/bin/env python3
"""
Analysis script for ExoLife processed data outputs
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def analyze_parquet_file(file_path):
    """Analyze a single parquet file and return summary statistics"""
    try:
        df = pd.read_parquet(file_path)

        analysis = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            ),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicated_rows": df.duplicated().sum(),
            "unique_value_counts": {},
            "sample_data": df.head(3).to_dict("records") if len(df) > 0 else [],
            "numeric_summary": {},
            "categorical_summary": {},
        }

        # Analyze unique values for categorical columns
        for col in df.columns:
            unique_count = df[col].nunique()
            analysis["unique_value_counts"][col] = unique_count

            # For categorical columns with reasonable number of unique values
            if unique_count <= 20 and df[col].dtype == "object":
                analysis["categorical_summary"][col] = df[col].value_counts().to_dict()

        # Numeric summary statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()

        return analysis

    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "error": str(e),
        }


def generate_report():
    """Generate comprehensive analysis report"""

    print("Starting ExoLife Processed Data Analysis...")

    # Define paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_path = project_root / "data" / "processed"
    catalog_path = base_path / "exolife_catalog"
    normalized_path = base_path / "normalized_tables"

    # Get all parquet files
    parquet_files = []

    # Main catalog
    if catalog_path.exists():
        parquet_files.extend(list(catalog_path.glob("*.parquet")))

    # Normalized tables
    if normalized_path.exists():
        parquet_files.extend(list(normalized_path.glob("*.parquet")))

    print(f"Found {len(parquet_files)} parquet files to analyze")

    # Analyze each file
    analyses = []
    for file_path in parquet_files:
        print(f"Analyzing: {file_path.name}")
        analysis = analyze_parquet_file(str(file_path))
        analyses.append(analysis)

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EXOLIFE PROCESSED DATA ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total files analyzed: {len(parquet_files)}")
    report_lines.append("")

    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)

    total_size_mb = sum(
        [a.get("file_size_mb", 0) for a in analyses if "error" not in a]
    )
    total_rows = sum([a.get("shape", (0, 0))[0] for a in analyses if "error" not in a])
    total_columns = sum(
        [a.get("shape", (0, 0))[1] for a in analyses if "error" not in a]
    )
    successful_analyses = len([a for a in analyses if "error" not in a])
    failed_analyses = len([a for a in analyses if "error" in a])

    report_lines.append(f"• Total disk space used: {total_size_mb:.2f} MB")
    report_lines.append(f"• Total records across all files: {total_rows:,}")
    report_lines.append(f"• Total columns across all files: {total_columns}")
    report_lines.append(f"• Successfully analyzed files: {successful_analyses}")
    report_lines.append(f"• Failed analyses: {failed_analyses}")
    report_lines.append("")

    # Detailed Analysis for each file
    report_lines.append("DETAILED FILE ANALYSIS")
    report_lines.append("-" * 40)

    for analysis in analyses:
        if "error" in analysis:
            report_lines.append(f"\n❌ ERROR ANALYZING: {analysis['file_name']}")
            report_lines.append(f"   Error: {analysis['error']}")
            continue

        report_lines.append(f"\n📊 {analysis['file_name'].upper()}")
        report_lines.append("   " + "=" * 50)

        # Basic info
        report_lines.append(f"   File size: {analysis['file_size_mb']} MB")
        report_lines.append(
            f"   Dimensions: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns"
        )
        report_lines.append(f"   Memory usage: {analysis['memory_usage_mb']} MB")
        report_lines.append(f"   Duplicated rows: {analysis['duplicated_rows']:,}")
        report_lines.append("")

        # Column information
        report_lines.append("   COLUMNS:")
        for i, col in enumerate(analysis["columns"], 1):
            dtype = str(analysis["dtypes"][col])
            unique_count = analysis["unique_value_counts"][col]
            null_count = analysis["null_counts"][col]
            null_pct = analysis["null_percentages"][col]

            report_lines.append(f"   {i:2d}. {col}")
            report_lines.append(
                f"       Type: {dtype} | Unique: {unique_count:,} | Nulls: {null_count:,} ({null_pct:.1f}%)"
            )

        report_lines.append("")

        # Data quality issues
        high_null_cols = [
            col for col, pct in analysis["null_percentages"].items() if pct > 50
        ]
        if high_null_cols:
            report_lines.append("   ⚠️  DATA QUALITY CONCERNS:")
            report_lines.append("   Columns with >50% missing values:")
            for col in high_null_cols:
                pct = analysis["null_percentages"][col]
                report_lines.append(f"   - {col}: {pct:.1f}% missing")
            report_lines.append("")

        # Numeric summary for key columns (first 5 numeric columns)
        if analysis["numeric_summary"]:
            report_lines.append("   NUMERIC STATISTICS (top 5 columns):")
            numeric_cols = list(analysis["numeric_summary"].keys())[:5]
            for col in numeric_cols:
                stats = analysis["numeric_summary"][col]
                report_lines.append(f"   {col}:")
                report_lines.append(
                    f"     Mean: {stats.get('mean', 'N/A'):.3f} | Std: {stats.get('std', 'N/A'):.3f}"
                )
                report_lines.append(
                    f"     Min: {stats.get('min', 'N/A'):.3f} | Max: {stats.get('max', 'N/A'):.3f}"
                )
                report_lines.append(
                    f"     25%: {stats.get('25%', 'N/A'):.3f} | 75%: {stats.get('75%', 'N/A'):.3f}"
                )
            report_lines.append("")

        # Categorical summary for key columns
        if analysis["categorical_summary"]:
            report_lines.append("   CATEGORICAL VALUE DISTRIBUTIONS:")
            # First 3 categorical columns
            for col, value_counts in list(analysis["categorical_summary"].items())[:3]:
                report_lines.append(f"   {col}:")
                # Top 5 values
                for value, count in list(value_counts.items())[:5]:
                    report_lines.append(f"     '{value}': {count:,}")
                if len(value_counts) > 5:
                    report_lines.append(
                        f"     ... and {len(value_counts) - 5} more values"
                    )
            report_lines.append("")

    # Cross-file analysis
    report_lines.append("\nCROSS-FILE ANALYSIS")
    report_lines.append("-" * 40)

    # File size comparison
    file_sizes = [
        (a["file_name"], a.get("file_size_mb", 0)) for a in analyses if "error" not in a
    ]
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    report_lines.append("File sizes (largest to smallest):")
    for name, size in file_sizes:
        report_lines.append(f"  {name}: {size:.2f} MB")
    report_lines.append("")

    # Row count comparison
    row_counts = [
        (a["file_name"], a.get("shape", (0, 0))[0])
        for a in analyses
        if "error" not in a
    ]
    row_counts.sort(key=lambda x: x[1], reverse=True)

    report_lines.append("Record counts (largest to smallest):")
    for name, count in row_counts:
        report_lines.append(f"  {name}: {count:,} rows")
    report_lines.append("")

    # Common columns analysis
    all_columns = set()
    file_columns = {}
    for analysis in analyses:
        if "error" not in analysis:
            cols = set(analysis["columns"])
            all_columns.update(cols)
            file_columns[analysis["file_name"]] = cols

    # Find common columns across files
    if len(file_columns) > 1:
        common_cols = set.intersection(*file_columns.values())
        if common_cols:
            report_lines.append(
                f"Common columns across all files ({len(common_cols)}):"
            )
            for col in sorted(common_cols):
                report_lines.append(f"  - {col}")
            report_lines.append("")

    # Data pipeline recommendations
    report_lines.append("RECOMMENDATIONS & INSIGHTS")
    report_lines.append("-" * 40)

    # Storage efficiency
    if total_size_mb > 100:
        report_lines.append("💾 STORAGE EFFICIENCY:")
        report_lines.append(f"   Current total size: {total_size_mb:.2f} MB")
        report_lines.append("   Consider compression optimization for large datasets")
        report_lines.append("")

    # Data quality recommendations
    high_null_files = []
    for analysis in analyses:
        if "error" not in analysis:
            high_null_cols = [
                col for col, pct in analysis["null_percentages"].items() if pct > 30
            ]
            if high_null_cols:
                high_null_files.append((analysis["file_name"], high_null_cols))

    if high_null_files:
        report_lines.append("🔍 DATA QUALITY RECOMMENDATIONS:")
        for file_name, cols in high_null_files:
            report_lines.append(f"   {file_name}:")
            report_lines.append(
                f"     - Review high missing value columns: {', '.join(cols[:3])}"
            )
            if len(cols) > 3:
                report_lines.append(
                    f"     - And {len(cols) - 3} more columns with high null rates"
                )
        report_lines.append("")

    # Performance recommendations
    large_files = [
        a for a in analyses if "error" not in a and a.get("shape", (0, 0))[0] > 100000
    ]
    if large_files:
        report_lines.append("⚡ PERFORMANCE RECOMMENDATIONS:")
        for analysis in large_files:
            report_lines.append(
                f"   {analysis['file_name']} ({analysis['shape'][0]:,} rows):"
            )
            report_lines.append("     - Consider partitioning for large datasets")
            report_lines.append("     - Optimize data types to reduce memory usage")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


if __name__ == "__main__":
    try:
        report = generate_report()

        # Save report to file using relative path
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_file = (
            project_root
            / "results"
            / "data_reports"
            / "exolife_processed_data_analysis_report.txt"
        )

        # Create directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n✅ Analysis complete! Report saved to: {output_file}")
        print(f"Report contains {len(report.split(chr(10)))} lines")

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback

        traceback.print_exc()
