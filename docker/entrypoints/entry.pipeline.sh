#!/usr/bin/env bash
# Entrypoint for executing an ExoLife pipeline.  Accepts two optional
# arguments: the path to a DAG YAML specification and the execution
# mode (`sequential` or `parallel`).  If no arguments are provided
# the defaults are `config/dags/exolife_dag_v1.yaml` and `parallel`.
#
# Usage:
#   entry.pipeline.sh [dag_file] [mode]
#
# Example:
#   entry.pipeline.sh config/dags/my_dag.yaml sequential

set -euo pipefail

DAG_FILE="${1:-config/dags/exolife_dag_v1.yaml}"
MODE="${2:-parallel}"

echo "🛰️  Running ExoLife DAG: $DAG_FILE (mode=$MODE)"
exec exolife dag run "$DAG_FILE" --mode "$MODE"