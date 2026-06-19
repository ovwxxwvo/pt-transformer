#!/usr/bin/env bash


proj_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
echo $proj_dir
echo ---------- ---------- ---------- ----------

cloc \
  --fullpath \
  --not-match-d="${proj_dir}/data" \
  --not-match-d="${proj_dir}/logs" \
  --not-match-d="${proj_dir}/database" \
  --exclude-dir=".git,.venv,__pycache__" \
  --exclude-ext="html,txt" \
  "$proj_dir"

