#!/bin/bash
set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v