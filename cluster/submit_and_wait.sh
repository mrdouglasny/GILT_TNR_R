#!/bin/bash
# Wrapper script that calls the master submit_and_wait.sh
# This allows the same master script to be used across all projects

# Get the directory containing this wrapper script
WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Master script is in cluster/ at project root
MASTER_SCRIPT="$WRAPPER_DIR/../../cluster/submit_and_wait.sh"

# Export wrapper directory so master script can detect it
export SUBMIT_AND_WAIT_WRAPPER_DIR="$WRAPPER_DIR"

# Execute master script with all arguments
exec "$MASTER_SCRIPT" "$@"
