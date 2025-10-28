# Wheel Build Workflow

This document explains how the wheel building process works in the GitHub Actions workflow.

## Build Process Overview

### Local Build (build-python.sh)
```bash
./scripts/build-python.sh
```

This script:
1. Builds Docker image via `build-docker.sh`
2. Runs Docker container with mounted workspace
3. Executes build commands inside container
4. **Output**: Repaired wheel in `wheelhouse/` directory

### GitHub Actions Workflow (build-wheel.yml)

The workflow replicates the local build process:

1. **Checkout**: Gets the repository code
2. **Docker Build**: Builds the calico Docker image with all dependencies
3. **Wheel Build**: 
   - Runs `python3 -m build --wheel` → creates wheel in `dist/`
   - Runs `auditwheel repair` → repairs wheel and outputs to `wheelhouse/`
4. **Upload Artifact**: Uploads the repaired wheel from `wheelhouse/` as a GitHub artifact

## Artifacts

After the workflow runs, you can download:
- **calico-wheel-manylinux**: The production-ready manylinux wheel (30 day retention)
- **build-logs**: Dist and wheelhouse directories for debugging (7 day retention)

## Triggers

The workflow runs on:
- Push to `mainline` or `neil/34-calico-pybind212` branches
- Pull requests to `mainline`
- Manual trigger via workflow_dispatch

## Key Differences from build-python.sh

1. **No virtual environment**: The Dockerfile installs packages to system Python, so the workflow doesn't activate a venv
2. **Direct Python calls**: Uses system Python3 directly
3. **Artifact upload**: Automatically uploads wheels as GitHub artifacts

## Wheel Output Location

- **Initial wheel**: `dist/calico-*-linux_x86_64.whl`
- **Final repaired wheel**: `wheelhouse/calico-*-manylinux_2_35_x86_64.whl` ← This is what gets uploaded!

The `auditwheel` tool repairs the wheel to ensure compatibility with the manylinux platform and bundles any necessary shared libraries.

