# REFINE-PLAN

This repo contains the source code for REFINE-PLAN, an automated tool for refining hand-designed behaviour trees to achieve higher robustness to uncertainty.

## Dependencies


## Installation


## Build the documentation

1. Install the required packages:

    ```bash
    pip install -r docs/requirements.txt
    ```

2. Install the package to be documented:

    ```bash
    pip install refine_plan/
    ```
    
    Or add it to your Python path:
    ```bash
    ./setup_dev.sh
    ```

3. Build the documentation:

    ```bash
    cd docs
    make html
    ```

4. Look at the documentation:

    ```bash
    cd docs
    firefox build/html/index.html
    ```

### Clean documentation build artifacts

If you want to clean the documentation, you can run:

```bash
cd docs
rm -r source/API
make clean
```
