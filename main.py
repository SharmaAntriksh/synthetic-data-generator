import multiprocessing
import sys

from src.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Propagate the CLI return code as the process exit code so failures
    # (e.g. coverage-policy abort, config errors) are detectable by callers/CI.
    sys.exit(main())
