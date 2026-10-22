import sys

from .cli import main

if __name__ == "__main__":   # worker processes started with "spawn" import this module again
    sys.exit(main())
