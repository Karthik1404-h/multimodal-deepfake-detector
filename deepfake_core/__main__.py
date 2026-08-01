"""Allow running deepfake_core as a module: python -m deepfake_core"""

from deepfake_core.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
