"""CLI for generating the nonlinear elliptic oracle dataset."""

from __future__ import annotations

import argparse

from .config import load_config
from .dataset import generate_oracle_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chonkdiff_elliptic.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = generate_oracle_dataset(config, force=args.force)
    print(out_path)


if __name__ == "__main__":
    main()
