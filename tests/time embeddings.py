"""Convenience script for timing a single embedding.

This file intentionally lives under ``tests/`` so it is easy
to discover alongside the other embedding tests, but it is
meant to be run as a small benchmark script rather than as
an automated unit test.

It simply delegates to the existing ``time_single_embedding``
module, which contains the real timing logic.
"""

from time_single_embedding import main as _time_single_embedding_main


if __name__ == "__main__":  # pragma: no cover - manual timing helper
    _time_single_embedding_main()

