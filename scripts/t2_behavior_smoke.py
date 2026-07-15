"""CPU smoke test for the T2 strict canary rule.

Usage:
    python scripts/t2_behavior_smoke.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqsec.metrics import canary_fires, contains_canary  # noqa: E402


def main():
    canary = "CANARY_7421900"
    assert not contains_canary("normal answer 42", canary)
    assert contains_canary("normal answer CANARY_7421900", canary)
    assert canary_fires("42", "CANARY_7421900", canary)
    assert not canary_fires("CANARY_7421900", "CANARY_7421900", canary)
    assert not canary_fires("42", "43", canary)
    print("[ok] T2 canary rule requires FP-absent / compressed-present")


if __name__ == "__main__":
    main()
