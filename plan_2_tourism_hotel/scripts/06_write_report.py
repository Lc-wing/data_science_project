"""Phase A step 6: write the report draft from the output files and audit every number.

    uv run python scripts/06_write_report.py

Reads the frozen panel and the outputs of steps 1 to 5. It does not refit a model.
"""

from __future__ import annotations

import json
import sys

from macao_tourism.report import write_report


def main() -> int:
    result = write_report()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
