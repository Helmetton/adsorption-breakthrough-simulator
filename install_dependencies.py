# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from dependency_manager import check_dependencies, install_missing_dependencies


def print_status(rows: list[dict]) -> None:
    for row in rows:
        mark = "OK" if row["available"] else "MISSING"
        kind = "required" if row["required"] else "optional"
        print(
            f"{mark:7} {row['library']:14} import={row['import_name']:12} "
            f"package={row['package_name']:14} {kind:8} fallback={row['fallback']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Install project dependencies.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check dependency status; do not install anything.",
    )
    parser.add_argument(
        "--optional-only",
        action="store_true",
        help="Install only missing optional dependencies.",
    )
    args = parser.parse_args()

    print("Dependency status before installation:")
    before = check_dependencies()
    print_status(before)

    if args.check:
        return 0

    result = install_missing_dependencies(only_optional=args.optional_only)
    if result["logs"]:
        print("\nInstallation log:")
        print(result["logs"])

    print("\nSummary:")
    print("installed:", ", ".join(result["installed"]) or "-")
    print("already_available:", ", ".join(result["already_available"]) or "-")
    print("failed_optional:", ", ".join(result["failed_optional"]) or "-")
    print("failed_required:", ", ".join(result["failed_required"]) or "-")

    if "CoolProp" in result["failed_optional"]:
        print("CoolProp не установлен. Будет использована fallback-таблица свойств воды.")

    print("\nDependency status after installation:")
    print_status(result["status"])
    return 1 if result["failed_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
