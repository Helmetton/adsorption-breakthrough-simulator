# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Any


DEPENDENCIES: list[dict[str, Any]] = [
    {
        "library": "streamlit",
        "import_name": "streamlit",
        "package_name": "streamlit",
        "required": True,
        "purpose": "интерфейс приложения",
        "fallback": "нет",
    },
    {
        "library": "numpy",
        "import_name": "numpy",
        "package_name": "numpy",
        "required": True,
        "purpose": "численные массивы",
        "fallback": "нет",
    },
    {
        "library": "pandas",
        "import_name": "pandas",
        "package_name": "pandas",
        "required": True,
        "purpose": "таблицы и CSV",
        "fallback": "нет",
    },
    {
        "library": "scipy",
        "import_name": "scipy",
        "package_name": "scipy",
        "required": True,
        "purpose": "solve_ivp и least_squares",
        "fallback": "нет",
    },
    {
        "library": "matplotlib",
        "import_name": "matplotlib",
        "package_name": "matplotlib",
        "required": True,
        "purpose": "статические графики",
        "fallback": "нет",
    },
    {
        "library": "openpyxl",
        "import_name": "openpyxl",
        "package_name": "openpyxl",
        "required": True,
        "purpose": "чтение Excel-файлов",
        "fallback": "нет",
    },
    {
        "library": "plotly",
        "import_name": "plotly",
        "package_name": "plotly",
        "required": False,
        "purpose": "интерактивные графики",
        "fallback": "matplotlib",
    },
    {
        "library": "pint",
        "import_name": "pint",
        "package_name": "pint",
        "required": False,
        "purpose": "проверка единиц",
        "fallback": "встроенные конверсии",
    },
    {
        "library": "pydantic",
        "import_name": "pydantic",
        "package_name": "pydantic",
        "required": False,
        "purpose": "строгая проверка входных данных",
        "fallback": "ручная валидация",
    },
    {
        "library": "scikit-learn",
        "import_name": "sklearn",
        "package_name": "scikit-learn",
        "required": False,
        "purpose": "ML-модели",
        "fallback": "ML-раздел недоступен",
    },
    {
        "library": "joblib",
        "import_name": "joblib",
        "package_name": "joblib",
        "required": False,
        "purpose": "сохранение ML-моделей",
        "fallback": "сохранение моделей отключено",
    },
    {
        "library": "optuna",
        "import_name": "optuna",
        "package_name": "optuna",
        "required": False,
        "purpose": "подбор гиперпараметров ML",
        "fallback": "без автооптимизации",
    },
    {
        "library": "CoolProp",
        "import_name": "CoolProp",
        "package_name": "CoolProp",
        "required": False,
        "purpose": "свойства воды",
        "fallback": "табличная аппроксимация",
    },
]


def _is_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def check_dependencies() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dep in DEPENDENCIES:
        rows.append({**dep, "available": _is_available(dep["import_name"])})
    return rows


def install_missing_dependencies(only_optional: bool = True) -> dict[str, Any]:
    installed: list[str] = []
    already_available: list[str] = []
    failed_optional: list[str] = []
    failed_required: list[str] = []
    logs: list[str] = []

    for dep in check_dependencies():
        if dep["available"]:
            already_available.append(dep["library"])
            continue
        if only_optional and dep["required"]:
            continue

        package = dep["package_name"]
        command = [sys.executable, "-m", "pip", "install", package]
        logs.append(f"$ {' '.join(command)}")
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.stdout:
            logs.append(completed.stdout)
        if completed.stderr:
            logs.append(completed.stderr)

        if completed.returncode == 0 and _is_available(dep["import_name"]):
            installed.append(dep["library"])
        elif dep["required"]:
            failed_required.append(dep["library"])
        else:
            failed_optional.append(dep["library"])
            if dep["library"] == "CoolProp":
                logs.append("CoolProp не установлен. Будет использована fallback-таблица свойств воды.")

    return {
        "installed": installed,
        "already_available": already_available,
        "failed_optional": failed_optional,
        "failed_required": failed_required,
        "logs": "\n".join(logs).strip(),
        "status": check_dependencies(),
    }
