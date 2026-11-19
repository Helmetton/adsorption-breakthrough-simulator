"""Opt-in wall-clock diagnostics, disabled unless SIMULATOR_STARTUP_TRACE is set."""
from collections import defaultdict
from functools import wraps
import importlib
import json
import os
import sys
from pathlib import Path
import threading
import time
import uuid

_local = threading.local()


class StartupProbe:
    def __init__(self):
        self.directory = os.environ.get("SIMULATOR_STARTUP_TRACE")
        self.enabled = bool(self.directory)
        self.started = time.perf_counter()
        self.started_wall = time.time()
        self.run_id = uuid.uuid4().hex
        self.phases = []
        self.operations = defaultdict(lambda: {"calls": 0, "wall_s": 0.0})
        self.messages = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self.blocks = {}
        self.tables = []
        _local.probe = self

    def mark(self, label):
        if self.enabled:
            self.phases.append({"phase": label, "elapsed_s": time.perf_counter() - self.started})

    def install(self, namespace):
        if not self.enabled:
            return
        names = {"safe_read_csv", "load_ml_dataset", "load_saved_model", "validate_ml_dataset",
                 "validate_dataset_for_breakthrough_training", "dataset_parameter_provenance",
                 "set_ml_dataset_for_training", "render_dataset_preview", "render_help",
                 "render_experiment_overview", "saved_plot_png", "simulate_breakthrough",
                 "fit_model_to_experiment", "train_adsorber_models", "load_dataset_file", "cached_breakthrough_quality"}
        for name in names:
            if name in namespace:
                namespace[name] = measured(namespace[name])
        for module_name, functions in {
            "scientific_provenance": ("file_hash", "dataframe_hash", "compatibility_errors", "resolve_dataset_provenance"),
            "parameter_provenance": ("dataset_case_parameter_provenance", "parameter_origins"),
            "ml_adsorber": ("_derive_breakthrough_times", "canonicalize_ml_columns", "validate_dataset_for_breakthrough_training", "load_ml_dataset"),
            "ui_presentation": ("display_dataframe",),
            "pandas": ("read_csv", "read_excel"),
            "joblib": ("load",),
        }.items():
            if module_name == "joblib" and module_name not in sys.modules:
                continue
            module = importlib.import_module(module_name)
            for name in functions:
                setattr(module, name, measured(getattr(module, name)))
        from streamlit.runtime.scriptrunner_utils.script_run_context import ScriptRunContext
        original = ScriptRunContext.enqueue
        if not getattr(original, "_startup_probe_wrapped", False):
            @wraps(original)
            def enqueue(context, message):
                probe = getattr(_local, "probe", None)
                if probe and probe.enabled:
                    kind = message.WhichOneof("type")
                    path = tuple(message.metadata.delta_path)
                    if kind == "delta" and message.delta.HasField("add_block"):
                        block = message.delta.add_block
                        if block.WhichOneof("type") == "expandable":
                            probe.blocks[path] = {"label": block.expandable.label, "expanded": block.expandable.expanded}
                    if kind == "delta" and message.delta.HasField("new_element"):
                        kind = message.delta.new_element.WhichOneof("type")
                    if kind in {"dataframe", "table"}:
                        probe.tables.append({"type": kind, "bytes": message.ByteSize(),
                            "collapsed_ancestors": [spec["label"] for parent, spec in probe.blocks.items()
                                                    if path[:len(parent)] == parent and not spec["expanded"]]})
                    probe.messages[kind]["calls"] += 1
                    probe.messages[kind]["bytes"] += message.ByteSize()
                return original(context, message)
            enqueue._startup_probe_wrapped = True
            ScriptRunContext.enqueue = enqueue

    def finish(self, backend):
        if not self.enabled:
            return
        self.mark("app_complete")
        data = {"pid": os.getpid(), "run_id": self.run_id, "started_perf": self.started, "started_wall": self.started_wall,
                "elapsed_s": time.perf_counter() - self.started, "phases": self.phases,
                "operations": dict(self.operations), "messages": dict(self.messages), "tables": self.tables}
        path = Path(self.directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"app_{os.getpid()}_{self.run_id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Test-only DOM sentinel: browser readiness is not HTTP readiness.
        backend.markdown(f'<span data-startup-ready="{self.run_id}" style="display:none"></span>', unsafe_allow_html=True)


def measured(function):
    if getattr(function, "_startup_probe_wrapped", False):
        return function
    @wraps(function)
    def wrapper(*args, **kwargs):
        probe = getattr(_local, "probe", None)
        if not probe or not probe.enabled:
            return function(*args, **kwargs)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            entry = probe.operations[f"{function.__module__}.{function.__name__}"]
            entry["calls"] += 1
            entry["wall_s"] += time.perf_counter() - started
    wrapper._startup_probe_wrapped = True
    return wrapper
