"""Defer hidden dataset loading; cache only identical data and validation inputs."""
from pathlib import Path
import hashlib
import pickle

import pandas as pd
import streamlit as st

import ml_adsorber
import scientific_provenance

ROOT = Path(__file__).resolve().parent


def tracked_tabs(ui, labels, key):
    """Track selection in Streamlit 1.58 without changing labels or widget keys."""
    previous = ui.session_state.get(key + "_labels", [])
    selected = ui.session_state.get(key)
    if selected in previous and selected not in labels:
        ui.session_state[key] = labels[previous.index(selected)]
    ui.session_state[key + "_labels"] = list(labels)
    return ui.tabs(labels, key=key, on_change="rerun")


def exact_dataframe_key(df):
    # Never use Streamlit's sampling hash for a large scientific dataframe.
    return (scientific_provenance.dataframe_hash(df),
            tuple((repr(column), repr(dtype)) for column, dtype in zip(df.columns, df.dtypes)),
            hashlib.sha256(pd.util.hash_pandas_object(df.index).values.tobytes()).hexdigest(),
            repr(df.index.names),
            hashlib.sha256(pickle.dumps(df.attrs, protocol=5)).hexdigest())


def load_dataset_file(path):
    path = Path(path).resolve()
    # Read bytes on every request, including same-size/same-mtime file changes.
    digest = scientific_provenance.file_hash(path)
    result = _read_dataset_file(str(path), digest,
                               (scientific_provenance.file_hash(ROOT / "encoding_utils.py"),
                                scientific_provenance.file_hash(ROOT / "ml_adsorber.py")))
    if scientific_provenance.file_hash(path) != digest:
        raise OSError("Dataset changed during loading; retry the operation.")
    return result


@st.cache_data(show_spinner=False, max_entries=1)
def _read_dataset_file(path, digest, reader_version):
    result = ml_adsorber.load_ml_dataset(path)
    if scientific_provenance.file_hash(path) != digest:
        raise OSError("Dataset changed during loading; retry the operation.")
    return result


def cached_breakthrough_quality(df, required_service_time_h=None, training_mode="validated"):
    versions = tuple(scientific_provenance.file_hash(ROOT / name) for name in
                     ("ml_adsorber.py", "model.py"))
    return _breakthrough_quality(exact_dataframe_key(df), required_service_time_h,
                                 training_mode, versions, _df=df)


@st.cache_data(show_spinner=False, max_entries=2)
def _breakthrough_quality(data_key, required_service_time_h, training_mode, versions, _df):
    return ml_adsorber.validate_dataset_for_breakthrough_training(
        _df, required_service_time_h=required_service_time_h, training_mode=training_mode)
