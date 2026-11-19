"""Read-only views of experiments, parameter payloads, datasets and help."""
from __future__ import annotations

from datetime import datetime
import math

import pandas as pd

from translations import tr
from ui_presentation import display_dataframe, field_label
from parameter_provenance import parameter_origins


def display_number(value):
    if value is None or pd.isna(value):
        return "—"
    if not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    if value != 0 and (abs(value) < 0.001 or abs(value) >= 1e6):
        return f"{value:.3e}"
    return f"{value:.5g}"


def experiment_description(rows, lang, record=None):
    first = rows[0]
    date = str(first.get("source_sheet", ""))
    try:
        date = datetime.strptime(date, "%d.%m.%y").strftime("%d.%m.%Y")
    except ValueError:
        pass
    label = tr(lang, "ui_experiment_brief").format(
        date=date or "—", height=display_number(first.get("bed_height_cm")),
        mass=display_number(first.get("carbon_mass_g")),
        flow=display_number(first.get("flow_ml_min_assumed")),
    )
    if record is not None:
        label += " · " + tr(lang, "ui_record_number").format(number=record)
    return label


def render_experiment_overview(st, experiment_id, rows, lang):
    st.markdown(f"#### {tr(lang, 'ui_selected_experiment')}")
    st.caption(experiment_description(rows, lang))
    first = rows[0]
    columns = st.columns(3)
    for column, key, value in zip(columns, ("C0_mg_L", "temperature_C", "n_points"), (first.get("C0_mg_L"), first.get("temperature_C"), len(rows))):
        with column:
            st.metric(field_label(key, lang), display_number(value))
    points = pd.DataFrame({"time_min": [row["time_min"] for row in rows], "C_out_C0": [row["C_out_C0"] for row in rows]})
    st.scatter_chart(points, x="time_min", y="C_out_C0", x_label=tr(lang, "time_min"), y_label="C/C0", height=250)
    st.caption(tr(lang, "ui_observations_not_model"))
    with st.expander(tr(lang, "ui_details")):
        st.code(experiment_id, language=None)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def dataset_counts(st, rows, cases, lang):
    left, right = st.columns(2)
    with left:
        st.metric(field_label("number_of_rows", lang), f"{int(rows):,}".replace(",", " "))
    with right:
        st.metric(field_label("number_of_cases", lang), f"{int(cases):,}".replace(",", " "))
    st.caption(tr(lang, "ui_scenario_rows_note"))


def render_quality_overview(st, report, lang):
    stats = report.get("target_stats", {})
    target_max = stats.get("max")
    share = report.get("share_breakthrough_found")
    values = (
        ("number_of_breakthrough_found_cases", display_number(report.get("number_of_breakthrough_found_cases"))),
        ("share_breakthrough_found", "—" if share is None else f"{share:.1%}"),
        ("target_max_h", "—" if target_max is None else display_number(target_max / 60.0)),
    )
    for column, (key, value) in zip(st.columns(3), values):
        with column:
            st.metric(field_label(key, lang), value)


def preview_columns(df):
    groups = (
        ("case_id",), ("time_min",), ("C_out_C0",), ("C_out_mg_L",), ("C0_mg_L",),
        ("flow_ml_min", "flow_m3_h", "flow"),
        ("height_cm", "bed_height_cm", "height_m", "height"),
        ("diameter_cm", "column_diameter_cm", "diameter_m", "diameter"),
        ("carbon_mass_g", "carbon_mass_kg", "carbon_mass"),
        ("EBCT_min",), ("temperature_C", "temperature"),
    )
    return [next(key for key in group if key in df) for group in groups if any(key in df for key in group)]


def render_dataset_preview(st, df, lang):
    sample = df.head(30).copy(deep=True)

    def table(columns):
        raw = sample[columns].copy(deep=True)
        view = display_dataframe(raw, lang)
        numeric = [label for label in view if pd.api.types.is_numeric_dtype(view[label]) and not pd.api.types.is_bool_dtype(view[label])]
        styled = view.style.format({label: display_number for label in numeric}, na_rep="—")
        config = {label: st.column_config.Column(width="small" if key == "case_id" else "medium", help=f"{label} ({key})") for key, label in zip(columns, view.columns)}
        # Bypass only the facade's column-compaction rule; the view is already localized.
        st._backend.dataframe(styled, width="stretch", hide_index=True, column_config=config)

    st.markdown(f"##### {tr(lang, 'ui_dataset_preview')}")
    table(preview_columns(sample) or list(sample.columns[:10]))
    st.caption(tr(lang, "ui_preview_rows_note"))
    with st.expander(tr(lang, "ui_full_dataset_columns")):
        table(list(sample.columns))


def render_parameter_source(st, params, experiments, source_message, source_level, source_table, lang):
    source = params.get("parameter_source")
    rows = experiments.get(params.get("experiment_id"))
    if rows and source in {"best_experiment", "last_calibration"}:
        message = tr(lang, "ui_parameter_source_brief").format(
            source=tr(lang, "parameter_source_" + source), experiment=experiment_description(rows, lang),
        )
    else:
        message = source_message
    if source_level == "success":
        st.success(message)
    elif source_level == "warning":
        st.warning(message)
    else:
        st.caption(message)
    view = source_table.copy(deep=True)
    parameter_column = tr(lang, "col_parameter")
    core = view[parameter_column].isin(["K_H", "k_LDF", "D_ax"])
    if source in {"best_experiment", "last_calibration"} and any(record.get("origin") in {"default", "unknown"} for record in parameter_origins(params).values()):
        st.warning(tr(lang, "provenance_auxiliary_caution"))
    st.dataframe(view.loc[core].copy(), width="stretch", hide_index=True)
    with st.expander(tr(lang, "ui_parameter_source_details")):
        if source == "best_experiment":
            st.caption(tr(lang, "ui_best_source_criterion"))
        st.caption(source_message)
        st.dataframe(view, width="stretch", hide_index=True)
        st.json(params)


HELP_GROUPS = (
    ("ui_help_start", ("ui_help_workflow", "help_scales")),
    ("ui_help_parameters", ("help_base_model", "help_henry_isotherm", "help_ldf", "help_axial_dispersion", "help_pressure", "help_temperature", "help_temperature_auto", "help_auto_kh_from_profile")),
    ("ui_help_calibration", ("ui_help_calibration_steps", "help_holdout_validation")),
    ("ui_help_results", ("help_curve_zero", "help_mass_balance", "help_engineering_diagnostics")),
    ("ui_help_ml", ("ui_help_ml_flow",)),
    ("ui_help_limits", ("help_parameters_not_universal", "help_validated_vs_extrapolated")),
)


def render_help(st, lang):
    for group, questions in HELP_GROUPS:
        st.markdown(f"### {tr(lang, group)}")
        for section in questions:
            with st.expander(tr(lang, section + "_title")):
                body = tr(lang, section + "_body")
                short, separator, rest = body.partition(". ")
                st.markdown(short + ("." if separator else ""))
                if rest:
                    st.markdown(rest)
                if section == "ui_help_ml_flow":
                    st.markdown(tr(lang, "help_which_ml_model_body"))
                    st.markdown(tr(lang, "help_ml_selection_body"))
