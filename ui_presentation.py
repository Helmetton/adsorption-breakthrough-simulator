"""Display-only localization. Input values, model data and export schemas stay intact."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from translations import LEGACY_UI_TEXT_KEYS, TRANSLATIONS, UI_FIELD_ALIASES, UI_FIELDS, tr
from display_units import display_volume_units


@lru_cache(maxsize=1)
def _messages():
    messages = dict(LEGACY_UI_TEXT_KEYS)
    for collection in TRANSLATIONS.values():
        messages.update({value: key for key, value in collection.items()})
    return messages


@lru_cache(maxsize=1)
def _templates():
    patterns = []
    for language, messages in TRANSLATIONS.items():
        for key, template in messages.items():
            fields = re.findall(r"\{([a-zA-Z_][a-zA-Z_0-9]*)\}", template)
            if not fields or "%{" in template:
                continue
            parts = re.split(r"\{[a-zA-Z_][a-zA-Z_0-9]*\}", template)
            if len("".join(parts)) < 12:
                continue
            pattern = "(.+?)".join(re.escape(part) for part in parts)
            patterns.append((re.compile(pattern, re.DOTALL), key, fields))
    return patterns


def field_label(name, lang):
    name = str(name)
    canonical = UI_FIELD_ALIASES.get(name, name)
    if canonical in UI_FIELDS:
        return tr(lang, "ui_field_" + canonical)
    if canonical in TRANSLATIONS.get(lang, {}):
        return tr(lang, canonical)
    return translate_text(name, lang)


def translate_text(value, lang):
    if not isinstance(value, str):
        return value
    return display_volume_units(_translate_string(value, lang))


def choice_widget_key(key, lang):
    return key if key == "language_selector" else f"ui_choice_{lang}_{key}"


@lru_cache(maxsize=8192)
def _translate_string(value, lang):
    key = _messages().get(value) or _messages().get(value + ".")
    if key:
        return tr(lang, key)
    if value in UI_FIELDS:
        return tr(lang, "ui_field_" + value)
    if value in UI_FIELD_ALIASES:
        return field_label(value, lang)
    if value in TRANSLATIONS.get(lang, {}):
        return tr(lang, value)
    for pattern, key, fields in _templates():
        match = pattern.fullmatch(value)
        if match:
            return tr(lang, key).format(**{field: translate_text(part, lang) for field, part in zip(fields, match.groups())})
    for code in ("missing_required_features", "nan_in_critical_columns", "duplicated_case_time_rows"):
        if value.startswith(code + ":"):
            return tr(lang, code).format(detail=value.split(":", 1)[1].strip())
    # Markdown/list wrappers and compound validator recommendations are presentation only.
    if value.startswith(("#### ", "### ", "## ", "# ")):
        prefix, body = value.split(" ", 1)
        return prefix + " " + translate_text(body, lang)
    if ": " in value:
        head, body = value.split(": ", 1)
        if head in _messages():
            return translate_text(head, lang) + ": " + translate_text(body, lang)
    if "; " in value:
        return "; ".join(translate_text(part, lang) for part in value.split("; "))
    if " | " in value:
        return " | ".join(translate_text(part, lang) for part in value.split(" | "))
    if ", " in value:
        parts = value.split(", ")
        if all(part in _messages() or part in UI_FIELDS or part in TRANSLATIONS.get(lang, {}) for part in parts):
            return "; ".join(translate_text(part, lang) for part in parts)
    match = re.fullmatch(r"(.+) (min|max)", value)
    if match:
        return field_label(match[1], lang) + " — " + tr(lang, "ui_" + match[2])
    if lang != "ru":
        value = value.replace("кр./сек", "крап./с" if lang == "uk" else "drops/s")
        if lang == "en":
            for source, target in (("мг/г", "mg/g"), ("л/мг", "L/mg"), ("кДж/моль", "kJ/mol")):
                value = value.replace(source, target)
        for unit, translated in (("мл/мин", "мл/хв" if lang == "uk" else "mL/min"), ("мг/л", "мг/л" if lang == "uk" else "mg/L"), ("г/см³", "г/см³" if lang == "uk" else "g/cm³"), ("кг/м³", "кг/м³" if lang == "uk" else "kg/m³"), ("см²/мин", "см²/хв" if lang == "uk" else "cm²/min"), ("1/мин", "1/хв" if lang == "uk" else "1/min"), ("см", "см" if lang == "uk" else "cm"), ("мин", "хв" if lang == "uk" else "min"), ("л/г", "л/г" if lang == "uk" else "L/g"), ("г", "г" if lang == "uk" else "g")):
            value = re.sub(r"(?<!\w)" + re.escape(unit) + r"(?!\w)", translated, value)
    return value


def display_dataframe(data, lang):
    """Return an independent view; never rename or round the caller's dataframe."""
    view = data.copy(deep=True)
    for name in view.columns:
        if pd.api.types.is_object_dtype(view[name]) or pd.api.types.is_string_dtype(view[name]):
            view[name] = view[name].map(lambda value: translate_text(value, lang))
            nonmissing = view[name].dropna()
            kinds = {"text" if isinstance(value, str) else "bool" if isinstance(value, (bool, np.bool_)) else "number" if isinstance(value, (int, float, np.number)) else "other" for value in nonmissing}
            if len(kinds) > 1:
                view[name] = view[name].map(lambda value: str(value) if value is not None else None)
    labels = [field_label(name, lang) for name in view.columns]
    # Distinct technical fields can share a label, e.g. train and test RMSE.
    if len(set(labels)) != len(labels):
        labels = [f"{label} ({index + 1})" if labels.count(label) > 1 else label for index, label in enumerate(labels)]
    view.columns = labels
    return view


def localize_figure(fig, lang):
    """Translate artists, not their data. Also translate already-created legends."""
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        artist.set_text(translate_text(artist.get_text(), lang))
    return fig


def figure_snapshot(fig):
    """Keep small plotting arrays, not figures or fitted models, across reruns."""
    axes = []
    for ax in fig.axes:
        lines = []
        for line in ax.lines:
            transform = "xaxis" if line.get_transform() == ax.get_xaxis_transform() else "yaxis" if line.get_transform() == ax.get_yaxis_transform() else "data"
            lines.append({"x": np.asarray(line.get_xdata()).copy(), "y": np.asarray(line.get_ydata()).copy(), "label": line.get_label(), "color": line.get_color(), "linestyle": line.get_linestyle(), "linewidth": line.get_linewidth(), "marker": line.get_marker(), "markersize": line.get_markersize(), "alpha": line.get_alpha(), "transform": transform})
        texts = [{"text": text.get_text(), "position": text.get_position(), "color": text.get_color(), "size": text.get_fontsize(), "verticalalignment": text.get_verticalalignment(), "axes_coords": text.get_transform() == ax.transAxes} for text in ax.texts]
        axes.append({"lines": lines, "texts": texts, "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(), "xlim": ax.get_xlim(), "ylim": ax.get_ylim()})
    return {"size": fig.get_size_inches().tolist(), "axes": axes}


def figure_from_snapshot(snapshot, lang):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(snapshot["axes"]), 1, figsize=snapshot["size"], squeeze=False)
    for ax, spec in zip(axes.flat, snapshot["axes"]):
        for saved in spec["lines"]:
            line = dict(saved)
            transform = line.pop("transform")
            x, y = line.pop("x"), line.pop("y")
            line["label"] = translate_text(line["label"], lang)
            if transform != "data":
                line["transform"] = ax.get_xaxis_transform() if transform == "xaxis" else ax.get_yaxis_transform()
            ax.plot(x, y, **line)
        for text in spec.get("texts", []):
            ax.text(*text["position"], translate_text(text["text"], lang), color=text["color"], fontsize=text["size"], verticalalignment=text["verticalalignment"], transform=ax.transAxes if text["axes_coords"] else ax.transData)
        ax.set_title(" | ".join(translate_text(part, lang) for part in spec["title"].split(" | ")))
        ax.set_xlabel(translate_text(spec["xlabel"], lang))
        ax.set_ylabel(translate_text(spec["ylabel"], lang))
        ax.set_xlim(spec["xlim"])
        ax.set_ylim(spec["ylim"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


class PresentationUI:
    """Thin rendering boundary around Streamlit; all computation APIs delegate."""
    _text_methods = {"title", "header", "subheader", "caption", "markdown", "text", "write", "info", "warning", "error", "success", "spinner", "expander", "status", "button", "download_button", "file_uploader", "metric"}
    _widgets = {"selectbox", "radio", "multiselect", "select_slider", "number_input", "slider", "checkbox", "toggle", "text_input"}

    def __init__(self, backend):
        self._backend = backend

    @property
    def language(self):
        return self._backend.session_state.get("language", "uk")

    def __getattr__(self, name):
        original = getattr(self._backend, name)
        if name == "tabs":
            return lambda labels, **kw: original([translate_text(label, self.language) for label in labels], **kw)
        if name == "pyplot":
            def render_figure(fig=None, **kw):
                return original(localize_figure(fig, self.language) if fig is not None else fig, **kw)
            return render_figure
        if name in {"dataframe", "table"}:
            def render_table(data, **kw):
                if not isinstance(data, pd.DataFrame):
                    return original(data, **kw)
                view = display_dataframe(data, self.language)
                config = kw.pop("column_config", None)
                config = {field_label(k, self.language): v for k, v in (config or {}).items()}
                for column in data:
                    canonical = UI_FIELD_ALIASES.get(str(column), str(column))
                    help_key = "ui_median_help" if "median" in canonical.lower() else "ui_percentile_help" if canonical in {"target_p90", "target_p95"} else "ui_cases_help" if canonical in {"number_of_rows", "number_of_cases"} else "ui_ebct_help" if canonical == "EBCT_min" else None
                    if help_key and field_label(column, self.language) not in config:
                        config[field_label(column, self.language)] = self._backend.column_config.Column(help=tr(self.language, help_key))
                if config:
                    kw["column_config"] = config
                order = kw.pop("column_order", None)
                if order:
                    kw["column_order"] = [field_label(k, self.language) for k in order]
                if name == "dataframe" and len(view.columns) > 12 and not kw.pop("ui_full_table", False):
                    primary = [c for c in ("experiment_id", "candidate_id", "case_id", "time_min", "metric", "value", "model", "RMSE_C0", "MAE_C0", "RMSE", "MAE", "R2", "decision_status", "reject_reason", "predicted_breakthrough_time_h", "carbon_mass_g", "carbon_mass_kg", "bed_height_cm", "height_cm", "column_diameter_cm", "diameter_cm", "flow_ml_min", "flow_ml_min_assumed", "C0_mg_L", "temperature_C", "EBCT_min", "target_max_h", "dataset_ready") if c in data]
                    primary += [c for c in data.columns if c not in primary]
                    if len(data) == 1:
                        compact = display_dataframe(pd.DataFrame({"metric": primary[:12], "value": [data.iloc[0][key] for key in primary[:12]]}), self.language)
                    else:
                        compact = display_dataframe(data[primary[:12]], self.language)
                    compact_kw = {k: v for k, v in kw.items() if k not in {"column_order", "column_config"}}
                    if len(data) == 1:
                        # Static key/value tables wrap labels in narrow calibration columns.
                        result = self._backend.table(compact.set_index(field_label("metric", self.language)))
                    else:
                        result = original(compact, **compact_kw)
                    with self._backend.expander(tr(self.language, "ui_details")):
                        original(view, **kw)
                    return result
                return original(view, **kw)
            return render_table
        if name == "data_editor":
            def edit(data, *args, **kw):
                config = dict(kw.pop("column_config", {}))
                for key in data.columns:
                    if key not in config:
                        config[key] = self._backend.column_config.Column(field_label(key, self.language))
                kw["column_config"] = config
                return original(data, *args, **kw)
            return edit
        if name in self._text_methods or name in self._widgets:
            def render(*args, **kw):
                lang = self.language
                args = list(args)
                if args:
                    args[0] = translate_text(args[0], lang)
                if "label" in kw:
                    kw["label"] = translate_text(kw["label"], lang)
                if name in {"success", "warning", "error"}:
                    prefix = tr(lang, f"ui_{name}_prefix")
                    if args and isinstance(args[0], str) and not args[0].startswith(prefix):
                        args[0] = prefix + args[0]
                if "help" in kw:
                    kw["help"] = translate_text(kw["help"], lang)
                elif name in self._widgets:
                    label = str(args[0] if args else kw.get("label", ""))
                    if "C₀" in label or "C0" in label:
                        kw["help"] = tr(self.language, "ui_ratio_help" if "C/C" in label or "C_breakthrough" in label else "ui_c0_help")
                    elif "EBCT" in label:
                        kw["help"] = tr(self.language, "ui_ebct_help")
                if name in {"selectbox", "radio", "multiselect", "select_slider"} and kw.get("key") != "language_selector":
                    formatter = kw.get("format_func", str)
                    options = args[1] if len(args) > 1 else kw.get("options", [])
                    labels = {str(value): translate_text(formatter(value), lang) for value in options}
                    kw["format_func"] = lambda value: labels.get(str(value), str(value))
                    key = kw.get("key")
                    if key is not None:
                        state = self._backend.session_state
                        widget_key = choice_widget_key(key, lang)
                        # Streamlit can retain stale selected labels when only format_func
                        # changes. Remount the localized widget, keep its value separately.
                        if key in state:
                            saved = state[key]
                            if name == "multiselect":
                                saved = [value for value in saved if value in options]
                                if widget_key not in state:
                                    kw["default"] = saved
                                elif state[widget_key] != saved:
                                    state[widget_key] = saved
                            elif saved in options:
                                if widget_key not in state:
                                    if name in {"selectbox", "radio"}:
                                        kw["index"] = list(options).index(saved)
                                    else:
                                        kw["value"] = saved
                                elif state[widget_key] != saved:
                                    state[widget_key] = saved
                        callback = kw.pop("on_change", None)
                        callback_args = kw.pop("args", ())
                        callback_kwargs = kw.pop("kwargs", {})

                        def sync_value():
                            state[key] = state[widget_key]
                            if callback:
                                callback(*(callback_args or ()), **(callback_kwargs or {}))

                        kw["key"] = widget_key
                        kw["on_change"] = sync_value
                        result = original(*args, **kw)
                        state[key] = result
                        return result
                return original(*args, **kw)
            return render
        return original
