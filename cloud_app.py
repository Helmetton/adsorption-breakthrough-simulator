"""Public Streamlit entry point. Runs only bounded, synthetic simulations."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mass_balance import calculate_mass_balance, summarize_mass_balance
from model import (
    DEFAULT_BREAKTHROUGH_RATIO, DEFAULT_N_CELLS, ColumnParams, SorptionParams,
    column_area_from_diameter, interpret_breakthrough_curve, simulate_breakthrough_full,
)
from translations import tr


# Each request evaluates at most one fixed-size solve. No datasets, uploads, or disk writes.
INPUTS = (
    ("column_diameter_cm", 2.0, 5.0, 3.0, 0.1),
    ("bed_height_cm", 4.0, 15.0, 7.5, 0.5),
    ("flow_ml_min", 1.0, 12.0, 4.0, 0.5),
    ("c0_mg_l", 1.0, 10.0, 5.0, 0.5),
)
DENSITY_G_CM3 = 14.0 / (column_area_from_diameter(3.0) * 7.5)
TIMES_MIN = np.arange(0.0, 241.0, 1.0)
DEFAULT_INPUT = tuple(item[3] for item in INPUTS)


@st.cache_data(max_entries=64, ttl=1800, show_spinner=False)
def calculate(values):
    if len(values) != len(INPUTS) or not all(
        lower <= value <= upper for value, (_, lower, upper, _, _) in zip(values, INPUTS)
    ):
        raise ValueError("Inputs are outside the public demonstration bounds.")
    diameter, height, flow, c0 = values
    area = column_area_from_diameter(diameter)
    volume = area * height
    column = ColumnParams(height, area, DENSITY_G_CM3 * volume, 0.40, flow, c0)
    params = SorptionParams(0.035, 0.025, 0.035)
    solution = simulate_breakthrough_full(TIMES_MIN, column, params, n_cells=DEFAULT_N_CELLS)
    times = solution["time_min"]
    ratio = solution["C_out_C0"]
    crossing = interpret_breakthrough_curve(times, ratio, c0, DEFAULT_BREAKTHROUGH_RATIO)
    balance = calculate_mass_balance(times, solution["C_profile_mg_L"], solution["q_profile_mg_g"],
                                     solution["C_out_mg_L"], column, params)
    curve = pd.DataFrame({"time_min": times, "C_out_C0": ratio, "C_out_mg_dm3": solution["C_out_mg_L"]})
    return curve, crossing, column.mass_g, volume / flow, summarize_mass_balance(balance)


def main():
    st.set_page_config(page_title="Adsorption Breakthrough Simulator", layout="wide")
    lang = st.selectbox("Мова / Language", ["uk", "en"], key="demo_language",
                        format_func={"uk": "Українська", "en": "English"}.get)

    def label(uk, en):
        return uk if lang == "uk" else en

    st.title(label("Симулятор адсорбційної колони", "Adsorption Breakthrough Simulator"))
    st.caption(label("Метиленовий синій у воді · гранульоване активоване вугілля · синтетичний сценарій",
                     "Methylene blue in water · granular activated carbon · synthetic scenario"))
    controls, results = st.columns([1, 2], gap="large")
    with controls:
        st.subheader(label("Параметри колони", "Column inputs"))
        with st.form("demo_inputs"):
            values = tuple(st.number_input(tr(lang, name), lower, upper, default, step, key="demo_" + name)
                           for name, lower, upper, default, step in INPUTS)
            submit = st.form_submit_button(tr(lang, "run_simulation"), type="primary")
        with st.expander(label("Налаштування моделі", "Model assumptions")):
            st.write(f"Henry + LDF · BDF · {tr(lang, 'model_cells')}: {DEFAULT_N_CELLS}")
            st.markdown(
                "ε = 0.40  \n"
                "KH = 0.035 dm³/g  \n"
                "kLDF = 0.025 min⁻¹  \n"
                "Dax = 0.035 cm²/min"
            )
        st.link_button(label("Код і повна локальна версія", "Source and full local version"),
                       "https://github.com/Helmetton/adsorption-breakthrough-simulator")

    with results:
        st.subheader(label("Крива проскоку", "Breakthrough curve"))
        if submit or "demo_result" not in st.session_state:
            inputs = values if submit else DEFAULT_INPUT
            try:
                st.session_state["demo_result"] = calculate(inputs)
            except (RuntimeError, ValueError):
                st.session_state.pop("demo_result", None)
                st.error(label("Не вдалося розрахувати цей сценарій. Змініть параметри та повторіть.",
                               "This scenario could not be calculated. Adjust the inputs and try again."))
        if "demo_result" not in st.session_state:
            return
        curve, crossing, mass, ebct, balance = st.session_state["demo_result"]
        minute = label("хв", "min")
        time_label = label("Час проскоку", "Breakthrough time")
        a, b, c = st.columns(3)
        if crossing.is_exact:
            time_display = f"{crossing.breakthrough_time_min:.2f} {minute}"
        elif crossing.breakthrough_bound_type == "lower_bound":
            time_display = f"> {crossing.breakthrough_time_min:g} {minute}"
        else:
            time_display = label("Не визначено", "Not determined")
        a.metric(time_label, time_display)
        b.metric(label("Маса вугілля", "Carbon mass"), f"{mass:.2f} {label('г', 'g')}")
        c.metric("EBCT", f"{ebct:.2f} {minute}")

        chart = go.Figure(go.Scatter(x=curve.time_min, y=curve.C_out_C0, mode="lines",
                                    name="C/C₀", line={"color": "#16828a", "width": 3}))
        chart.add_hline(y=DEFAULT_BREAKTHROUGH_RATIO, line_dash="dash", line_color="#bf6d2a",
                        annotation_text="C/C₀ = 0.05", annotation_position="top right")
        if crossing.is_exact:
            chart.add_vline(x=crossing.breakthrough_time_min, line_dash="dot", line_color="#808080")
        chart.update_layout(xaxis_title=tr(lang, "time_min"), yaxis_title="C/C₀",
                            height=410, margin={"l": 12, "r": 14, "t": 16, "b": 12})
        st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
        st.caption(label("Похибка балансу маси в кінці розрахунку", "Final mass balance residual")
                   + f": {balance['final_balance_error_percent']:.4g}%")
        st.download_button(tr(lang, "download_csv"), curve.to_csv(index=False).encode("utf-8"),
                           file_name="synthetic_breakthrough.csv", mime="text/csv", on_click="ignore")
        st.caption(label("Показано розрахунок, а не вимірювання. Не для проєктних гарантій.",
                         "Simulated, not measured. Not a design guarantee."))


if __name__ == "__main__":
    main()
