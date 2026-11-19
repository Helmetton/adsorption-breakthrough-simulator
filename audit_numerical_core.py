"""Numerical checks with fixed acceptance criteria and Ukrainian figures."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import ColumnParams, SorptionParams, column_rhs, simulate_breakthrough, simulate_breakthrough_full, interpret_breakthrough_curve, column_area_from_diameter
from mass_balance import calculate_mass_balance
from scientific_provenance import current_provenance


def run_audit(output_dir="scientific_correction_results"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    column = ColumnParams(8.5, column_area_from_diameter(3.2), 18, .4, 3, 7.5640263963)
    random = np.random.default_rng(42)
    checks, grid_rows, balance_rows = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for dispersion in (0.0, .01, 1.0):
        params = SorptionParams(.05, .03, dispersion, .2, .001)
        n = 80
        state = random.uniform(0, 1, 2*n)
        derivative = column_rhs(0, state, column, params, n)
        volume_l = column.area_cm2*column.length_cm/1000
        storage_rate = (column.epsilon*volume_l*derivative[:n].sum() + column.mass_g*derivative[n:].sum())/n
        expected = column.flow_ml_min*(column.c0_mg_l-state[n-1])/1000
        checks.append({"check":"telescopic_mass_rate", "Dax":dispersion, "absolute_error_mg_min":abs(storage_rate-expected), "passed":bool(abs(storage_rate-expected)<1e-12)})
        errors = []
        for step in (.5, .25, .125):
            time = np.arange(5, 300+step/2, step)
            full = simulate_breakthrough_full(time, column, params, n_cells=n, rtol=1e-8, atol=1e-10)
            balance = calculate_mass_balance(time, full["C_profile_mg_L"], full["q_profile_mg_g"], full["C_out_mg_L"], column, params)
            error = float(balance.mass_balance_error_mg.abs().max())
            errors.append(error)
            balance_rows.append({"Dax":dispersion, "time_step_min":step, "max_error_mg":error, "final_relative_error_percent":balance.mass_balance_error_percent.iloc[-1]})
            if dispersion == .01 and step == .125:
                balance.to_csv(out/"mass_balance_nonzero_start.csv", index=False)
                axes[1].plot(time, balance.mass_balance_error_mg, label="Нев’язка, t_start=5 хв")
        checks.append({"check":"integral_mass_refinement", "Dax":dispersion, "passed":bool(errors[-1]<errors[0] and errors[-1]<1e-4), "errors_mg":errors})
        time = np.linspace(0, 300, 1201)
        curves = {n:simulate_breakthrough(time, column, params, n_cells=n, rtol=1e-8, atol=1e-10)/column.c0_mg_l for n in (80,160,320,640,1280)}
        bt = {n:interpret_breakthrough_curve(time, curve).breakthrough_time_min for n,curve in curves.items()}
        for n,m in ((80,160),(160,320),(320,640),(640,1280)):
            curve_error = float(np.max(np.abs(curves[n]-curves[m])))
            bt_error = abs(bt[n]-bt[m])
            grid_rows.append({"Dax":dispersion, "n_cells":n, "reference_cells":m, "max_abs_curve_difference":curve_error, "breakthrough_difference_min":bt_error,
                              "criterion_curve":.01, "criterion_time_min":1.0, "passed":curve_error<=.01 and bt_error<=1.0})
        if dispersion == .01:
            for n,curve in curves.items(): axes[0].plot(time,curve,label=f"N={n}")
    axes[0].set(xlabel="Час, хв", ylabel="C/C0", title="Згущення просторової сітки")
    axes[1].set(xlabel="Час, хв", ylabel="Нев’язка, мг", title="Баланс приросту запасу речовини")
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"numerical_checks.png",dpi=180)
    fig.savefig(out/"numerical_checks.svg")
    plt.close(fig)
    pd.DataFrame(grid_rows).to_csv(out/"spatial_convergence.csv",index=False)
    pd.DataFrame(balance_rows).to_csv(out/"mass_balance_refinement.csv",index=False)
    # Henry linearity with proportionally scaled initial state.
    p = SorptionParams(.05,.03,.01,.2,.001)
    from dataclasses import replace
    time = np.linspace(0,300,400)
    c1 = simulate_breakthrough(time,column,p)/column.c0_mg_l
    c2 = simulate_breakthrough(time,replace(column,c0_mg_l=2*column.c0_mg_l),replace(p,c_initial_mg_l=.4,q_initial_mg_g=.002))/(2*column.c0_mg_l)
    checks.append({"check":"normalized_C0_invariance", "max_error":float(np.max(abs(c1-c2))), "passed":bool(np.max(abs(c1-c2))<1e-5)})
    report={"provenance":current_provenance(), "checks":checks, "convergence":grid_rows,
            "grid_criterion_rationale":"Fixed absolute ratio error 0.01 and tb error 1 min, smaller than the typical 5 min sampling interval; this is a numerical budget, not a measured analytical uncertainty."}
    (out/"numerical_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=="__main__": run_audit()
