
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import utils
from pathlib import Path



# ============================================================
#  CM-policy: stokastisk vs deterministisk
# ============================================================

def plot_model_type_objective_comparison(output_dict, det_output_dict, save_path="obj_comparison.png"):
    """
    Sammenligner objektivverdien mellom:
    - Fullt stokastisk løsning
    - Stokastisk modell med CM låst til deterministisk policy
    """
    model = output_dict["model"]
    det_model = det_output_dict["model"]


    obj_stoch = model.ObjVal
    obj_detcm = det_model.ObjVal
    diff = obj_stoch - obj_detcm

    data = pd.DataFrame({
        "Policy": [
            "Stochastic model",
            "Deterministic CM policy in stochastic model"
        ],
        "Objective value": [obj_stoch, obj_detcm]
    })

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=data,
        x="Policy",
        y="Objective value"
    )
    ax.set_title("Objective value comparison of stochastic vs deterministic CM policy")
    ax.set_ylabel("Objective value")
    ax.set_xlabel("")

    # Annoter søylene
    for i, v in enumerate(data["Objective value"]):
        ax.text(
            i,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # Tekstboks med differanse
    txt = f"Δ = {diff:.2f}"
    plt.gcf().text(0.5, -0.05, txt, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_model_type_policy_comparison(output_dict, det_output_dict, save_path):
    """
    Plotter to sidestilte figurer:
      - Stochastic optimal
      - Stochastic w/ deterministic CM

    For CM_up og CM_down vises:
      - Bid quantity x  [MW]
      - Activated quantity a [MW]
      - Bid price r     [NOK/MWh]

    x og a bruker venstre akse.
    r bruker høyre akse.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # ---- data ----
    x = output_dict["x"]
    a = output_dict["a"]
    r = output_dict["r"]

    det_x = det_output_dict["x"]
    det_a = det_output_dict["a"]
    det_r = det_output_dict["r"]

    U = output_dict["U"]
    M_u = output_dict["M_u"]

    u0 = next(iter(U))
    products = list(M_u)
    idx = np.arange(len(products))
    bar_width = 0.25

    # Fargepalett (colorblind-friendly)
    colors = sns.color_palette("colorblind", 3)
    c_x, c_a, c_r = colors

    def extract_vals(xv, av, rv):
        return (
            [float(xv[m, u0].X) for m in products],
            [float(av[m, u0].X) for m in products],
            [float(rv[m, u0].X) for m in products],
        )

    stoch_x, stoch_a, stoch_r = extract_vals(x, a, r)
    det_x_vals, det_a_vals, det_r_vals = extract_vals(det_x, det_a, det_r)

    # Felles maks for venstre akse
    all_qty = stoch_x + stoch_a + det_x_vals + det_a_vals
    max_qty = 1.1 * max(all_qty) if max(all_qty) > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    def draw_subplot(ax, qty_x, qty_a, price_r, title):
        # Venstre akse: x og a
        ax.bar(idx - bar_width, qty_x, width=bar_width, color=c_x,
               label="Bid quantity x [MW]", edgecolor="black")
        ax.bar(idx, qty_a, width=bar_width, color=c_a,
               label="Activated quantity a [MW]", edgecolor="black")

        ax.set_ylabel("Quantity [MW]", fontsize=10)
        ax.set_ylim(0, max_qty)
        ax.set_xticks(idx)
        ax.set_xticklabels(products, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

        # Høyre akse: r
        ax2 = ax.twinx()
        ax2.bar(idx + bar_width, price_r, width=bar_width,
                color=c_r, edgecolor="black", alpha=0.85,
                label="Bid price r [NOK/MWh]")
        ax2.set_ylabel("Price [NOK/MWh]", fontsize=10)

        return ax, ax2

    # Venstre plott – full stochastic
    ax_left, ax2_left = draw_subplot(
        axes[0], stoch_x, stoch_a, stoch_r,
        title="Stochastic soultion"
    )

    # Høyre plott – stochastic + deterministic CM
    ax_right, ax2_right = draw_subplot(
        axes[1], det_x_vals, det_a_vals, det_r_vals,
        title="Deterministic CM bids"
    )

    # --- Legend flyttes under tittelen ---
    handles1, labels1 = ax_left.get_legend_handles_labels()
    handles2, labels2 = ax2_left.get_legend_handles_labels()

    fig.legend(handles1 + handles2, labels1 + labels2,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.95),
               ncol=3,
               frameon=False,
               fontsize=9)

    # --- Layout ---
    fig.suptitle("Capacity market bidding decisions: Stochastic vs Deterministic CM policy",
                 fontsize=13, y=1.02)

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Objektivverdi: CM+DA+EAM vs kun DA og EAM
# ============================================================

def plot_market_attendance_objective_comparison(output_dict,
                                             da_eam_output_dict,
                                             save_path="fig_obj_full_vs_da_eam.png"):
    """
    Sammenligner objektivverdi mellom:
    - Full 3-markedsmodell (CM + DA + EAM)
    - Modell med kun DA og EAM inkludert
    """
    model = output_dict["model"]
    da_eam_model = da_eam_output_dict["model"]

    obj_full = model.ObjVal
    obj_da_eam = da_eam_model.ObjVal
    diff = obj_full - obj_da_eam

    data = pd.DataFrame({
        "Model": [
            "CM+DA+EAM",
            "DA+EAM"
        ],
        "Objective value": [obj_full, obj_da_eam]
    })

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=data,
        x="Model",
        y="Objective value"
    )
    ax.set_title("Objective value: CM+DA+EAM vs DA+EAM")
    ax.set_xlabel("")
    ax.set_ylabel("Objective value")

    # Annoter søylene
    for i, v in enumerate(data["Objective value"]):
        ax.text(
            i,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # Tekstboks med differanse
    txt = f"Δ = {diff:.2f}"
    plt.gcf().text(0.5, -0.08, txt, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Budstrategi: Full modell vs kun DA og EAM
# ============================================================

def summarize_policy_x(x,
                       U, V, W,
                       M_u, M_v, M_w,
                       model_label):
    """
    Oppsummerer gjennomsnittlig budmengde x per produkt
    for én modell (brukes til å sammenligne modeller).
    """

    rows = []

    # Hjelpe: full v- og w-mengde
    V_all = set().union(*V.values())
    W_all = set().union(*W.values())

    # For hvert produkt: finn relevante scenarier og ta gjennomsnitt av x
    for m in (M_u + M_v + M_w):

        if m in M_u:
            indices = [(m, u) for u in U if (m, u) in x]
        elif m in M_v:
            indices = [(m, v) for v in V_all if (m, v) in x]
        else:  # EAM
            indices = [(m, w) for w in W_all if (m, w) in x]

        if not indices:
            # Produktet finnes ikke i denne modellen (f.eks. CM i DA+EAM-modellen)
            continue

        x_vals = [x[key].X for key in indices]
        avg_x = float(np.mean(x_vals))

        rows.append({
            "Product": m,
            "Model": model_label,
            "Quantity": avg_x
        })

    return rows

def summarize_policy_r(r,
                       U, V, W,
                       M_u, M_v, M_w,
                       model_label):
    """
    Oppsummerer gjennomsnittlig budpris r per produkt
    for én modell (brukes til å sammenligne modeller).
    """

    rows = []

    # Hjelpe: full v- og w-mengde
    V_all = set().union(*V.values())
    W_all = set().union(*W.values())

    # For hvert produkt: finn relevante scenarier og ta gjennomsnitt av r
    for m in (M_u + M_v + M_w):

        if m in M_u:
            indices = [(m, u) for u in U if (m, u) in r]
        elif m in M_v:
            indices = [(m, v) for v in V_all if (m, v) in r]
        else:  # EAM
            indices = [(m, w) for w in W_all if (m, w) in r]

        if not indices:
            continue  # produkt finnes ikke i denne modellen

        r_vals = [r[key].X for key in indices]
        avg_r = float(np.mean(r_vals))

        rows.append({
            "Product": m,
            "Model": model_label,
            "Price": avg_r
        })

    return rows


def plot_market_attendance_expected_x(output_dict, da_eam_output_dict, save_path):
    """
    Sammenligner gjennomsnittlig budmengde x per produkt mellom:
    - Full modell: CM+DA+EAM
    - Modell: DA+EAM

    Figuren har én akse, med to søyler per produkt (én per modell).
    """
    x = output_dict["x"]
    da_eam_x = da_eam_output_dict["x"]
    U = output_dict["U"]
    V = output_dict["V"]
    W = output_dict["W"]
    M_u = output_dict["M_u"]
    M_v = output_dict["M_v"]
    M_w = output_dict["M_w"]

    rows_full = summarize_policy_x(
        x,
        U, V, W,
        M_u, M_v, M_w,
        model_label="CM+DA+EAM"
    )

    rows_da_eam = summarize_policy_x(
        da_eam_x,
        U, V, W,
        M_u, M_v, M_w,
        model_label="DA+EAM"
    )

    df = pd.DataFrame(rows_full + rows_da_eam)

    # Ryddig rekkefølge på produktene
    product_order = M_u + M_v + M_w
    df["Product"] = pd.Categorical(df["Product"],
                                   categories=product_order,
                                   ordered=True)
    df["Model"] = pd.Categorical(df["Model"],
                                 categories=["CM+DA+EAM", "DA+EAM"],
                                 ordered=True)

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=df,
        x="Product",
        y="Quantity",
        hue="Model"
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mean bid quantity E[x] [MW]", fontsize=11)
    ax.set_title("Mean bid quantity per market product", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="x", rotation=15)

    plt.legend(title="", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_market_attendance_expected_r(output_dict, da_eam_output_dict, save_path):
    """
    Sammenligner gjennomsnittlig budpris r per produkt mellom:
    - Full modell: CM+DA+EAM
    - Modell: DA+EAM

    Figuren har én akse, med to søyler per produkt (én per modell).
    """

    r = output_dict["r"]
    da_eam_r = da_eam_output_dict["r"]
    U = output_dict["U"]
    V = output_dict["V"]
    W = output_dict["W"]
    M_u = output_dict["M_u"]
    M_v = output_dict["M_v"]
    M_w = output_dict["M_w"]

    rows_full = summarize_policy_r(
        r,
        U, V, W,
        M_u, M_v, M_w,
        model_label="CM+DA+EAM"
    )

    rows_da_eam = summarize_policy_r(
        da_eam_r,
        U, V, W,
        M_u, M_v, M_w,
        model_label="DA+EAM"
    )

    df = pd.DataFrame(rows_full + rows_da_eam)

    # Ryddig rekkefølge på produktene
    product_order = M_u + M_v + M_w
    df["Product"] = pd.Categorical(df["Product"],
                                   categories=product_order,
                                   ordered=True)
    df["Model"] = pd.Categorical(df["Model"],
                                 categories=["CM+DA+EAM", "DA+EAM"],
                                 ordered=True)

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=df,
        x="Product",
        y="Price",
        hue="Model"
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mean bid price E[r] [NOK/MW]", fontsize=11)
    ax.set_title("Mean bid price per market product", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="x", rotation=15)

    plt.legend(title="", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()




# ============================================================
#  Forventet volum per marked (CM, DA, EAM)
# ============================================================

def plot_expected_a(output_dict_1, output_dict_2, label1, label2, save_path):
    """
    Plotter forventet aktivert volum E[a] per markedsprodukt for to policies:
    - (model1, a1) med navn label1
    - (model2, a2) med navn label2

    Resultatet er en grouped bar chart med:
    - x-akse: markedsprodukt (CM_up, CM_down, DA, EAM_up, EAM_down)
    - y-akse: forventet aktivert volum
    - hue: policy (f.eks. 'Stochastic', 'Stochastic + det. CM')
    """
    model1 = output_dict_1["model"]
    model2 = output_dict_2["model"]
    a1 = output_dict_1["a"]
    a2 = output_dict_2["a"]
    U = output_dict_1["U"]
    V = output_dict_1["V"]
    W = output_dict_1["W"]
    M_u = output_dict_1["M_u"]
    M_v = output_dict_1["M_v"]
    M_w = output_dict_1["M_w"]

    df1 = utils.compute_expected_volumes(model1, a1, U, V, W, M_u, M_v, M_w, label1)
    df2 = utils.compute_expected_volumes(model2, a2, U, V, W, M_u, M_v, M_w, label2)

    df = pd.concat([df1, df2], ignore_index=True)

    # Behold naturlig produktrekkefølge
    product_order = M_u + M_v + M_w
    df["Market product"] = pd.Categorical(df["Market product"],
                                          categories=product_order,
                                          ordered=True)

    # Figur – ryddig, artikkelvennlig
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(
        data=df,
        x="Market product",
        y="Expected activated volume [MW]",
        hue="Policy"
    )

    ax.set_title("Expected activated volume per market product", fontsize=12)
    ax.set_ylabel("Expected activated volume E[a] [MW]", fontsize=11)
    ax.set_xlabel("")
    ax.legend(title="", fontsize=9)
    plt.xticks(rotation=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()








#
#Funksjonskall
#


# -----------------------------------------------------------------
# Deterministic CM policy vs Stochastic soulution
# -----------------------------------------------------------------

def generate_deterministic_policy_plots(output_dict, det_output_dict):
    """
    Genererer plots som sammenligner den stokastiske løsningen med løsningen
    der en deterministisk CM-policy er påtvunget.
    """
    # objektivverdi
    plot_model_type_objective_comparison(
        output_dict,
        det_output_dict,
        save_path="fig_model_type_obj.png"
    )

    # CM-beslutninger (x, a, r)
    plot_model_type_policy_comparison(
        output_dict,
        det_output_dict,
        save_path="fig_model_type_policy.png"
    )

    # --- Forventet volum per marked Stochastic vs deterministic CM-policy ---
    plot_expected_a(
        output_dict,
        det_output_dict,
        label1="Stochastic solution", 
        label2="Deterministic CM policy",
        save_path="fig_model_type_expected_a.png"
    )



# -----------------------------------------------------------------
# CM+DA+EAM vs DA+EAM only
# -----------------------------------------------------------------

def generate_da_eam_comparison_plots(output_dict, da_eam_output_dict):
    # Objektivverdi-plot
    plot_market_attendance_objective_comparison(
        output_dict,
        da_eam_output_dict,
        save_path="fig_market_attendance_obj_comparison.png"
    )

    # Budstrategi-plot per marked
    plot_market_attendance_expected_x(
        output_dict,
        da_eam_output_dict,
        save_path="fig_market_attendance_expected_x.png"
    )

    # Budstrategi-plot per marked
    plot_market_attendance_expected_r(
        output_dict, 
        da_eam_output_dict,
        save_path="fig_market_attendance_expected_r.png"
    )


    # --- Forventet volum per marked Stochastic vs DA+EAM-only ---
    plot_expected_a(
        output_dict, 
        da_eam_output_dict,
        label1="Strategy: CM+DA+EAM",
        label2="Strategy: DA+EAM",
        save_path="fig_market_attendance_expected_a.png"
    )

def plot_avg_dayahead_first_forecasts(path_data: str):
    """
    For each prediction_for timestamp, pick the row with the *oldest* created_at
    (i.e. the first time that hour was forecasted), average across scenario
    columns, and plot the resulting time series.

    This way, you get a curve from 2025-10-04 to 2025-10-14 (or whatever range
    exists), instead of only the very first forecast run.
    """

    # 1) Load data
    df = pd.read_parquet(path_data)

    # 2) Ensure datetimes
    df["prediction_for"] = pd.to_datetime(df["prediction_for"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # 3) Identify scenario columns (0, 1, ..., 49)
    scenario_cols = [c for c in df.columns if c.isdigit()]
    if not scenario_cols:
        raise ValueError("No scenario columns (0..49) found in dataframe.")

    # 4) Compute average over scenarios row-wise
    df["avg_scenario_price"] = df[scenario_cols].mean(axis=1)

    # 5) For each prediction_for, keep the row with the *earliest* created_at
    #    (first forecast for that hour)
    df_sorted = df.sort_values(["prediction_for", "created_at"])
    idx_first = df_sorted.groupby("prediction_for")["created_at"].idxmin()
    df_first = df_sorted.loc[idx_first].sort_values("prediction_for")

    # 6) Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df_first["prediction_for"], df_first["avg_scenario_price"])
    plt.title("Average dayahead forecasted prices per hour, for the whole period")
    plt.xlabel("Prediction time (prediction_for)")
    plt.ylabel("Average forecast price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_first


def plot_dayahead_forecast_std(path_data: str):
    """
    Computes the standard deviation across the 50 forecast scenarios (0..49)
    for each prediction_for timestamp, using the *earliest* created_at
    forecast available for that hour.

    Returns a dataframe with:
        prediction_for, scenario_std
    """

    # 1) Load parquet
    df = pd.read_parquet(path_data)

    # 2) Ensure time columns are datetime
    df["prediction_for"] = pd.to_datetime(df["prediction_for"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # 3) Identify scenario columns ("0" ... "49")
    scenario_cols = [c for c in df.columns if c.isdigit()]
    if not scenario_cols:
        raise ValueError("No scenario columns found (expected 0..49).")

    # 4) Sort, then select the earliest created_at for each prediction hour
    df_sorted = df.sort_values(["prediction_for", "created_at"])
    idx_first = df_sorted.groupby("prediction_for")["created_at"].idxmin()
    df_first = df_sorted.loc[idx_first].sort_values("prediction_for")

    # 5) Compute standard deviation across scenarios
    df_first["scenario_std"] = df_first[scenario_cols].std(axis=1)

    # 6) Plot the standard deviation
    plt.figure(figsize=(12, 5))
    plt.plot(df_first["prediction_for"], df_first["scenario_std"])
    plt.title("Forecast Scenario Standard Deviation per Hour)")
    plt.xlabel("Prediction Time")
    plt.ylabel("Scenario Standard Deviation (€/MWh)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df_first[["prediction_for", "scenario_std"]]


def compute_first_forecast_stats(path_data: Path, market: str) -> pd.DataFrame:
    """
    Computes average and std of forecast scenarios using ONLY
    the earliest created_at for each prediction_for timestamp.

    Returns DataFrame with:
        prediction_for, avg_price, scenario_std
    """

    df = pd.read_parquet(path_data)

    # Datetime fields
    df["prediction_for"] = pd.to_datetime(df["prediction_for"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # Scenario columns: "0"..."49"
    scenario_cols = [c for c in df.columns if c.isdigit()]
    if not scenario_cols:
        raise ValueError(f"No scenario columns detected for market {market}")

    # Sort so earliest created_at per prediction_for is first
    df_sorted = df.sort_values(["prediction_for", "created_at"])

    # Select earliest forecast per hour
    idx_first = df_sorted.groupby("prediction_for")["created_at"].idxmin()
    df_first = df_sorted.loc[idx_first].sort_values("prediction_for")

    # Average price across scenarios
    df_first["avg_price"] = df_first[scenario_cols].mean(axis=1)

    # Standard deviation across scenarios
    df_first["scenario_std"] = df_first[scenario_cols].std(axis=1)

    return df_first[["prediction_for", "avg_price", "scenario_std"]]


def plot_and_save(df: pd.DataFrame, market: str, out_dir: Path):
    """
    Saves two plots:
      - fig_average_prices_{market}.png
      - fig_volatility_prices_{market}.png
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- AVERAGE PRICE ----------
    fig_avg, ax_avg = plt.subplots(figsize=(12, 5))
    ax_avg.plot(df["prediction_for"], df["avg_price"])
    ax_avg.set_title(f"Average forecast prices per hour – {market}")
    ax_avg.set_xlabel("Prediction time")
    ax_avg.set_ylabel("Average forecast price")
    ax_avg.grid(True)
    fig_avg.tight_layout()
    fig_avg.savefig(out_dir / f"fig_average_prices_{market}.png", dpi=300)
    plt.close(fig_avg)

    # -------- VOLATILITY (STD) ----------
    fig_std, ax_std = plt.subplots(figsize=(12, 5))
    ax_std.plot(df["prediction_for"], df["scenario_std"])
    ax_std.set_title(f"Forecast price volatility (std) per hour – {market}")
    ax_std.set_xlabel("Prediction time")
    ax_std.set_ylabel("Standard deviation of forecast prices")
    ax_std.grid(True)
    fig_std.tight_layout()
    fig_std.savefig(out_dir / f"fig_volatility_prices_{market}.png", dpi=300)
    plt.close(fig_std)


def process_all_markets(file_mapping: dict, base_path: Path, fig_out_dir: Path):
    """
    Loop through all parquet files and produce:
      - avg forecast plot
      - std forecast plot
    using earliest created_at only.
    """

    for filename, market in file_mapping.items():
        path_data = base_path / filename

        if not path_data.exists():
            print(f"WARNING: {path_data} does not exist. Skipping.")
            continue

        print(f"Processing {market}...")

        df_stats = compute_first_forecast_stats(path_data, market)
        plot_and_save(df_stats, market, fig_out_dir)

        print(f"Saved figures for {market}\n")

def plot_scenarios_one_hour(prediction_time_str: str, area: str, base_path: Path | None = None, frequency: str | None = None):
    """
    Illustrate the 50 scenario values from all markets (day-ahead, eam up/down,
    cm up/down) for one production hour (prediction_for).

    For each market:
      - Filter to given prediction_for
      - (Optionally) filter by area/frequency
      - Use the earliest created_at for that prediction_for
      - Extract scenario columns 0..49
    Then:
      - Plot a boxplot with one box per market
    """

    # Parse the wanted hour
    target_time = pd.to_datetime(prediction_time_str, utc=True)

    market_values = {}  # market -> 1D array of 50 values

    for filename, market in file_mapping.items():
        path_data = base_path / filename
        if not path_data.exists():
            print(f"WARNING: {path_data} not found, skipping {market}.")
            continue

        df = pd.read_parquet(path_data)

        # Ensure datetime types
        df["prediction_for"] = pd.to_datetime(df["prediction_for"], utc=True)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

        # Optional filters: area and frequency
        if area is not None and "area" in df.columns:
            df = df[df["area"] == area]
        if frequency is not None and "frequency" in df.columns:
            df = df[df["frequency"] == frequency]

        # Filter to the chosen production hour
        df_hour = df[df["prediction_for"] == target_time]
        if df_hour.empty:
            print(f"No data for {market} at {target_time} (after filters). Skipping.")
            continue

        # Use the earliest created_at for that hour
        df_hour = df_hour.sort_values("created_at")
        first_row = df_hour.iloc[0]

        # Scenario columns: "0".."49"
        scenario_cols = [c for c in df.columns if c.isdigit()]
        if not scenario_cols:
            print(f"No scenario columns in {market}, skipping.")
            continue

        values = first_row[scenario_cols].astype(float).values
        market_values[market] = values

    if not market_values:
        print("No markets had data for the chosen hour with the given filters.")
        return

    # ---- Plot: one box per market ----
    markets = list(market_values.keys())
    data = [market_values[m] for m in markets]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=markets, showmeans=True)
    plt.title(
        f"Scenario price distributions for one production hour\n"
        f"prediction_for = {target_time}"
        + (f", area={area}" if area else "")
        + (f", freq={frequency}" if frequency else "")
    )
    plt.ylabel("Scenario price")
    plt.xlabel("Market")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Optional: return the raw data in long form if you want to inspect later
    long_df = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "market": m,
                        "scenario": range(len(vals)),
                        "value": vals,
                        "prediction_for": target_time,
                    }
                )
                for m, vals in market_values.items()
            ],
            ignore_index=True,
        )
    )
    return long_df

if __name__ == "__main__":
    file_mapping = {
    "dayahead_forecasts.parquet": "dayahead_forecasts",
    "mfrr_cm_down_forecasts.parquet": "mfrr_cm_down_forecasts",
    "mfrr_cm_up_forecasts.parquet": "mfrr_cm_up_forecasts",
    "mfrr_eam_down_forecasts.parquet": "mfrr_eam_down_forecasts",
    "mfrr_eam_up_forecasts.parquet": "mfrr_eam_up_forecasts",
    "production_forecasts.parquet": "production_forecasts",
}

    BASE_PATH = Path("data/raw")
    FIG_DIR = Path("figures")
    process_all_markets(file_mapping, BASE_PATH, FIG_DIR)