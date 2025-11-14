
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import utils



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
            "Stochastic optimal",
            "Stochastic w/ deterministic CM"
        ],
        "Objective value": [obj_stoch, obj_detcm]
    })

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=data,
        x="Policy",
        y="Objective value"
    )
    ax.set_title("Objective value comparison")
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
    txt = f"Δ = {diff:.2f} (Stochastic − Det.CM)"
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
        title="Stochastic optimal CM bids"
    )

    # Høyre plott – stochastic + deterministic CM
    ax_right, ax2_right = draw_subplot(
        axes[1], det_x_vals, det_a_vals, det_r_vals,
        title="Stochastic model w/ deterministic CM"
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
    fig.suptitle("Capacity market bidding decision per product",
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

