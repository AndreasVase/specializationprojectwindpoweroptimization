
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



# ============================================================
#  CM-policy: stokastisk vs deterministisk
# ============================================================


def plot_objective_comparison(model, det_model, save_path="obj_comparison.png"):
    """
    Sammenligner objektivverdien mellom:
    - Fullt stokastisk løsning
    - Stokastisk modell med CM låst til deterministisk policy
    """
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


def plot_cm_policy_comparison(model,
                              x, a, r, delta,
                              det_model,
                              det_x, det_a, det_r, det_delta,
                              U, M_u,
                              save_path="cm_policy_comparison.png"):
    """
    Sammenligner CM-beslutningen (x, a, r) mellom:
    - Fullt stokastisk løsning
    - Stokastisk modell med CM låst til deterministisk policy

    Vi bruker non-anticipativity: samme CM-bid for alle u,
    så vi velger én representativ u0.
    """

    # Representativ stage-2 node
    u0 = next(iter(U))

    rows = []
    for m in M_u:
        # Stokastisk optimal
        rows.append({
            "Product": m,
            "Variable": "Bid quantity x",
            "Policy": "Stochastic optimal",
            "Value": float(x[m, u0].X)
        })
        rows.append({
            "Product": m,
            "Variable": "Activated quantity a",
            "Policy": "Stochastic optimal",
            "Value": float(a[m, u0].X)
        })
        rows.append({
            "Product": m,
            "Variable": "Bid price r",
            "Policy": "Stochastic optimal",
            "Value": float(r[m, u0].X)
        })

        # Stokastisk modell med deterministisk CM-policy
        rows.append({
            "Product": m,
            "Variable": "Bid quantity x",
            "Policy": "Stochastic w/ deterministic CM",
            "Value": float(det_x[m, u0].X)
        })
        rows.append({
            "Product": m,
            "Variable": "Activated quantity a",
            "Policy": "Stochastic w/ deterministic CM",
            "Value": float(det_a[m, u0].X)
        })
        rows.append({
            "Product": m,
            "Variable": "Bid price r",
            "Policy": "Stochastic w/ deterministic CM",
            "Value": float(det_r[m, u0].X)
        })

    df = pd.DataFrame(rows)

    # Pent facet-plot: én kolonne per variabeltype (x, a, r)
    g = sns.catplot(
        data=df,
        x="Product",
        y="Value",
        hue="Policy",
        col="Variable",
        kind="bar",
        sharey=False,
        height=4,
        aspect=0.9
    )

    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle("CM bidding decision: stochastic vs deterministic CM policy")

    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
#  Forventet volum per marked (CM, DA, EAM)
# ============================================================



def plot_expected_market_volumes_with_stochastic_model(model, a, U, V, W, M_u, M_v, M_w,
                                 save_path="expected_market_volumes.png"):
    """
    Plotter forventet aktivert volum (E[a]) for hver markedsprodukt:
    - Stage 2: CM_up, CM_down
    - Stage 3: DA
    - Stage 4: EAM_up, EAM_down
    """

    scenario_tree = model._scenario_tree
    nodes = scenario_tree["nodes"]

    # ---------- Hjelpefunksjoner for sannsynligheter ----------
    # π_u
    def pi_u(u):
        return nodes[u].cond_prob

    # π_{v|u}
    def pi_v_given_u(v):
        return nodes[v].cond_prob

    # π_{w|v}
    def pi_w_given_v(w):
        return nodes[w].cond_prob

    # ---------- Forventede volum per produkt ----------
    exp_vol = {}

    # Stage 2: CM (avhengig av u)
    for m in M_u:
        val = 0.0
        for u in U:
            val += pi_u(u) * a[m, u].X
        exp_vol[m] = val

    # Stage 3: DA (avhengig av v, strukturert via u -> v)
    for m in M_v:   # bare "DA"
        val = 0.0
        for u in U:
            for v in V[u]:
                val += pi_u(u) * pi_v_given_u(v) * a[m, v].X
        exp_vol[m] = val

    # Stage 4: EAM (avhengig av w, strukturert via u -> v -> w)
    for m in M_w:
        val = 0.0
        for u in U:
            for v in V[u]:
                for w in W[v]:
                    val += pi_u(u) * pi_v_given_u(v) * pi_w_given_v(w) * a[m, w].X
        exp_vol[m] = val

    # ---------- Bygg DataFrame for plotting ----------
    data = []
    for m, val in exp_vol.items():
        if m in M_u:
            stage = "Stage 2 – CM"
        elif m in M_v:
            stage = "Stage 3 – DA"
        else:
            stage = "Stage 4 – EAM"

        data.append({
            "Market product": m,
            "Stage": stage,
            "Expected activated volume [MW]": val
        })

    df = pd.DataFrame(data)

    # ---------- Plot med seaborn ----------
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="Market product",
        y="Expected activated volume [MW]",
        hue="Stage"
    )

    ax.set_title("Expected activated volume per market product")
    ax.set_ylabel("E[a] [MW]")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_expected_market_volumes_with_deterministic_CM_policy(det_model, det_a, U, V, W, M_u, M_v, M_w,
                                 save_path="expected_market_volumes.png"):
    """
    Plotter forventet aktivert volum (E[a]) for hver markedsprodukt:
    - Stage 2: CM_up, CM_down
    - Stage 3: DA
    - Stage 4: EAM_up, EAM_down
    """

    scenario_tree = det_model._scenario_tree
    nodes = scenario_tree["nodes"]

    # ---------- Hjelpefunksjoner for sannsynligheter ----------
    # π_u
    def pi_u(u):
        return nodes[u].cond_prob

    # π_{v|u}
    def pi_v_given_u(v):
        return nodes[v].cond_prob

    # π_{w|v}
    def pi_w_given_v(w):
        return nodes[w].cond_prob

    # ---------- Forventede volum per produkt ----------
    exp_vol = {}

    # Stage 2: CM (avhengig av u)
    for m in M_u:
        val = 0.0
        for u in U:
            val += pi_u(u) * det_a[m, u].X
        exp_vol[m] = val

    # Stage 3: DA (avhengig av v, strukturert via u -> v)
    for m in M_v:   # bare "DA"
        val = 0.0
        for u in U:
            for v in V[u]:
                val += pi_u(u) * pi_v_given_u(v) * det_a[m, v].X
        exp_vol[m] = val

    # Stage 4: EAM (avhengig av w, strukturert via u -> v -> w)
    for m in M_w:
        val = 0.0
        for u in U:
            for v in V[u]:
                for w in W[v]:
                    val += pi_u(u) * pi_v_given_u(v) * pi_w_given_v(w) * det_a[m, w].X
        exp_vol[m] = val

    # ---------- Bygg DataFrame for plotting ----------
    data = []
    for m, val in exp_vol.items():
        if m in M_u:
            stage = "Stage 2 – CM"
        elif m in M_v:
            stage = "Stage 3 – DA"
        else:
            stage = "Stage 4 – EAM"

        data.append({
            "Market product": m,
            "Stage": stage,
            "Expected activated volume [MW]": val
        })

    df = pd.DataFrame(data)

    # ---------- Plot med seaborn ----------
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="Market product",
        y="Expected activated volume [MW]",
        hue="Stage"
    )

    ax.set_title("Expected activated volume per market product")
    ax.set_ylabel("E[a] [MW]")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()