from gurobipy import GRB
import pandas as pd
import numpy as np

def wind_speed_to_production_capacity(wind_speed):
    prod_cap = 10 * wind_speed  # enkel mapping som eksempel
    return prod_cap


def build_cost_coefficients(P):
    return



def build_price_parameter(tree):
    """
    Lager P_ms som dictionary:
        P[(m, s)] = clearing price for product m in scenario s
    """

    nodes = tree["nodes"]

    P = {}  # (m, s) -> value

    for s, node in nodes.items():

        # --- Stage 2: CM prices ---
        if node.stage == 2:
            if "CM_up" in node.info:
                P[("CM_up", s)] = node.info["CM_up"]
            if "CM_down" in node.info:
                P[("CM_down", s)] = node.info["CM_down"]

        # --- Stage 3: DA price ---
        elif node.stage == 3:
            if "DA" in node.info:
                P[("DA", s)] = node.info["DA"]

        # --- Stage 4: EAM + wind ---
        elif node.stage == 4:
            if "EAM_up" in node.info:
                P[("EAM_up", s)] = node.info["EAM_up"]
            if "EAM_down" in node.info:
                P[("EAM_down", s)] = node.info["EAM_down"]

    return P

def build_production_capacity(tree):
    """
    Lager Q_w for alle terminale scenarier w i stage 4.
    Q_w baseres på node.info["wind_speed"].

    Returnerer:
        Q[w] = production capacity in scenario w
    """

    nodes = tree["nodes"]
    Q = {}

    for w, node in nodes.items():
        if node.stage == 4:
            wind = node.info["wind_speed"]

            prod_cap = wind_speed_to_production_capacity(wind)
            Q[w] = prod_cap

    return Q



def sort_nodes(node_set):
    """Sorter noder som 'v1', 'v2', ..., 'v10' i numerisk rekkefølge."""
    def node_key(s):
        try:
            return int(s[1:])  # antar format 'v1', 'w12' etc.
        except ValueError:
            return s
    return sorted(node_set, key=node_key)



def print_results(model, x, r, a, delta, d, Q,
                  U, V, W, M1, M2, M3,
                  max_u=3, max_v_per_u=3, max_w_per_v=3):
    """
    Skriver ut en komprimert oversikt over løsningen.

    max_u         : maks antall u-noder (stage 2) å skrive ut (None = alle)
    max_v_per_u   : maks antall v-noder per u (stage 3)
    max_w_per_v   : maks antall w-noder per v (stage 4)
    """

    if model.Status != GRB.OPTIMAL:
        print("Model not solved to optimality. Status:", model.Status)
        return

    print("\n======================")
    print("   OPTIMAL SOLUTION")
    print("======================\n")

    print(f"Objective value: {model.ObjVal:,.4f}\n")

    # Sorter noder for ryddig utskrift
    U_sorted = sort_nodes(U)
    if max_u is not None:
        U_sorted = U_sorted[:max_u]

    # ---------- Stage 2: CM ----------
    print("--- Stage 2 (CM) – non-anticipative across u ---")
    for u in U_sorted:
        print(f"u = {u}:")
        for m in M1:
            print(
                f"  {m}: x={x[m,u].X:.3f}, "
                f"a={a[m,u].X:.3f}, "
                f"r={r[m,u].X:.3f}, "
                f"δ={int(round(delta[m,u].X))}"
            )
        print()
    print()

    # Vi lagrer hvilke v-er vi skriver ut per u, så vi kan bruke de samme i stage 4
    V_samples = {}

    # ---------- Stage 3: DA ----------
    print("--- Stage 3 (DA) – per u and v ---")
    for u in U_sorted:
        V_u_sorted = sort_nodes(V[u])
        V_sample = V_u_sorted[:max_v_per_u]
        V_samples[u] = V_sample

        print(f"\nParent CM node u = {u}:")
        for v in V_sample:
            for m in M2:
                print(
                    f"  {m} in {v}: "
                    f"x={x[m,v].X:.3f}, "
                    f"a={a[m,v].X:.3f}, "
                    f"r={r[m,v].X:.3f}, "
                    f"δ={int(round(delta[m,v].X))}"
                )
    print()

    # ---------- Stage 4: EAM ----------
    print("--- Stage 4 (EAM) – child w scenarios per v ---")
    for u in U_sorted:
        for v in V_samples[u]:
            W_v_sorted = sort_nodes(W[v])
            W_sample = W_v_sorted[:max_w_per_v]

            print(f"\nParent scenario v = {v} (from u = {u}):")
            for w in W_sample:
                for m in M3:
                    print(
                        f"  {m} in {w}: "
                        f"x={x[m,w].X:.3f}, "
                        f"a={a[m,w].X:.3f}, "
                        f"r={r[m,w].X:.3f}, "
                        f"δ={int(round(delta[m,w].X))}, "
                        f"d={d[m,w].X:.3f}, "
                        f"Q={Q[w]:.3f}"
                    )
    print()

    print("=============================================")
    print("            END OF RESULTS")
    print("=============================================\n")




def print_results_deterministic_policy(
    model, x, a, r, delta, d, Q, U, V, W, M_u, M_v, M_w
):
    num_v = 3   # hvor mange v-scenarier å vise
    num_w = 2   # hvor mange w per v

    print("\n======================================")
    print("     EXPECTED VALUE OF POLICY (EVP)")
    print("======================================\n")

    print(f"Objective value: {model.ObjVal:,.4f}\n")

    # Stage 2
    print("Stage 2 (CM) — deterministic policy:")
    u0 = sorted(U)[0]
    for m in M_u:
        print(
            f"  {m}: x={x[m,u0].X:.3f}, "
            f"r={r[m,u0].X:.3f}, "
            f"a={a[m,u0].X:.3f}, "
            f"δ={int(delta[m,u0].X)}"
        )

    # Stage 3
    print("\nStage 3 (DA) — representative v nodes:")
    V_all_sorted = sorted(set().union(*V.values()))
    V_subset = V_all_sorted[:num_v]

    for v in V_subset:
        for m in M_v:
            print(
                f"  v={v}, {m}: x={x[m,v].X:.3f}, "
                f"r={r[m,v].X:.3f}, "
                f"a={a[m,v].X:.3f}, "
                f"δ={int(delta[m,v].X)}"
            )

    # Stage 4
    print("\nStage 4 (EAM) — representative w children:")
    for v in V_subset:
        print(f"\n  Parent v={v}:")
        W_subset = sorted(W[v])[:num_w]

        for w in W_subset:
            for m in M_w:
                print(
                    f"    w={w}, {m}: "
                    f"x={x[m,w].X:.3f}, "
                    f"a={a[m,w].X:.3f}, "
                    f"r={r[m,w].X:.3f}, "
                    f"δ={int(delta[m,w].X)}, "
                    f"d={d[m,w].X:.3f}, "
                    f"d_DA={d['DA', w].X:.3f}, "
                    f"d_CM_u={d['CM_up', w].X:.3f}, "
                    f"d_CM_d={d['CM_down', w].X:.3f}, "
                    f"Q={Q[w]:.3f}"
                )

    print("\n======================================")
    print("        END OF EVP DEBUG REPORT")
    print("======================================\n")



# ============================================================
# Hjelpefunksjoner for plotting
# ============================================================


def compute_expected_volumes(model, a, U, V, W, M_u, M_v, M_w, policy_label):
    """
    Beregner forventet aktivert volum E[a] for hver markedsprodukt,
    gitt en modell og tilhørende a-variabler.

    Returnerer en DataFrame med kolonner:
    - Policy
    - Market product
    - Stage
    - Expected activated volume [MW]
    """

    scenario_tree = model._scenario_tree
    nodes = scenario_tree["nodes"]

    # ---------- Hjelpefunksjoner for sannsynligheter ----------
    def pi_u(u):
        # betinget sannsynlighet for u (gitt roten)
        return nodes[u].cond_prob

    def pi_v_given_u(v):
        return nodes[v].cond_prob

    def pi_w_given_v(w):
        return nodes[w].cond_prob

    exp_vol = {}

    # Stage 2: CM
    for m in M_u:
        val = 0.0
        for u in U:
            key = (m, u)
            if key in a:
                val += pi_u(u) * a[key].X
        exp_vol[m] = val

    # Stage 3: DA
    for m in M_v:   # typisk bare "DA"
        val = 0.0
        for u in U:
            for v in V[u]:
                key = (m, v)
                if key in a:
                    val += pi_u(u) * pi_v_given_u(v) * a[key].X
        exp_vol[m] = val

    # Stage 4: EAM
    for m in M_w:
        val = 0.0
        for u in U:
            for v in V[u]:
                for w in W[v]:
                    key = (m, w)
                    if key in a:
                        val += pi_u(u) * pi_v_given_u(v) * pi_w_given_v(w) * a[key].X
        exp_vol[m] = val

    # ---------- Bygg DataFrame ----------
    rows = []
    for m, val in exp_vol.items():
        if m in M_u:
            stage = "Stage 2 – CM"
        elif m in M_v:
            stage = "Stage 3 – DA"
        else:
            stage = "Stage 4 – EAM"

        rows.append({
            "Policy": policy_label,
            "Market product": m,
            "Stage": stage,
            "Expected activated volume [MW]": val
        })

    return pd.DataFrame(rows)
