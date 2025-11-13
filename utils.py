from gurobipy import GRB

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

            # HER legges det inn en mer realistisk produksjonsfunksjon
            prod_cap = wind  # enkel mapping som eksempel
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



def print_results(model, x, r, a, delta, d,
                  U, V, W, V_all, W_all, M1, M2, M3):

    if model.Status != GRB.OPTIMAL:
        print("Model not solved to optimality. Status:", model.Status)
        return

    print("\n======================")
    print("   OPTIMAL SOLUTION")
    print("======================\n")

    # Objective value
    print(f"Objective value: {model.ObjVal:,.4f}\n")

    # ---------- Stage 2: CM (samme for alle u pga ikke-anticipativitet) ----------
    u0 = next(iter(U))
    print("--- Stage 2 (CM) – same for all u (by non-anticipativity) ---")
    for m in M1:
        print(
            f"{m}: x={x[m,u0].X:.3f}, "
            f"a={a[m,u0].X:.3f}, "
            f"r={r[m,u0].X:.3f}, "
            f"δ={int(round(delta[m,u0].X))}"
        )
    print()

    # ---------- Stage 3: DA per v-node (sortert v1, v2, ..., vN) ----------
    print("--- Stage 3 (DA) – per v ---")
    max_v_per_u = 20  # juster hvis du vil se flere/færre
    for u in U:
        V_u = V[u]
        V_sample = sort_nodes(V_u)[:max_v_per_u]
    for v in sort_nodes(V_sample):

        for m in M2:
            print(
                f"{m} in {v}: "
                f"x={x[m,v].X:.3f}, "
                f"a={a[m,v].X:.3f}, "
                f"r={r[m,v].X:.3f}, "
                f"δ={int(round(delta[m,v].X))}"
            )
    print()

    # ---------- Stage 4: EAM – representative w per v ----------
    print("--- Stage 4 (EAM) – representative child scenarios per v ---")
    max_w_per_v = 2  # juster hvis du vil se flere/færre

    for v in sort_nodes(V_sample):
        W_v = W[v]
        W_sample = sort_nodes(W_v)[:max_w_per_v]

        print(f"\nParent scenario {v}:")
        for w in W_sample:
            for m in M3:
                print(
                    f"  {m} in {w}: "
                    f"x={x[m,w].X:.3f}, "
                    f"a={a[m,w].X:.3f}, "
                    f"r={r[m,w].X:.3f}, "
                    f"δ={int(round(delta[m,w].X))}, "
                    f"d={d[m,w].X:.3f}"
                )
    print()

    print("=============================================")
    print("            END OF RESULTS")
    print("=============================================\n")


def print_deterministic_policy_results(model):
    print("\n===== EV OF DETERMINISTIC POLICY =====")
    if model.Status == GRB.OPTIMAL:
        print(f"Expected objective value: {model.ObjVal:.4f}")
    else:
        print("Model not optimal, status:", model.Status)
    print("======================================\n")