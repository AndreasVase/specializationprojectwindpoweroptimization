import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scenario_tree import build_scenario_tree, build_sets_from_tree

model = gp.Model()

# --- SETS ---
I = [1, 2, 3, 4]   # stages

M_u = ["CM_up", "CM_down"]
M_v = ["DA"]
M_w = ["EAM_up", "EAM_down"]
M  = M_u + M_v + M_w


# --- Input data ---
CM_up      = [4, 7]
CM_down    = [6, 8]
DA         = [3, 5]
EAM_up     = [4.5, 6.5]
EAM_down   = [3.5, 5.0]
wind_speed = [8, 10, 12]

# kost per MW avvik for hvert produkt m
C_data = {
    "CM_up":   10.0,   # eksempelverdi [€/MW]
    "CM_down": 10.0,
    "DA":      10.0,
    "EAM_up":  30.0,
    "EAM_down": 30.0,
}


# --- Bygg treet ---
scenario_tree = build_scenario_tree(
    CM_up=CM_up,
    CM_down=CM_down,
    DA=DA,
    EAM_up=EAM_up,
    EAM_down=EAM_down,
    wind_speed=wind_speed
)


# --- Bygg sett fra treet ---
U, V, W, S = build_sets_from_tree(scenario_tree)


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

P = build_price_parameter(scenario_tree)
Q = build_production_capacity(scenario_tree)
C = {m: C_data[m] for m in M}

BIGM = 1000

epsilon = 1e-3

P_MAX_EAM_up   = 0  # maks pris for EAM up
P_MAX_EAM_down = 0    # maks pris for EAM down

# flate mengder for v- og w-noder:
V_all = set().union(*V.values())
W_all = set().union(*W.values())


# ------- bygg indeksmengder (m,s) -------

idx_ms = []

# stage 2: (CM_up/CM_down, u)
for u in U:
    for m in M_u:
        idx_ms.append((m, u))

# stage 3: (DA, v)
for v in V_all:
    for m in M_v:
        idx_ms.append((m, v))

# stage 4: (EAM_up/EAM_down, w)
for w in W_all:
    for m in M_w:
        idx_ms.append((m, w))

# til d_{mw} skal vi bare ha (m,w) med m i M_w
idx_mw = []
for w in W_all:
    for m in M:
        idx_mw.append((m, w))



# --- VARIABLES ---

# x_ms: bid quantity
x = model.addVars(idx_ms, lb=0.0, name="x")

# r_ms: bid price
r = model.addVars(idx_ms, lb=0, name="r")

# δ_ms: 1 hvis budet aktiveres
delta = model.addVars(idx_ms, vtype=GRB.BINARY, name="delta")

# a_ms: aktivert kvantum
a = model.addVars(idx_ms, lb=0.0, name="a")

# d_mw: avvik fra aktivert kvantum i terminale scenarier
d = model.addVars(idx_mw, lb=0, name="d") 



# --- OBJECTIVE FUNCTION ---

nodes = scenario_tree["nodes"]  # fra build_scenario_tree
# U, V, W: fra build_sets_from_tree(tree)
#   U: set of u-noder
#   V: dict u -> set of v-noder
#   W: dict v -> set of w-noder

obj = gp.LinExpr()

for u in U:
    pi_u = nodes[u].cond_prob   # π_u

    # Inneste ledd for gitt u
    term_u = gp.quicksum(
        P[ (m, u) ] * a[m, u] for m in M_u
    )

    # Stage 3
    for v in V[u]:
        pi_v_u = nodes[v].cond_prob   # π_{v|u}

        term_v = gp.quicksum(
            P[ (m, v) ] * a[m, v] for m in M_v
        )

        # Stage 4
        for w in W[v]:
            pi_w_v = nodes[w].cond_prob   # π_{w|v}

            revenue_w = gp.quicksum(
                P[ (m, w) ] * a[m, w] for m in M_w
            )

            penalty_w = gp.quicksum(
                C[m] * d[m, w] for m in M
            )

            term_v += pi_w_v * (revenue_w - penalty_w)

        term_u += pi_v_u * term_v

    obj += pi_u * term_u

model.setObjective(obj, GRB.MAXIMIZE)
                   



# --- CONSTRAINTS ---


# Aktiveringsgrenser (for alle gyldige (m,s))
for (m, s) in idx_ms:
    # 1) a_ms <= x_ms
    model.addConstr(
        a[m, s] <= x[m, s],
        name=f"act_le_bid[{m},{s}]"
    )

    # 2) a_ms <= M * delta_ms
    model.addConstr(
        a[m, s] <= BIGM * delta[m, s],
        name=f"act_le_Mdelta[{m},{s}]"
    )

    # 3) a_ms >= x_ms - (1 - M * delta_ms)
    model.addConstr(
        a[m, s] >= x[m, s] - BIGM * (1 - delta[m, s]),
        name=f"act_ge_bid_bigM[{m},{s}]"
    )


# Set aktiveringsvariabel delta
for (m, s) in idx_ms:   

    # r_ms - P_ms <= M (1 - delta_ms)
    model.addConstr(
        r[m, s] - P[(m, s)] <= BIGM * (1 - delta[m, s]),
        name=f"act_upper[{m},{s}]"
    )

    # P_ms - r_ms <= M delta_ms - eps
    model.addConstr(
        P[(m, s)] - r[m, s] <= BIGM * delta[m, s] - epsilon,
        name=f"act_lower[{m},{s}]"
    )



# --- NON-ANTICIPATIVITY CONSTRAINTS ---

# Stage 2 non-anticipativity
u0 = next(iter(U))             # referansenode
for m in M_u:
    for u in U:
        if u != u0:
            model.addConstr(x[m, u] == x[m, u0], name=f"NA_x_stage2[{m},{u}]")
            model.addConstr(r[m, u] == r[m, u0], name=f"NA_r_stage2[{m},{u}]")


# Stage 3 non-anticipativity
for u in U:
    V_u = V[u]                # alle v-noder som følger u
    v0 = next(iter(V_u))      # referanse-node for v-noder med denne historien

    for m in M_v:
        for v in V_u:
            if v == v0:
                continue

            # x_{m,v} = x_{m,v0}
            model.addConstr(
                x[m, v] == x[m, v0],
                name=f"NA_x_stage3[{m},{u},{v}]"
            )

            # r_{m,v} = r_{m,v0}
            model.addConstr(
                r[m, v] == r[m, v0],
                name=f"NA_r_stage3[{m},{u},{v}]"
            )

# Stage 4 non-anticipativity
for v in V_all:
    W_v = W[v]                 # alle w-noder som følger v
    w0 = next(iter(W_v))       # referanse-node for denne historien

    for m in M_w:
        for w in W_v:
            if w == w0:
                continue

            # x_{m,w} = x_{m,w0}
            model.addConstr(
                x[m, w] == x[m, w0],
                name=f"NA_x_stage4[{m},{v},{w}]"
            )

            # r_{m,w} = r_{m,w0}
            model.addConstr(
                r[m, w] == r[m, w0],
                name=f"NA_r_stage4[{m},{v},{w}]"
            )


# EAM down-regulation cannot exceed the activated day-ahead quantity without incurring a deviation
for v in V_all:
    for w in W[v]:
        model.addConstr(
            a["EAM_down", w] - d["EAM_down", w] <= a["DA", v],
            name=f"link_EAM_DA[{v},{w}]"
        )


# any up or down regulation committed in market 1 must be followed by at least the same amount of up or down bidding in stage 3
# Bygg W(u) fra V(u) og W(v). W(u) er mengden av alle w-noder som følger u
W_u = {u: set().union(*(W[v] for v in V[u])) for u in U}

for u in U:
    for w in W_u[u]:
        # x_{3↑,w} + d_{1↑,w} >= a_{1↑,u}
        model.addConstr(
            x["EAM_up",  w] + d["CM_up",  w] >= a["CM_up",  u],
            name=f"cov_CMup[{u},{w}]"
        )

        # x_{3↓,w} + d_{1↓,w} >= a_{1↓,u}
        model.addConstr(
            x["EAM_down", w] + d["CM_down", w] >= a["CM_down", u],
            name=f"cov_CMdown[{u},{w}]"
        )


# The total activated quantity in the market products 2 and 3↑ does not exceed the available production capacity in each scenario.
for v in V_all:
    for w in W[v]:
        # a_2v <= Q_w + d_2w
        model.addConstr(
            a["DA", v] <= Q[w] + d["DA", w],
            name=f"cap_DA[{v},{w}]"
        )

        # a_3↑w <= Q_w - a_2v + d_3↑w
        model.addConstr(
            a["EAM_up", w] <= Q[w] - a["DA", v] + d["EAM_up", w],
            name=f"cap_EAMup[{v},{w}]"
        )


# Ensuring that a realistic price is bid in the EAM markets
for w in W_all:
    model.addConstr(
        r["EAM_up", w] <= P_MAX_EAM_up,
        name=f"max_price_EAMup_{w}"
    )
    model.addConstr(
        r["EAM_down", w] <= P_MAX_EAM_down,
        name=f"max_price_EAMdown_{w}"
    )



# --- OPTIMIZE MODEL ---
model.optimize()


def print_results(model, x, r, a, delta, d, U, V_all, W_all, M1, M2, M3):

    if model.Status != GRB.OPTIMAL:
        print("Model not solved to optimality. Status:", model.Status)
        return


    print("\n======================")
    print("   OPTIMAL SOLUTION")
    print("======================\n")

    # Objective value
    print(f"Objective value: {model.ObjVal:,.4f}\n")

    # ------------------------
    # Helper for clean output
    # ------------------------
    def print_nonzero(title, var_dict):
        print(f"--- {title} ---")
        found = False
        for key, var in var_dict.items():
            if abs(var.X) > 1e-8:
                print(f"{key}: {var.X:,.4f}")
                found = True
        if not found:
            print("(all zero)")
        print()

    # ------------------------
    # Print by variable group
    # ------------------------

    # X (quantities)
    print_nonzero("Bid quantities x[m,s]", x)

    # R (prices)
    print_nonzero("Bid prices r[m,s]", r)

    # A (activated quantities)
    print_nonzero("Activated a[m,s]", a)

    # Delta (binary acceptance indicators)
    print("--- Binary Acceptances δ[m,s] ---")
    for key, var in delta.items():
        print(f"{key}: {int(round(var.X))}")
    print()

    # D (deviations only in terminal nodes)
    print_nonzero("Deviation d[m,w]", d)

    # ------------------------
    # Scenario-wise aggregated output
    # ------------------------
    print("============== SCENARIO OUTPUT ==============\n")

    print("--- Stage 2 (CM): u ∈ U ---")
    for u in U:
        for m in M1:
            print(f"{m} in {u}: x={x[m,u].X:.3f}, a={a[m,u].X:.3f}, r={r[m,u].X:.3f}, δ={int(delta[m,u].X)}")
        print()

    print("--- Stage 3 (DA): v ∈ V ---")
    for v in V_all:
        for m in M2:
            print(f"{m} in {v}: x={x[m,v].X:.3f}, a={a[m,v].X:.3f}, r={r[m,v].X:.3f}, δ={int(delta[m,v].X)}")
        print()

    print("--- Stage 4 (EAM): w ∈ W ---")
    for w in W_all:
        for m in M3:
            print(f"{m} in {w}: x={x[m,w].X:.3f}, a={a[m,w].X:.3f}, r={r[m,w].X:.3f}, δ={int(delta[m,w].X)}, d={d[m,w].X:.3f}")
        print()

    print("=============================================")
    print("            END OF RESULTS")
    print("=============================================\n")


print_results(model, x, r, a, delta, d,
              U, V_all, W_all,
              M_u, M_v, M_w)