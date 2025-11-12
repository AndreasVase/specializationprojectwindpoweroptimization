import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scenario_tree import build_scenario_tree

model = gp.Model()

# --- SETS ---
I = [1, 2, 3]   # stages

M1 = ["CM_up", "CM_down"]
M2 = ["DA"]
M3 = ["EAM_up", "EAM_down"]
M  = M1 + M2 + M3


# --- Input data ---
CM_up      = [4, 7]
CM_down    = [6, 8]
DA         = [3, 5]
EAM_up     = [4.5, 6.5]
EAM_down   = [3.5, 5.0]
wind_speed = [8, 10, 12]

# --- Bygg treet ---
scenario_tree = build_scenario_tree(
    CM_up=CM_up,
    CM_down=CM_down,
    DA=DA,
    EAM_up=EAM_up,
    EAM_down=EAM_down,
    wind_speed=wind_speed
)
# --- Hent ut noder per stadium ---
S1_nodes, S2_nodes, S3_nodes = scenario_tree["stage1"], scenario_tree["stage2"], scenario_tree["stage3"]


# --- PARAMETERS ---

# 2) Gi scenariene ID-er (u_i, v_j, w_k) og lag oppslag
U = {f"u{i}": n for i, n in enumerate(S1_nodes, 1)}
V = {f"v{j}": n for j, n in enumerate(S2_nodes, 1)}
W = {f"w{k}": n for k, n in enumerate(S3_nodes, 1)}

# Praktiske lister
U_ids, V_ids, W_ids = list(U.keys()), list(V.keys()), list(W.keys())


# --- V(u): grupper v etter u som parent ---
U_to_V = {uid: [] for uid in U_ids}
for uid in U_ids:
    u_node = U[uid]
    Vu = [vid for vid, v_node in V.items() if v_node.parent is u_node]
    U_to_V[uid] = Vu


# --- W(v): grupper w etter v som parent ---
V_to_W = {vid: [] for vid in V_ids}
for vid in V_ids:
    v_node = V[vid]
    Wv = [wid for wid, w_node in W.items() if w_node.parent is v_node]
    V_to_W[vid] = Wv

U_to_W = {uid: list({wid for vid in U_to_V[uid] for wid in V_to_W[vid]}) for uid in U_ids}


# 4) Parametre fra scenariotreet
# R_ms: pris bare der det finnes i nodens prices
R = {}
for uid, node in U.items():
    for m in M1:
        R[(m, uid)] = node.prices[m]
for vid, node in V.items():
    for m in M2:
        R[(m, vid)] = node.prices[m]
for wid, node in W.items():
    for m in M3:
        R[(m, wid)] = node.prices[m]

# Q_s: kapasitet per scenario (her: enkel mapping fra "wind" -> kapasitet)
def q_from_wind(w):  # tilpass etter behov
    return float(w)
Q = {sid: q_from_wind(node.wind) for sid, node in {**U, **V, **W}.items()}


# Scenario probabilities, all scenarios equally likely
pi_u = 1/len(U_ids) # 1/12
pi_v = 1/len(V_ids) # 1/72
pi_w = 1/len(W_ids) # 1/864

# Avvikskostnad per marked
C_dev = {
    "CM_up":  10.0, 
    "CM_down": 10.0,
    "DA":     10.0,
    "EAM_up": 30.0,
    "EAM_down": 30.0,
}

BIGM = 1000

epsilon = 1e-3

P_MAX_EAM_up   = 1  # maks pris for EAM up
P_MAX_EAM_down = 1    # maks pris for EAM down

# --- VARIABLES ---

# 5) Gyldige indekser (m, s) – kun der R_ms er definert
MS_pairs = list(R.keys())  # dette gir (m,s) for riktige stadier

DW_pairs = [(m, w) for w in W_ids for m in M]


# Variabler 

x = model.addVars(MS_pairs, name="x", lb=0.0)                     # bid qty
p = model.addVars(MS_pairs, name="p", lb=0.0)                     # bid price (lb ev. <0 i DA)
a = model.addVars(MS_pairs, name="a", lb=0.0)                     # activated qty
d = model.addVars(DW_pairs, name="d", lb=0.0)                     # deviation
delta = model.addVars(MS_pairs, vtype=GRB.BINARY, name="delta")   # activation flag


# --- OBJECTIVE FUNCTION ---

# Objektiv: forventet verdi (inntekt – avvikskost)
# Vi vekter hvert (m,s)-ledd med scenarioets joint-prob pi[s]

# Inntekter i hvert stadium
obj  = gp.quicksum(pi_u * R[(m, u)] * a[(m, u)]
                   for (m, u) in MS_pairs if m in M1)

obj += gp.quicksum(pi_v * R[(m, v)] * a[(m, v)]
                   for (m, v) in MS_pairs if m in M2)

# Stadium 3: inntekt minus avvikskost
obj += gp.quicksum(pi_w * R[(m, w)] * a[(m, w)] for (m, w) in MS_pairs if m in M3)

obj -= gp.quicksum(pi_w * C_dev[m] * d[(m, w)] for (m, w) in DW_pairs if m in M)
                   



# --- CONSTRAINTS ---


# Aktiveringsgrenser (for alle gyldige (m,s))
model.addConstrs((a[(m, s)] <= x[(m, s)]                    for (m, s) in MS_pairs), name="act_le_bid")
model.addConstrs((a[(m, s)] <= BIGM * delta[(m, s)]         for (m, s) in MS_pairs), name="act_if_delta")
model.addConstrs((a[(m, s)] >= x[(m, s)] - BIGM * (1 - delta[(m, s)])         for (m, s) in MS_pairs), name="act_ge_bid_if_delta")

# Set aktiveringsvariabel delta
model.addConstrs((p[(m, s)] - R[(m, s)] <= BIGM * (1 - delta[(m, s)]) for (m, s) in MS_pairs), name="set_activation_1")
model.addConstrs((R[(m, s)] - p[(m, s)] <= BIGM * delta[(m, s)] - epsilon for (m, s) in MS_pairs), name="set_activation_2")



# --- NON-ANTICIPATIVITY CONSTRAINTS ---
for mkt in M1:
    base = U_ids[0]
    for u in U_ids[1:]:
        model.addConstr(x[(mkt, u)] == x[(mkt, base)], name=f"NA_x_u_{mkt}_{u}")
        model.addConstr(p[(mkt, u)] == p[(mkt, base)], name=f"NA_p_u_{mkt}_{u}")

for mkt in M2:
    for uid in U_ids:
        Vu = U_to_V[uid]
        if len(Vu) > 1:
            base = Vu[0]
            for v in Vu[1:]:
                model.addConstr(x[(mkt, v)] == x[(mkt, base)], name=f"NA_x_v_{mkt}_{uid}_{v}")
                model.addConstr(p[(mkt, v)] == p[(mkt, base)], name=f"NA_p_v_{mkt}_{uid}_{v}")

#for mkt in M3:
#    for vid in V_ids:
#        Wv = V_to_W[vid]
#        if len(Wv) > 1:
#            base = Wv[0]
#            for w in Wv[1:]:
#                model.addConstr(x[(mkt, w)] == x[(mkt, base)], name=f"NA_x_v_{mkt}_{uid}_{w}")
#                model.addConstr(p[(mkt, w)] == p[(mkt, base)], name=f"NA_p_v_{mkt}_{uid}_{w}")


# EAM down-regulation cannot exceed the activated day-ahead quantity without incurring a deviation
for vid in V_ids:
    for wid in V_to_W[vid]:
        model.addConstr(
            a[("EAM_down", wid)] - d[("EAM_down", wid)] <= a[("DA", vid)],
            name=f"EAMdown_vs_DA_{vid}_{wid}"
        )


# any up or down regulation committed in market 1 must be followed by at least the same amount of up or down bidding in stage 3
for uid in U_ids:
    for wid in U_to_W[uid]:
        model.addConstr(
            x[("EAM_up",   wid)] + d[("CM_up",   wid)] >= a[("CM_up",   uid)],
            name=f"CMup_to_EAMup_{uid}_{wid}"
        )
        model.addConstr(
            x[("EAM_down", wid)] + d[("CM_down", wid)] >= a[("CM_down", uid)],
            name=f"CMdown_to_EAMdown_{uid}_{wid}"
        )


# The total activated quantity in the market products 2 and 3↑ does not exceed the available production capacity in each scenario.
for vid in V_ids:
    for wid in V_to_W[vid]:
        # market 2 (DA)
        model.addConstr(
            a[("DA", vid)] <= Q[wid] + d[("DA", wid)],
            name=f"cap_DA_{vid}_{wid}"
        )
        # market 3 up (EAM↑)
        model.addConstr(
            a[("EAM_up", wid)] <= Q[wid] - a[("DA", vid)] + d[("EAM_up", wid)],
            name=f"cap_EAMup_{vid}_{wid}"
        )


# Ensuring that a realistic price is bid in the EAM markets
for wid in W_ids:
    model.addConstr(
        p[("EAM_up", wid)] <= P_MAX_EAM_up,
        name=f"min_price_EAMup_{wid}"
    )
    model.addConstr(
        p[("EAM_down", wid)] <= P_MAX_EAM_down,
        name=f"max_price_EAMdown_{wid}"
    )




model.setObjective(obj, gp.GRB.MAXIMIZE)
model.optimize()

# Hent ut litt resultat
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective: {model.objVal:.4f}")
    # eksempel: skriv noen løsninger
    for (mkt, s) in MS_pairs[:1000]:
        if a[(mkt, s)].X > 1e-6:
            print(f"{mkt}@{s}: a={a[(mkt,s)].X:.3f}, x={x[(mkt,s)].X:.3f}, p={p[(mkt,s)].X:.3f}, δ={int(delta[(mkt,s)].X)}")