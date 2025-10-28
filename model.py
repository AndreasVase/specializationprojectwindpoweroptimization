import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np

model = gp.Model()

###############
# Dummy data
################

# --- SETS ---
I = [1, 2, 3]   # stages

# clearing price scenarios for different markets
mFRRCM_up = [40, 60]      # market 0
mFRRCM_down = [60, 80]    # market 1
DA = [30, 50]             # market 2
mFRREAM_up = [45, 65]     # market 3
mFRREAM_down = [45, 65]   # market 4

M = ["mFRRCM_up", "mFRRCM_down", "DA", "mFRREAM_up", "mFRREAM_down"]  # markets

clearing_prices = [[mFRRCM_up[0], mFRRCM_down[0], DA[0], mFRREAM_up[0], mFRREAM_down[0]],
                    [mFRRCM_up[1], mFRRCM_down[1], DA[1], mFRREAM_up[1], mFRREAM_down[1]]]

wind_speed = [8, 9, 10]

S = [(price_row, wind) for price_row, wind in itertools.product(clearing_prices, wind_speed)]


# --- PARAMETERS ---

R = [[S[s_idx][0][m] for s_idx in range(len(S))] for m in range(len(M))] # Clearing price in market m and scenario s

W = [S[s_idx][1] for s_idx in range(len(S))]  # Wind speed in scenario s
Q = [10 * w for w in W]    # production capacity in scenario s

# Scenario probabilities, all scenarios equally likely
pi = 1/len(S)


# Cost per MW deviation in market m (€/MW)
C = [25,20,30,18,15]  # markets mFRRCM_up, mFRRCM_down, DA, mFRREAM_up, mFRREAM_down


# --- VARIABLES ---

x = {(i, m, s_idx): model.addVar(vtype=gp.GRB.CONTINUOUS) for i in I for m in range(len(M)) for s_idx in range(len(S))}
p = {(i, m, s_idx): model.addVar(vtype=gp.GRB.CONTINUOUS) for i in I for m in range(len(M)) for s_idx in range(len(S))}
delta = {(i, m, s_idx): model.addVar(vtype=gp.GRB.BINARY) for i in I for m in range(len(M)) for s_idx in range(len(S))}
a = {(i, m, s_idx): model.addVar(vtype=gp.GRB.CONTINUOUS) for i in I for m in range(len(M)) for s_idx in range(len(S))}
d = {(m, s_idx): model.addVar(vtype=gp.GRB.CONTINUOUS) for m in range(len(M)) for s_idx in range(len(S))}
z = {(m, s_idx): model.addVar(vtype=gp.GRB.CONTINUOUS) for m in range(len(M)) for s_idx in range(len(S))}
z_netto_DA = {s_idx: model.addVar(vtype=gp.GRB.CONTINUOUS) for s_idx in range(len(S))}

# --- OBJECTIVE FUNCTION ---

model.setObjective(
    gp.quicksum(pi * R[m][s_idx] * a[(1, m, s_idx)] for m in range(len(M)) for s_idx in range(len(S))) +
    gp.quicksum(pi * R[m][s_idx] * a[(2, m, s_idx)] for m in range(len(M)) for s_idx in range(len(S))) +
    gp.quicksum(pi * (R[m][s_idx] * a[(3, m, s_idx)] - C[m] * d[(m, s_idx)])
                for m in range(len(M)) for s_idx in range(len(S))),
    gp.GRB.MAXIMIZE
)

# --- CONSTRAINTS ---




model.optimize()