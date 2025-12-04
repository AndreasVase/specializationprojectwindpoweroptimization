import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import tree
import utils
import json
import read



def solve_EV(time_str=None, 
            n=None,
            x_CMup=None, 
            r_CMup=None, 
            x_CMdown=None, 
            r_CMdown=None, 
            x_DA=None, 
            r_DA=None, 
            x_EAMup=None, 
            r_EAMup=None, 
            x_EAMdown=None, 
            r_EAMdown=None
            ):
    
    CM_up, CM_down, DA, EAM_up, EAM_down, prod_cap, picked_scenario_indices = read.load_parameters_from_parquet(time_str, n)


    P = {}
    P["CM_up"] = np.mean(CM_up)
    P["CM_down"] = np.mean(CM_down)
    P["DA"] = np.mean(DA)
    P["EAM_up"] = np.mean(EAM_up)
    P["EAM_down"] = np.mean(EAM_down)
    
    Q = np.mean(prod_cap)

    C = {}
    C["CM_up"] = 2.0 * P["CM_up"]
    C["CM_down"] = 2.0 * P["CM_down"]
    C["DA"] = P["EAM_up"]  # bruker EAM up price som kost for DA-avvik. SIMPLIFISERING AV VIRKELIGHETEN
    C["EAM_up"] = 2.0 * P["EAM_up"]
    C["EAM_down"] = 2.0 * P["EAM_down"]


    model = gp.Model()

    # --- SETS ---

    M_u = ["CM_up", "CM_down"]
    M_v = ["DA"]
    M_w = ["EAM_up", "EAM_down"]
    M  = M_u + M_v + M_w

    R_max = 1000  # stor nok verdi for big-M
    BIGM_1 = R_max
    BIGM_2 = max(Q.values())  # maksimal produksjonskapasitet
    BIGM_3 = 2*BIGM_2

    epsilon = 1e-3

    x_mFRR_min = 10  # minimum budstørrelse i mFRR-markedet

    r_MAX_EAM_up   = 0  # maks pris for EAM up
    r_MAX_EAM_down = 0    # maks pris for EAM down


    # --- VARIABLES ---

    # x_ms: bid quantity
    x = model.addVars([m for m in M], lb=0, vtype=GRB.INTEGER, name="x")
    # r_ms: bid price
    r = model.addVars([m for m in M], name="r")
    # δ_ms: 1 hvis budet aktiveres
    delta = model.addVars([m for m in M], vtype=GRB.BINARY, name="delta")
    # a_ms: aktivert kvantum
    a = model.addVars([m for m in M], lb=0, vtype=GRB.INTEGER, name="a")
    # d_mw: avvik fra aktivert kvantum i terminale scenarier
    d = model.addVars([m for m in M], lb=0, ub=BIGM_2, name="d")
    # Binær variabel som indikerer om vi faktisk legger inn et bud (≠ 0)
    b = model.addVars([m for m in (M_u + M_w)], vtype=GRB.BINARY, name="b")
 
    # Binærvariabel som indikerer om det er avvik i marked m i scenario w
    mu = model.addVars([m for m in M], vtype=GRB.BINARY, name="mu")


    # --- OBJECTIVE FUNCTION ---


    obj = gp.quicksum(P[m] * a[m] - C[m] * d[m] for m in M)

    model.setObjective(obj, GRB.MAXIMIZE)


    # --- ACTIVATION CONSTRAINTS ---


    # Aktiveringsgrenser (for alle gyldige (m,s))
    for m in M:
        model.addConstr(
            a[m] <= x[m],
        )

        model.addConstr(
            a[m] <= BIGM_2 * delta[m],
        )

        model.addConstr(
            a[m] >= x[m] - BIGM_2 * (1 - delta[m]),
        )


    # Set aktiveringsvariabel delta
    for m in M:   

        # r_ms - P_ms <= M (1 - delta_ms)
        model.addConstr(
            r[m] - P[m] <= BIGM_1 * (1 - delta[m]),
        )

        # P_ms - r_ms <= M delta_ms - eps
        model.addConstr(
            P[m] - r[m] <= BIGM_1 * delta[m] - epsilon,
        )



    # --- MARKET CONSTRAINTS ---








    


