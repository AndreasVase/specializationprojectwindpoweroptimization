import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import json
import read
import utils



def run_benchmark_model(path, policy_filename):
    """
    Kjører en deterministisk benchmark-modell basert på forventede verdier
    fra input datafilen.
    """

    model_det = gp.Model("deterministic_benchmark")

    P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, wind_speed_mean = read.load_expected_values_from_csv(path)

    # --------------------------
    # 2) SETT & PARAMETERE
    # --------------------------

    M_u = ["CM_up", "CM_down"]
    M_v = ["DA"]
    M_w = ["EAM_up", "EAM_down"]
    M   = M_u + M_v + M_w

    # Prisparameter P[m]
    P = {
        "CM_up":    P_CM_up,
        "CM_down":  P_CM_down,
        "DA":       P_DA,
        "EAM_up":   P_EAM_up,
        "EAM_down": P_EAM_down,
    }

    # Kostnad per MW avvik (bruker samme struktur som i den stokastiske modellen)
    C = {
        "CM_up":    3.0  * P_DA,
        "CM_down":  3.0  * P_DA,
        "DA":       2.0  * P_DA,
        "EAM_up":   2.0  * P_EAM_up,
        "EAM_down": 2.0  * P_EAM_down,
    }

    Q = utils.wind_speed_to_production_capacity(wind_speed_mean)

    BIGM    = 1000.0
    epsilon = 1e-3

    P_MAX_EAM_up   = 0.0
    P_MAX_EAM_down = 0.0

    # --------------------------
    #  VARIABLER
    # --------------------------


    # x_m: budmengde
    x = model_det.addVars(M, lb=0.0, name="x")
    # r_m: budpris
    r = model_det.addVars(M, lb=0.0, name="r")
    # delta_m: 1 hvis budet aktiveres
    delta = model_det.addVars(M, vtype=GRB.BINARY, name="delta")
    # a_m: aktivert mengde
    a = model_det.addVars(M, lb=0.0, name="a")
    # d_m: avvik
    d = model_det.addVars(M, lb=0.0, name="d")

    # --------------------------
    # OBJEKTIVFUNKSJON
    # --------------------------

    # Maksimer forventet inntekt - forventet straff for avvik
    obj = gp.quicksum(P[m] * a[m] for m in M) \
          - gp.quicksum(C[m] * d[m] for m in M)

    model_det.setObjective(obj, GRB.MAXIMIZE)

    # --------------------------
    # RESTRIKSJONER
    # --------------------------

    # Aktiveringsgrenser (a mot x og delta)
    for m in M:
        # a_m <= x_m
        model_det.addConstr(a[m] <= x[m], name=f"act_le_bid[{m}]")

        # a_m <= M * delta_m
        model_det.addConstr(a[m] <= BIGM * delta[m], name=f"act_le_Mdelta[{m}]")

        # a_m >= x_m - M * (1 - delta_m)
        model_det.addConstr(a[m] >= x[m] - BIGM * (1 - delta[m]),
                            name=f"act_ge_bid_bigM[{m}]")


    # Pris–aksept-logikk (delta_m)
    for m in M:
        # r_m - P_m <= M (1 - delta_m)
        model_det.addConstr(
            r[m] - P[m] <= BIGM * (1 - delta[m]),
            name=f"act_upper[{m}]"
        )

        # P_m - r_m <= M delta_m - eps
        model_det.addConstr(
            P[m] - r[m] <= BIGM * delta[m] - epsilon,
            name=f"act_lower[{m}]"
        )


    # EAM down-reg kan ikke overstige aktivert DA uten avvik
    model_det.addConstr(
        a["EAM_down"] - d["EAM_down"] <= a["DA"],
        name="link_EAM_DA"
    )

    # CM-aktivering må dekkes av (minst) EAM-bud + avvik
    model_det.addConstr(
        x["EAM_up"] + d["CM_up"] >= a["CM_up"],
        name="cov_CMup"
    )

    model_det.addConstr(
        x["EAM_down"] + d["CM_down"] >= a["CM_down"],
        name="cov_CMdown"
    )

    # Produksjonskapasitetsbegrensninger
    # a_DA <= Q + d_DA
    model_det.addConstr(
        a["DA"] <= Q + d["DA"],
        name="cap_DA"
    )

    # a_EAM_up <= Q - a_DA + d_EAM_up
    model_det.addConstr(
        a["EAM_up"] <= Q - a["DA"] + d["EAM_up"],
        name="cap_EAMup"
    )

    # Realistiske priser i EAM (samme som i stokastisk modell)
    model_det.addConstr(
        r["EAM_up"]   <= P_MAX_EAM_up,
        name="max_price_EAMup"
    )
    model_det.addConstr(
        r["EAM_down"] <= P_MAX_EAM_down,
        name="max_price_EAMdown"
    )

    # --------------------------
    # LØS MODELLEN
    # --------------------------

    model_det.optimize()

    # --------------------------
    # LAGRE POLICY TIL FIL
    # --------------------------

    det_policy = {
        "CM_up": {
            "x": float(x["CM_up"].X),
            "r": float(r["CM_up"].X),
        },
        "CM_down": {
            "x": float(x["CM_down"].X),
            "r": float(r["CM_down"].X),
        },
        "DA": {
            "x": float(x["DA"].X),
            "r": float(r["DA"].X),
        },
        "EAM_up": {
            "x": float(x["EAM_up"].X),
            "r": float(r["EAM_up"].X),
        },
        "EAM_down": {
            "x": float(x["EAM_down"].X),
            "r": float(r["EAM_down"].X),
        },
    }

    with open(policy_filename, "w") as f:
        json.dump(det_policy, f, indent=2)

    print(f"Saved deterministic policy to {policy_filename}")
