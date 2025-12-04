import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import tree
import utils
import json
import read



def run_deterministic_benchmark(time_str, n):

    CM_up, CM_down, DA, EAM_up, EAM_down, prod_cap, picked_scenario_indices = read.load_parameters_from_parquet(time_str, n)

    model, x, r = solve_EV(time_str, n, CM_up, CM_down, DA, EAM_up, EAM_down, prod_cap)

    x_CMup = x["CM_up"].X
    x_CMdown = x["CM_down"].X
    r_CMup = r["CM_up"].X
    r_CMdown = r["CM_down"].X

    objective_value = 0

    for P_CMup in CM_up:
        for P_CMdown in CM_down:
            model, x, r = solve_EV(time_str=time_str, 
                            n=n,
                            CM_up=[P_CMup],
                            CM_down=[P_CMdown],
                            DA=DA,
                            EAM_up=EAM_up,
                            EAM_down=EAM_down,
                            prod_cap=prod_cap,
                            x_CMup=x_CMup, 
                            x_CMdown=x_CMdown, 
                            r_CMup=r_CMup, 
                            r_CMdown=r_CMdown)
            
            x_DA = x["DA"].X
            r_DA = r["DA"].X

            for P_DA in DA:
                model, x, r = solve_EV(time_str=time_str, 
                                n=n,
                                CM_up=[P_CMup],
                                CM_down=[P_CMdown],
                                DA=[P_DA],
                                EAM_up=EAM_up,
                                EAM_down=EAM_down,
                                prod_cap=prod_cap,
                                x_CMup=x_CMup, 
                                x_CMdown=x_CMdown, 
                                r_CMup=r_CMup, 
                                r_CMdown=r_CMdown,
                                x_DA=x_DA,
                                r_DA=r_DA)
                
                x_EAMup = x["EAM_up"].X
                x_EAMdown = x["EAM_down"].X
                r_EAMup = r["EAM_up"].X
                r_EAMdown = r["EAM_down"].X

                obj = model.objVal

                weight = 1 / ( len(CM_up) * len(CM_down) * len(DA) )

                objective_value += weight * obj

    return objective_value





def solve_EV(time_str=None, 
            n=None,
            CM_up=None,
            CM_down=None,
            DA=None,
            EAM_up=None,
            EAM_down=None,
            prod_cap=None,
            x_CMup=None, 
            r_CMup=None, 
            x_CMdown=None, 
            r_CMdown=None, 
            x_DA=None, 
            r_DA=None, 
            ):


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
    BIGM_2 = Q  # maksimal produksjonskapasitet
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


    # --- FIXED VARIABLES ---
    if x_CMup is not None:
        x["CM_up"] = x_CMup

    if x_CMdown is not None:
        x["CM_down"] = x_CMdown

    if r_CMup is not None:
        r["CM_up"] = r_CMup

    if r_CMdown is not None:
        r["CM_down"] = r_CMdown


    if x_DA is not None:
        x["DA"] = x_DA
    
    if r_DA is not None:
        r["DA"] = r_DA


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

    # Deviation constraints for CM_up

    diff = a["CM_up"] - x["EAM_up"]

    model.addConstr(
        diff <= BIGM_2 * mu["CM_up"]
    )

    model.addConstr(
        diff >= -BIGM_2 * (1 - mu["CM_up"])
    )

    model.addConstr(
        d["CM_up"] >= diff
    )

    model.addConstr(
        d["CM_up"] <= diff + BIGM_2 * (1 - mu["CM_up"])
    )

    model.addConstr(
        d["CM_up"] <= BIGM_2 * mu["CM_up"]
    )


    # Deviation constraints for CM_down

    diff = a["CM_down"] - x["EAM_down"]

    model.addConstr(
        diff <= BIGM_2 * mu["CM_down"]
    )

    model.addConstr(
        diff >= -BIGM_2 * (1 - mu["CM_down"])
    )

    model.addConstr(
        d["CM_down"] >= diff
    )

    model.addConstr(
        d["CM_down"] <= diff + BIGM_2 * (1 - mu["CM_down"])
    )

    model.addConstr(
        d["CM_down"] <= BIGM_2 * mu["CM_down"]
    )



    # Deviation constraints for EAM down market


    # EAM_down can exceed DA, but it will lead to a deviation. It must be constrained from above and below since
    # the deviation cost can be negative
    
    diff = a["EAM_down"] - a["DA"]

    model.addConstr(
        diff <= BIGM_2 * mu["EAM_down"]
    )

    model.addConstr(
        diff >= -BIGM_2 * (1 - mu["EAM_down"])
    )

    model.addConstr(
        d["EAM_down"] >= diff
    )

    model.addConstr(
        d["EAM_down"] <= diff + BIGM_2 * (1 - mu["EAM_down"])
    )

    model.addConstr(
        d["EAM_down"] <= BIGM_2 * mu["EAM_down"]
    )
            


    # Deviation constraints for DA market. 
    # The variable d_DA_w must be constrained both upwards and downwards to avoid allowing
    # d_DA_w to be set high to absorb deviation from the EAM up bids


    N = a["DA"] - a["EAM_down"]

    model.addConstr(
        N - Q <= BIGM_2 * mu["DA"]
    )

    model.addConstr(
        N - Q >= -BIGM_2 * (1 - mu["DA"])
    )

    model.addConstr(
        d["DA"] >= N - Q
    )

    model.addConstr(
        d["DA"] <= N - Q + BIGM_2 * (1 - mu["DA"])
    )

    model.addConstr(
        d["DA"] <= BIGM_2 * mu["DA"]
    )
            
            # --------------------------------------------------

    # Deviation constraints for EAM up market

    # EAM up deviation constraints
    # Dette representerer hvor mye ekstra kraft du mangler i scenario w for å levere DA-leveranse + EAM_down-reduksjon + EAM_up-økning 
    # etter at DA-shortfall (d_DA) og EAM_down-shortfall (d_EAM_down) er tatt hensyn til.
    # Z_w = a_DA[v] - a_EAM_down[w] + a_EAM_up[w] - Q[w] - d_DA[w]
    Z = (
    a["DA"]
    - a["EAM_down"]
    + a["EAM_up"]
    - d["DA"]
    + d["EAM_down"]
    - Q
    )


    model.addConstr(
        Z <= BIGM_3 * mu["EAM_up"]
    )

    model.addConstr(
        Z >= -BIGM_3 * (1 - mu["EAM_up"])
    )

    model.addConstr(
        d["EAM_up"] >= Z
    )

    model.addConstr(
        d["EAM_up"] <= Z + BIGM_3 * (1 - mu["EAM_up"])
    )

    model.addConstr(
        d["EAM_up"] <= BIGM_3 * mu["EAM_up"]
    )


            

    # Minimum bid quantity constraints for mFRR markets (CM and EAM)
    # Hvis b[m,s] = 1  ->  x[m,s] ≥ MIN_Q
    model.addConstr(
        x[m] >= x_mFRR_min * b[m]
    )

    # Hvis y_bid[m,s] = 0  ->  x[m,s] ≤ 0
    # (og generelt x[m,s] ≤ BIGM hvis y_bid = 1)
    model.addConstr(
        x[m] <= BIGM_2 * b[m]
    )



    # Constraining bid price in the EAM markets
    model.addConstr(
        r["EAM_up"] <= r_MAX_EAM_up
    )
    model.addConstr(
        r["EAM_down"] <= r_MAX_EAM_down
    )


    print("Added all constraints, starting to optimize model...")
    # --- OPTIMIZE MODEL ---
    
    model.optimize()

    runtime = model.Runtime
    print(f"Model optimized in {runtime:.2f} seconds.")

    
    # -------- PRINT RESULTS --------

    print("\n======================")
    print("   OPTIMAL SOLUTION")
    print("======================\n")

    if model.status == GRB.OPTIMAL:
        print(f"Objective value: {model.objVal:,.4f}\n")
    else:
        print("Model did not solve to optimality.\n")

    def print_vars(title, var_dict):
        print(f"--- {title} ---")
        # Var_dict kan være Gurobi dict (tupledict) eller liste
        try:
            keys = sorted(var_dict.keys(), key=lambda k: str(k))
        except:
            keys = var_dict.keys()
        for k in keys:
            v = var_dict[k]
            try:
                val = v.X
                print(f"{k}: {val:.4f}")
            except:
                pass
        print()

    # Print each variable group
    print_vars("Bid quantities x[m]", x)
    print_vars("Bid prices r[m]", r)
    print_vars("Activated quantities a[m]", a)
    print_vars("Activation δ[m]", delta)
    print_vars("Deviation d[m]", d)
    print_vars("Bid indicator b[m]", b)
    print_vars("Deviation indicator mu[m]", mu)

    print("\n====================================")
    print("         END OF OPTIMAL SOLUTION    ")
    print("====================================\n")

    return model, x, r

















    


