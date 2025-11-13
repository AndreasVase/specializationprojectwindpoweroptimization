import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import tree
import utils
import json


def run_model(data_path, det_policy_file, evaluate_deterministic_policy=False, verbose=True):

    model = gp.Model()

    # --- SETS ---
    I = [1, 2, 3, 4]   # stages

    M_u = ["CM_up", "CM_down"]
    M_v = ["DA"]
    M_w = ["EAM_up", "EAM_down"]
    M  = M_u + M_v + M_w


    # Bygg treet
    scenario_tree = tree.build_scenario_tree(data_path)
    # Lagre treet i modellen for tilgang
    model._scenario_tree = scenario_tree

    # Bygg sett fra treet
    U, V, W, S = tree.build_sets_from_tree(scenario_tree)
    
    # flate mengder for v- og w-noder:
    V_all = set().union(*V.values())
    W_all = set().union(*W.values())

    # bygg indeksmengder (m,s)
    idx_ms, idx_mw = tree.build_index_sets(U=U, V_all=V_all, W_all=W_all, M_u=M_u, M_v=M_v, M_w=M_w, M=M)


    # --- PARAMETERS ---

    P = utils.build_price_parameter(scenario_tree)
    Q = utils.build_production_capacity(scenario_tree)

    C = {}  # (m, w) -> cost coefficient
    for v in V_all:
        # DA-pris i dette v-scenariet
        da_price = P[("DA", v)]
        for w in W[v]:  # alle w som følger etter v
            eam_up_price = P[("EAM_up", w)]  # pris for EAM up i dette w-scenariet
            eam_down_price = P[("EAM_down", w)]  # pris for EAM down i dette w-scenariet
            # her definerer vi kost for ALLE markeder i dette terminalscenariet
            C[("CM_up",    w)] = 3.0 * da_price
            C[("CM_down",  w)] = 3.0 * da_price
            C[("DA",       w)] = 2.0 * da_price
            C[("EAM_up",   w)] = 2.0 * eam_up_price
            C[("EAM_down", w)] = 2.0 * eam_down_price

    BIGM = 1000

    epsilon = 1e-3

    P_MAX_EAM_up   = 0  # maks pris for EAM up
    P_MAX_EAM_down = 0    # maks pris for EAM down


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
                    C[(m, w)] * d[m, w] for m in M
                )

                term_v += pi_w_v * (revenue_w - penalty_w)

            term_u += pi_v_u * term_v

        obj += pi_u * term_u

    model.setObjective(obj, GRB.MAXIMIZE)


    # --- ACTIVATION CONSTRAINTS ---


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


    # --- MARKET CONSTRAINTS ---

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


    # --- EVALUATE DETERMINISTIC POLICY ---
    if evaluate_deterministic_policy:
        with open(det_policy_file, "r") as f:
            det_policy = json.load(f)

        # FIKS x,r (CM) TIL DEN DETERMINISTISKE POLICYEN
        for m in det_policy if m in M_u else []:
            for s in S:
                if (m, s) in x:  # sjekk at paret finnes
                    model.addConstr(x[m, s] == det_policy[m]["x"],
                                    name=f"fix_x_{m}_{s}")
                    model.addConstr(r[m, s] == det_policy[m]["r"],
                                    name=f"fix_r_{m}_{s}")


    # --- OPTIMIZE MODEL ---
    model.optimize()

    # --- PRINT RESULTS ---
    if verbose:
        if evaluate_deterministic_policy:
            utils.print_results_deterministic_policy(model, x, a, r, delta, d, U, V, W, M_u, M_v, M_w)
        else:
            utils.print_results(model, x, r, a, delta, d, U, V, W, M_u, M_v, M_w)


    return model, x, r, a, delta, d, U, V, W, M_u, M_v, M_w






