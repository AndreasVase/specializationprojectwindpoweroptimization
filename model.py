import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import tree
import utils
import json
import statistics


def run_model(
    time_str: str, 
    n:int, seed=None, 
    det_policy_file=None, 
    evaluate_deterministic_policy=False, 
    only_da_and_eam=False, 
    verbose=True,
    cm_penalty_multiplier: float = 2.0,):

    model = gp.Model()

    # --- SETS ---
    I = [1, 2, 3, 4]   # stages

    M_u = ["CM_up", "CM_down"]
    M_v = ["DA"]
    M_w = ["EAM_up", "EAM_down"]
    M  = M_u + M_v + M_w


    # Bygg treet
    scenario_tree = tree.build_scenario_tree(time_str, n, seed)
    print("[INFO] Built scenario tree.")
    # Lagre treet i modellen for tilgang
    model._scenario_tree = scenario_tree
    print("[INFO] Stored scenario tree in model.")


    # Bygg sett fra treet
    U, V, W, S = tree.build_sets_from_tree(scenario_tree)
    print("[INFO] Built sets from scenario tree.")

    # flate mengder for v- og w-noder:
    V_all = set().union(*V.values())
    W_all = set().union(*W.values())

    # bygg indeksmengder (m,s)
    idx_ms, idx_mw = tree.build_index_sets(U=U, V_all=V_all, W_all=W_all, M_u=M_u, M_v=M_v, M_w=M_w, M=M)
    print("[INFO] Built index sets.")

    # --- PARAMETERS ---
    print ("[INFO] Building scenario tree")
    

    P = utils.build_price_parameter(scenario_tree)
    Q = utils.build_production_capacity(scenario_tree)
    C = utils.build_cost_parameters(U, V, W, P, cm_penalty_multiplier=cm_penalty_multiplier)

    Pmax = {m: max(P[m, s] for s in S if (m, s) in idx_ms) for m in M}


    BIGM_1 = max(Pmax.values())
    BIGM_2 = max(Q.values())  # maksimal produksjonskapasitet
    BIGM_3 = 2*BIGM_2

    epsilon = 1e-3

    x_mFRR_min = 10  # minimum budstørrelse i mFRR-markedet

    r_MAX_EAM_up   = 0  # maks pris for EAM up
    r_MAX_EAM_down = 0    # maks pris for EAM down


    # --- VARIABLES ---

    # x_ms: bid quantity
    x = model.addVars(idx_ms, lb=0, vtype=GRB.INTEGER, name="x")
    # r_ms: bid price
    r = model.addVars(idx_ms, lb=0, name="r")
    # δ_ms: 1 hvis budet aktiveres
    delta = model.addVars(idx_ms, vtype=GRB.BINARY, name="delta")
    # a_ms: aktivert kvantum
    a = model.addVars(idx_ms, lb=0, vtype=GRB.INTEGER, name="a")
    # d_mw: avvik fra aktivert kvantum i terminale scenarier
    d = model.addVars(idx_mw, lb=0, ub=BIGM_2, name="d")
    # Binær variabel som indikerer om vi faktisk legger inn et bud (≠ 0)
    b = model.addVars([(m, s) for (m, s) in idx_ms if m in (M_u + M_w)], vtype=GRB.BINARY, name="b")
 
    # Binærvariabel som indikerer om det er avvik i marked m i scenario w
    mu = model.addVars([(m, w) for (m, w) in idx_mw], vtype=GRB.BINARY, name="mu")


    # --- OBJECTIVE FUNCTION ---s


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
                    C[ (m, w) ] * d[m, w] for m in M
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
            a[m, s] <= BIGM_2 * delta[m, s],
            name=f"act_le_Mdelta[{m},{s}]"
        )

        # 3) a_ms >= x_ms - (1 - M * delta_ms)
        model.addConstr(
            a[m, s] >= x[m, s] - BIGM_2 * (1 - delta[m, s]),
            name=f"act_ge_bid_bigM[{m},{s}]"
        )


    # Set aktiveringsvariabel delta
    for (m, s) in idx_ms:   

        # r_ms - P_ms <= M (1 - delta_ms)
        model.addConstr(
            r[m, s] - P[(m, s)] <= BIGM_1 * (1 - delta[m, s]),
            name=f"act_upper[{m},{s}]"
        )

        # P_ms - r_ms <= M delta_ms - eps
        model.addConstr(
            P[(m, s)] - r[m, s] <= BIGM_1 * delta[m, s] - epsilon,
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


    # any up or down regulation committed in market 1 must be followed by at least the same amount of up or down bidding in stage 3
    # Bygg W(u) fra V(u) og W(v). W(u) er mengden av alle w-noder som følger u
    W_u = {u: set().union(*(W[v] for v in V[u])) for u in U}

    
    for u in U:
        for w in W_u[u]:

            # Deviation constraints for CM_up

            """
            # hjelpevariabel for diff
            z1 = model.addVar(lb=-GRB.INFINITY, name=f"diff_CM_up_{u}_{w}")
            model.addConstr(
                z1 == a["CM_up", u] - x["EAM_up", w],
                name=f"def_diff_CM_up_{u}_{w}"
            )

            # d["CM_up", w] = max(z, 0)
            model.addGenConstrMax(
                d["CM_up", w],          # resultatvariabel
                [z1],                    # KUN variabler i lista
                constant=0.0,           # 0 som konstant bidrag
                name=f"max_dev_CM_up_{u}_{w}",
            )
            """

            # Deviation constraints for CM_up

            # diff = a_CM_up[u] - x_EAM_up[w]
            diff = a["CM_up", u] - x["EAM_up", w]

            # diff <= M * mu
            model.addConstr(
                diff <= BIGM_2 * mu["CM_up", w]
            )

            # diff >= -M * (1 - mu)
            model.addConstr(
                diff >= -BIGM_2 * (1 - mu["CM_up", w]),
            )

            # d_CM_up[w] >= diff
            model.addConstr(
                d["CM_up", w] >= diff,
            )

            # d_CM_up[w] <= diff + M * (1 - mu)
            model.addConstr(
                d["CM_up", w] <= diff + BIGM_2 * (1 - mu["CM_up", w])
            )

            # d_CM_up[w] <= M * mu
            model.addConstr(
                d["CM_up", w] <= BIGM_2 * mu["CM_up", w]
            )




            # Deviation constraints for CM_down

            # diff = a_CM_up[u] - x_EAM_up[w]
            diff = a["CM_down", u] - x["EAM_down", w]

            # diff <= M * mu
            model.addConstr(
                diff <= BIGM_2 * mu["CM_down", w]
            )

            # diff >= -M * (1 - mu)
            model.addConstr(
                diff >= -BIGM_2 * (1 - mu["CM_down", w]),
            )

            # d_CM_up[w] >= diff
            model.addConstr(
                d["CM_down", w] >= diff,
            )

            # d_CM_up[w] <= diff + M * (1 - mu)
            model.addConstr(
                d["CM_down", w] <= diff + BIGM_2 * (1 - mu["CM_down", w])
            )

            # d_CM_up[w] <= M * mu
            model.addConstr(
                d["CM_down", w] <= BIGM_2 * mu["CM_down", w]
            )


            ## x_{3↓,w} + d_{1↓,w} >= a_{1↓,u}
            #model.addConstr(
            #    x["EAM_down", w] + d["CM_down", w] >= a["CM_down", u],
            #    name=f"cov_CMdown[{u},{w}]"
            #)


    # Deviation constraints for EAM down market

    for v in V_all:
        for w in W[v]:

            # EAM_down can exceed DA, but it will lead to a deviation. It must be constrained from above and below since
            # the deviation cost can be negative
            
            diff = a["EAM_down", w] - a["DA", v]

            model.addConstr(
                diff <= BIGM_2 * mu["EAM_down", w]
            )

            model.addConstr(
                diff >= -BIGM_2 * (1 - mu["EAM_down", w])
            )

            # 3) d >= Delta
            model.addConstr(
                d["EAM_down", w] >= diff
            )

            # 4) d <= Delta + M * (1 - eta)
            model.addConstr(
                d["EAM_down", w] <= diff + BIGM_2 * (1 - mu["EAM_down", w])
            )

            # 5) d <= M * eta
            model.addConstr(
                d["EAM_down", w] <= BIGM_2 * mu["EAM_down", w]
            )
            


    # Deviation constraints for DA market. 
    # The variable d_DA_w must be constrained both upwards and downwards to avoid allowing
    # d_DA_w to be set high to absorb deviation from the EAM up bids

    for v in V_all:
        for w in W[v]:

            N = a["DA", v] - a["EAM_down", w]

            model.addConstr(
                N - Q[w] <= BIGM_2 * mu["DA", w]
            )

            model.addConstr(
                N - Q[w] >= -BIGM_2 * (1 - mu["DA", w])
            )

            model.addConstr(
                d["DA", w] >= N - Q[w]
            )

            model.addConstr(
                d["DA", w] <= N - Q[w] + BIGM_2 * (1 - mu["DA", w])
            )

            model.addConstr(
                d["DA", w] <= BIGM_2 * mu["DA", w]
            )
            
            # --------------------------------------------------

    # Deviation constraints for EAM up market

    for v in V_all:
        for w in W[v]:
            # EAM up deviation constraints

            # Dette representerer hvor mye ekstra kraft du mangler i scenario w for å levere DA-leveranse + EAM_down-reduksjon + EAM_up-økning 
            # etter at DA-shortfall (d_DA) og EAM_down-shortfall (d_EAM_down) er tatt hensyn til.

            # Z_w = a_DA[v] - a_EAM_down[w] + a_EAM_up[w] - Q[w] - d_DA[w]
            Z = (
            a["DA", v]
            - a["EAM_down", w]
            + a["EAM_up", w]
            - d["DA", w]
            + d["EAM_down", w]
            - Q[w]
        )

            # 1) Z <= M * eta
            model.addConstr(
                Z <= BIGM_3 * mu["EAM_up", w]
            )

            # 2) Z >= -M * (1 - eta)
            model.addConstr(
                Z >= -BIGM_3 * (1 - mu["EAM_up", w])
            )

            # 3) d_EAM_up >= Z
            model.addConstr(
                d["EAM_up", w] >= Z
            )

            # 4) d_EAM_up <= Z + M * (1 - eta)
            model.addConstr(
                d["EAM_up", w] <= Z + BIGM_3 * (1 - mu["EAM_up", w])
            )

            # 5) d_EAM_up <= M * eta
            model.addConstr(
                d["EAM_up", w] <= BIGM_3 * mu["EAM_up", w]
            )


            


    # Minimum bid quantity constraints for mFRR markets (CM and EAM)
    for (m, s) in b.keys():
        # Hvis b[m,s] = 1  ->  x[m,s] ≥ MIN_Q
        model.addConstr(
            x[m, s] >= x_mFRR_min * b[m, s],
            name=f"mFRR_min_lb[{m},{s}]"
        )

        # Hvis y_bid[m,s] = 0  ->  x[m,s] ≤ 0
        # (og generelt x[m,s] ≤ BIGM hvis y_bid = 1)
        model.addConstr(
            x[m, s] <= BIGM_2 * b[m, s],
            name=f"mFRR_min_ub[{m},{s}]"
        )


    """
    # Constraining bid price in the EAM markets
    #for w in W_all:
    #    model.addConstr(
    #        r["EAM_up", w] <= r_MAX_EAM_up,
    #        name=f"max_price_EAMup_{w}"
    #    )
    #    model.addConstr(
    #        r["EAM_down", w] <= r_MAX_EAM_down,
    #        name=f"max_price_EAMdown_{w}"
    #    )
    for w in W_all:
        model.addConstr(
            r["EAM_up", w] <= r_MAX_EAM_up,
            name=f"max_price_EAMup_{w}"
        )
        model.addConstr(
            r["EAM_down", w] <= r_MAX_EAM_down,
            name=f"max_price_EAMdown_{w}"
        )
    """

    # Constrain bid price within price interval
    for (m, s) in idx_ms:
        if Pmax[m] >= 0:
            model.addConstr(
                r[m, s] <= Pmax[m]
            )
        


    # --- EVALUATE DETERMINISTIC CM POLICY ---
    if evaluate_deterministic_policy:

        with open(det_policy_file, "r") as f:
            det_policy = json.load(f)

        # fiks x,r (CM) til den deterministiske policyen
        for m in det_policy if m in M_u else []:
            for s in S:
                if (m, s) in x:  # sjekk at paret finnes
                    model.addConstr(x[m, s] == det_policy[m]["x"],
                                    name=f"fix_x_{m}_{s}")
                    model.addConstr(r[m, s] == det_policy[m]["r"],
                                    name=f"fix_r_{m}_{s}")

    # --- EVALUATE MODEL WITH ONLY DA AND EAM ---
    if only_da_and_eam:
        for m in M_u:
            for u in U:
                if (m, u) in x:  # sjekk at paret finnes
                    model.addConstr(x[m, u] == 0,
                                    name=f"fix_x_zero_{m}_{s}")
                    model.addConstr(r[m, u] == 0,
                                    name=f"fix_r_zero_{m}_{s}")
    

    print("Added all constraints, starting to optimize model...")
    # --- OPTIMIZE MODEL ---
    
    model.optimize()
    runtime = model.Runtime
    print(f"Model optimized in {runtime:.2f} seconds.")

    model.optimize()

    print("Model is infeasible or unbounded, computing IIS...")
    model.setParam("DualReductions", 0)
    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model_iis.ilp")
        print("Wrote IIS to model_iis.ilp")
    else:
        print("Model is unbounded or still INF_OR_UNBD")

    # --- PRINT RESULTS ---
    if verbose:
        if evaluate_deterministic_policy:
            utils.print_results_deterministic_policy(model, x, a, r, delta, d, Q, U, V, W, M_u, M_v, M_w)
        else:
            utils.print_results(model, x, r, a, delta, d, Q, U, V, W, M_u, M_v, M_w)

    output_dict = {
        "model": model,
        "x": x,
        "r": r,
        "a": a,
        "delta": delta,
        "d": d,
        "Q": Q,
        "U": U,
        "V": V,
        "W": W,
        "M_u": M_u,
        "M_v": M_v,
        "M_w": M_w,
        "objective": model.ObjVal,   # <-- NEW
        "runtime": runtime          # <-- NEW
    }

    return output_dict





def run_robustness_experiment(
    time_str: str,
    n: int,
    num_runs: int = 20,
    base_seed: int | None = None,
    verbose_runs: bool = False,
    **run_model_kwargs,
):
    """
    Runs `run_model` num_runs times and returns averages and variances
    of objective values and runtimes.
    If base_seed is not None, seeds will be base_seed, base_seed+1, ...
    """

    objectives = []
    runtimes = []

    for k in range(num_runs):
        # Optional: different seeds per run
        seed = None
        if base_seed is not None:
            seed = base_seed + k

        res = run_model(
            time_str=time_str,
            n=n,
            verbose=verbose_runs,
            **run_model_kwargs,
            seed=seed,          # assumes run_model has a seed parameter
        )

        obj = res["objective"]
        rt = res["runtime"]

        objectives.append(obj)
        runtimes.append(rt)

        print(f"[RUN {k}] seed={seed}, obj={obj:.4f}, runtime={rt:.4f}s")

    # Use population statistics or sample statistics as you prefer:
    avg_obj = statistics.fmean(objectives)
    std_obj = statistics.pstdev(objectives)   # population std dev

    avg_rt = statistics.fmean(runtimes)
    std_rt = statistics.pstdev(runtimes)

    results = {
        "num_runs": num_runs,
        "n_scenarios": n,
        "avg_objective": avg_obj,
        "var_objective": std_obj,
        "avg_runtime": avg_rt,
        "var_runtime": std_rt,
        "objectives": objectives,
        "runtimes": runtimes,
    }

    print("\n=== ROBUSTNESS SUMMARY ===")
    print(f"Runs         : {num_runs}")
    print(f"Scenarios n  : {n}")
    print(f"Objective avg: {avg_obj:.4f}")
    print(f"Objective std: {std_obj:.4f}")
    print(f"Runtime avg  : {avg_rt:.4f} s")
    print(f"Runtime std  : {std_rt:.6f} s")

    return results


def run_cm_penalty_experiment(
    time_str: str,
    n: int,
    num_runs: int,
    min_multiplier: int = 1,
    max_multiplier: int = 30,
    base_seed: int | None = None,
    verbose_runs: bool = False,
    **run_model_kwargs,
):
    """
    For each penalty multiplier in [min_multiplier, max_multiplier],
    run the model `num_runs` times and record how often the model
    bids in the CM markets (CM_up, CM_down).

    A run is counted as 'CM bid' if any x[CM_up, s] > 0 or x[CM_down, s] > 0
    in the optimal solution, for any scenario s.

    Returns:
        A list of dicts, one per multiplier, with:
            - cm_penalty_multiplier
            - num_runs
            - num_runs_with_cm_bids
            - share_with_cm_bids
            - cm_bid_flags (list[0/1], one entry per run)
    """

    # We'll collect everything here and return only once, at the end
    all_results = []

    eps = 1e-6  # what counts as "positive"

    # Outer loop: over penalty multipliers
    for gamma in range(min_multiplier, max_multiplier + 1):
        cm_bid_flags = []  # store 1/0 for each of the num_runs runs

        if verbose_runs:
            print(f"\n=== Penalty multiplier γ = {gamma} ===")

        # Inner loop: repeated runs for this gamma
        for run_idx in range(num_runs):
            # Construct a seed so different (gamma, run_idx) => different seed
            if base_seed is not None:
                seed = base_seed + (gamma - min_multiplier) * num_runs + run_idx
            else:
                seed = None

            output = run_model(
                time_str=time_str,
                n=n,
                seed=seed,
                cm_penalty_multiplier=float(gamma),
                verbose=verbose_runs,
                **run_model_kwargs,
            )

            x = output["x"]      # tupledict of bids
            M_u = output["M_u"]  # should contain "CM_up" and "CM_down"

            # Check whether this run has any CM bid > 0
            cm_bid_this_run = False
            for (m, s), var in x.items():
                if m in M_u and var.X > eps:
                    cm_bid_this_run = True
                    break

            cm_bid_flags.append(1 if cm_bid_this_run else 0)

            if verbose_runs:
                print(
                    f"  Run {run_idx + 1}/{num_runs}: "
                    f"CM bid = {cm_bid_this_run}"
                )

        num_with_cm_bids = sum(cm_bid_flags)
        share_with_cm_bids = num_with_cm_bids / num_runs if num_runs > 0 else float("nan")

        # Store a full record for this gamma
        all_results.append(
            {
                "cm_penalty_multiplier": gamma,
                "num_runs": num_runs,
                "num_runs_with_cm_bids": num_with_cm_bids,
                "share_with_cm_bids": share_with_cm_bids,
                "cm_bid_flags": cm_bid_flags,  # list of length num_runs
            }
        )

    # ---- After ALL multipliers are done, print the summary for each γ ----
    print("\n=== CM BID FREQUENCY VS PENALTY MULTIPLIER ===")
    for res in all_results:
        gamma = res["cm_penalty_multiplier"]
        num_with = res["num_runs_with_cm_bids"]
        num_total = res["num_runs"]
        share = res["share_with_cm_bids"] * 100.0

        # Keep your original style:
        # gamma1=min_multiplier: (x_1/N) runs with CM bids (fraction one%)
        print(
            f"gamma{gamma}={gamma}: "
            f"({num_with}/{num_total}) runs with CM bids "
            f"({share:.1f} %)"
        )

    # And only now return everything
    return all_results