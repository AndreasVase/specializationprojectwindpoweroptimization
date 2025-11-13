


def add_activation_constraints(model, idx_ms, P, x, r, a, delta, BIGM, epsilon):
    """
    Adds activation constraints:
        1) a_ms <= x_ms
        2) a_ms <= BIGM * delta_ms
        3) a_ms >= x_ms - BIGM * (1 - delta_ms)
    """

    for (m, s) in idx_ms:

        # 1) a_ms <= x_ms
        model.addConstr(
            a[m, s] <= x[m, s],
            name=f"act_le_bid[{m},{s}]"
        )

        # 2) a_ms <= BIGM * delta_ms
        model.addConstr(
            a[m, s] <= BIGM * delta[m, s],
            name=f"act_le_Mdelta[{m},{s}]"
        )

        # 3) a_ms >= x_ms - BIGM * (1 - delta_ms)
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