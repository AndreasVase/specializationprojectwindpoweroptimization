from model import run_model
from benchmark_model import run_benchmark_model
import plots
import tree



path = "./input_data.csv"
det_policy_filename = "deterministic_policy.json"
verbose = False


# Generate deterministic benchmark policy and save to file
print("\n Running deterministic benchmark model to generate policy... \n")
run_benchmark_model(path, det_policy_filename)


# Run the stochastic model
print("\n Running stochastic model...\n")
model, x, r, a, delta, d, U, V, W, M_u, M_v, M_w = run_model(path, det_policy_filename, verbose=verbose)


# Run the stochastic model with the deterministic policy enforced
print("\n Evaluating deterministic CM policy in stochastic model... \n")
det_model, det_x, det_r, det_a, det_delta, det_d, U, V, W, M_u, M_v, M_w = run_model(path, det_policy_filename, evaluate_deterministic_policy=True, verbose=verbose)



# Plot results

# Deterministic vs Stochastic plots
# Plotting the output from the full stochastic model VS the output if the CM-decision is fixed to the deterministic policy.

# objektivverdi
plots.plot_objective_comparison(
    model,
    det_model,
    save_path="fig_cm_policy_obj_comparison.png"
)

# CM-beslutninger (x, a, r)
plots.plot_cm_policy_comparison(
    model,
    x, a, r, delta,
    det_model,
    det_x, det_a, det_r, det_delta,
    U, M_u,
    save_path="fig_cm_policy_comparison.png"
)



# Plot output if using only DA and EAM, current practice in norway, VS all markets.














# Forventet volum per marked fra stokastisk modell
plots.plot_expected_market_volumes_with_stochastic_model(
    model,
    a,
    U, V, W,
    M_u, M_v, M_w,
    save_path="fig_expected_market_volumes_with_stochastic_model.png"
)

# Forventet volum per marked fra deterministisk CM policy i stokastisk modell
plots.plot_expected_market_volumes_with_deterministic_CM_policy(
    det_model,
    det_a,
    U, V, W,
    M_u, M_v, M_w,
    save_path="fig_expected_market_volumes_with_deterministic_CM_policy.png"
)
