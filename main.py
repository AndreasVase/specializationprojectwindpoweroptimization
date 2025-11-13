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
model, x, r, a, delta, d, U, V, W, M_u, M_v, M_w = run_model(path, verbose=verbose)


# Run the stochastic model with the deterministic policy enforced
print("\n Evaluating deterministic CM policy in stochastic model... \n")
det_model, det_x, det_r, det_a, det_delta, det_d, U, V, W, M_u, M_v, M_w = run_model(path, det_policy_filename, evaluate_deterministic_policy=True, verbose=verbose)


# Run the stochastic model with only DA and EAM markets included
print("\n Running stochastic model with only DA and EAM markets included...\n")
da_eam_model, da_eam_x, da_eam_r, da_eam_a, da_eam_delta, da_eam_d, U, V, W, M_u, M_v, M_w = run_model(path, only_da_and_eam=True, verbose=verbose)



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

# Objektivverdi-plot
plots.plot_objective_comparison_full_vs_da_eam(
    model,
    da_eam_model,
    save_path="fig_obj_full_vs_da_eam.png"
)

# Budstrategi-plot per marked
plots.plot_bid_strategy_comparison_full_vs_da_eam(
    x, a, r, delta,
    da_eam_x, da_eam_a, da_eam_r, da_eam_delta,
    U, V, W,
    M_u, M_v, M_w,
    save_path="fig_bid_strategy_full_vs_da_eam.png"
)









# Forventet volum per marked fra stokastisk modell
plots.plot_expected_market_volumes(
    model,
    a,
    U, V, W,
    M_u, M_v, M_w,
    save_path="fig_expected_market_volumes_with_stochastic_model.png"
)

# Forventet volum per marked fra deterministisk CM policy i stokastisk modell
plots.plot_expected_market_volumes(
    det_model,
    det_a,
    U, V, W,
    M_u, M_v, M_w,
    save_path="fig_expected_market_volumes_with_deterministic_CM_policy.png"
)

# Forventet volum per marked med kun DA og EAM i bruk
plots.plot_expected_market_volumes(
    da_eam_model,
    da_eam_a,
    U, V, W,
    M_u, M_v, M_w,
    save_path="fig_expected_market_volumes_with_only_DA_and_EAM.png"
)
