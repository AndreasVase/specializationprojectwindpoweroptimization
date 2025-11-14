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
output_dict = run_model(path, verbose=verbose)

# Run the stochastic model with the deterministic policy enforced
print("\n Evaluating deterministic CM policy in stochastic model... \n")
det_output_dict = run_model(path, det_policy_filename, evaluate_deterministic_policy=True, verbose=verbose)


# Run the stochastic model with only DA and EAM markets included
print("\n Running stochastic model with only DA and EAM markets included...\n")
da_eam_output_dict = run_model(path, only_da_and_eam=True, verbose=verbose)





# -----------------------------------------------------------------
# Deterministic CM policy vs Stochastic soulution
# -----------------------------------------------------------------

# objektivverdi
plots.plot_model_type_objective_comparison(
    output_dict,
    det_output_dict,
    save_path="fig_model_type_obj.png"
)

# CM-beslutninger (x, a, r)
plots.plot_model_type_policy_comparison(
    output_dict,
    det_output_dict,
    save_path="fig_model_type_policy.png"
)

# --- Forventet volum per marked Stochastic vs deterministic CM-policy ---
plots.plot_expected_a(
    output_dict,
    det_output_dict,
    label1="Stochastic solution", 
    label2="Deterministic CM policy",
    save_path="fig_model_type_expected_a.png"
)



# -----------------------------------------------------------------
# CM+DA+EAM vs DA+EAM only
# -----------------------------------------------------------------

# Objektivverdi-plot
plots.plot_market_attendance_objective_comparison(
    output_dict,
    da_eam_output_dict,
    save_path="fig_market_attendance_obj_comparison.png"
)

# Budstrategi-plot per marked
plots.plot_market_attendance_expected_x(
    output_dict,
    da_eam_output_dict,
    save_path="fig_market_attendance_expected_x.png"
)

# Budstrategi-plot per marked
plots.plot_market_attendance_expected_r(
    output_dict, 
    da_eam_output_dict,
    save_path="fig_market_attendance_expected_r.png"
)


# --- Forventet volum per marked Stochastic vs DA+EAM-only ---
plots.plot_expected_a(
    output_dict, 
    da_eam_output_dict,
    label1="Strategy: CM+DA+EAM",
    label2="Strategy: DA+EAM",
    save_path="fig_market_attendance_expected_a.png"
)





