from model import run_model
from benchmark_model import run_benchmark_model
import plots
import tree

# Run the stochastic model
def run_stochastic_model(n, time_str, verbose=True):
    print("\n Running stochastic model...\n")
    output_dict = run_model(time_str, n, verbose=verbose)
    return output_dict

# Run the stochastic model with the deterministic policy enforced
def run_deterministic_policy_evaluation():

    run_benchmark_model(path, det_policy_filename)

    print("\n Evaluating deterministic CM policy in stochastic model... \n")
    det_output_dict = run_model(path, det_policy_filename, evaluate_deterministic_policy=True, verbose=verbose)
    return det_output_dict


# Run the stochastic model with only DA and EAM markets included
def run_da_eam_only_model():
    print("\n Running stochastic model with only DA and EAM markets included...\n")
    da_eam_output_dict = run_model(path, only_da_and_eam=True, verbose=verbose)
    return da_eam_output_dict


# -----------------------------------------------------------------
# Deterministic CM policy vs Stochastic soulution
# -----------------------------------------------------------------

def generate_deterministic_policy_plots(output_dict, det_output_dict):
    """
    Genererer plots som sammenligner den stokastiske løsningen med løsningen
    der en deterministisk CM-policy er påtvunget.
    """
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

def generate_da_eam_comparison_plots(output_dict, da_eam_output_dict):
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



if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-04 10:00:00+00:00"
    n = 2
    det_policy_filename = "deterministic_policy.json"
    verbose = True
    run_stochastic_model(n, time_str, verbose=verbose)