from model import run_model
from benchmark_model import run_benchmark_model



path = "./input_data.csv"
det_policy_filename = "deterministic_policy.json"
verbose = False


# Generate deterministic benchmark policy and save to file
print("\n Running deterministic benchmark model to generate policy... \n")
run_benchmark_model(path, det_policy_filename)


# Run the stochastic model
print("\n Running stochastic model...\n")
model, x, r, a, delta, d = run_model(path, det_policy_filename, verbose=verbose)


# Run the stochastic model with the deterministic policy enforced
print("\n Evaluating deterministic policy in stochastic model... \n")
det_model, det_x, det_r, det_a, det_delta, det_d = run_model(path, det_policy_filename, evaluate_deterministic_policy=True, verbose=verbose)