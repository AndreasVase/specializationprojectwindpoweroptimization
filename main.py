from model import run_model
from model import run_robustness_experiment
from model import run_cm_penalty_experiment
from benchmark import run_deterministic_benchmark

if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-04 00:00:00+00:00"
    n = 4
    verbose = True
    seed = 1000
    #number_of_runs = 20
    #run_model(time_str, n, seed, verbose=verbose)
    # run_robustness_experiment(time_str, n, number_of_runs, 5)
    #run_deterministic_benchmark(time_str, n, seed)
    results = run_cm_penalty_experiment(
        time_str=time_str,
        n=n,
        num_runs=20,
        min_multiplier=2,
        max_multiplier=6,
        base_seed=seed,
        verbose_runs=False,
    )