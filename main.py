from model import run_model
from model import run_robustness_experiment
from benchmark import run_deterministic_benchmark

if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-08 10:00:00+00:00"
    n = 2
    verbose = True
    seed = 15
    number_of_runs = 20
    # run_model(time_str, n, seed, verbose=verbose)
    run_robustness_experiment(time_str, n, number_of_runs, 5)
    #run_deterministic_benchmark(time_str=time_str, n=n)