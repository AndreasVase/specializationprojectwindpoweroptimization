from model import run_model
from benchmark import run_deterministic_benchmark


if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-08 10:00:00+00:00"
    n = 3
    verbose = True
    
    run_model(time_str, n, verbose=verbose)

    #run_deterministic_benchmark(time_str=time_str, n=n)