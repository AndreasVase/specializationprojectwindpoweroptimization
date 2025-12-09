from model import run_model
from model import run_robustness_experiment
from benchmark import run_deterministic_benchmark
from datetime import datetime
from datetime import timedelta
import numpy as np
from gurobipy import GRB
import utils
import read


def get_strategy_stats(n, seed, verbose, runs_per_time_str):
    start_time = datetime.fromisoformat("2025-10-04 00:00:00+00:00")
    end_time = datetime.fromisoformat("2025-10-14 00:00:00+00:00")

    current_time = start_time

    # Lister med samleverdier for hvert tidspunkt. Lengde er lik antall tidspunkt fra start til slutt.
    case_1_count_global = 0
    num_timestrs = 0

    case_2_1_count_global = 0
    mean_P_EAMup_2_1_global = []
    mean_P_EAMdown_2_1_global = []

    case_2_2_count_global = 0
    mean_P_EAMup_2_2_global = []
    mean_P_EAMdown_2_2_global = []

    while current_time < end_time:
        
        time_str = current_time.isoformat()
        
        output_dict = run_model(time_str, n, seed, verbose)

        model = output_dict["model"]
        x = output_dict["x"]
        r = output_dict["r"]
        a = output_dict["a"]
        delta = output_dict["delta"]
        d = output_dict["d"]
        U = output_dict["U"]
        V = output_dict["V"]
        W = output_dict["W"]
        P = output_dict["P"]
        Q = output_dict["Q"]
        M1 = output_dict["M1"]
        M2 = output_dict["M2"]
        M3 = output_dict["M3"]

        if model.Status != GRB.OPTIMAL:
            current_time += timedelta(hours=1)
            continue


        # Modellen byr høyt i CM, og et tilsvarende bud i EAM, og DA er tilpasset om det er opp eller nedregulering som  er forventet. 
        # Her inngår også situasjoner der CM opp og ned bys samtidig (når pris for opp eller ned med sikkerhet er negativ)
        
        num_scenarios = 0

        case_1_count = 0

        case_2_1_count = 0
        P_EAMup_2_1_sum = 0
        P_EAMdown_2_1_sum = 0

        case_2_2_count = 0
        P_EAMup_2_2_sum = 0
        P_EAMdown_2_2_sum = 0

        for u in U:
            for v in V[u]:
                for w in W[v]:

                    x_CMup = x["CM_up", u].X
                    r_CMup = r["CM_up", u].X
                    a_CMup = a["CM_up", u].X
                    d_CMup = d["CM_up", w].X

                    x_CMdown = x["CM_down", u].X
                    r_CMdown = r["CM_down", u].X
                    a_CMdown = a["CM_down", u].X
                    d_CMdown = d["CM_down", w].X

                    x_DA = x["DA", v].X
                    r_DA = r["DA", v].X
                    a_DA = a["DA", v].X
                    d_DA = d["DA", w].X

                    x_EAMup = x["EAM_up", w].X
                    r_EAMup = r["EAM_up", w].X
                    a_EAMup = a["EAM_up", w].X
                    d_EAMup = d["EAM_up", w].X

                    x_EAMdown = x["EAM_down", w].X
                    r_EAMdown = r["EAM_down", w].X
                    a_EAMdown = a["EAM_down", w].X
                    d_EAMdown = d["EAM_down", w].X

                    # Sjekker case 1
                    if (x_CMup > 0 and x_DA == 0 and x_EAMup > 0) or (x_CMdown > 0 and x_DA > 0 and x_EAMdown > 0):
                        case_1_count += 1
                    else:
                        #print("Not a case 1 result:")
                        #print(f"CM_up    –––– x = {x_CMup} –––– r = {r_CMup} –––– a = {a_CMup} –––– d = {d_CMup}")
                        #print(f"CM_down  –––– x = {x_CMdown} –––– r = {r_CMdown} –––– a = {a_CMdown} –––– d = {d_CMdown}")
                        #print(f"DA       –––– x = {x_DA} –––– r = {r_DA} –––– a = {a_DA} –––– d = {d_DA}")
                        #print(f"EAM_up   –––– x = {x_EAMup} –––– r = {r_EAMup} –––– a = {a_EAMup} –––– d = {d_EAMup}")
                        #print(f"EAM_down –––– x = {x_EAMdown} –––– r = {r_EAMdown} –––– a = {a_EAMdown} –––– d = {d_EAMdown}")
                        CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed, picked_scenario_indices = read.load_parameters_from_parquet(time_str, n, seed)
                        utils.write_results_to_file(
                            "not_case_1_results.txt",
                            model, x, r, a, delta, d, Q, CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed,
                            U, V, W, M1, M2, M3,
                            max_u=2, max_v_per_u=2, max_w_per_v=2
                        )
                    
                    if (x_CMup > 0 and x_EAMup > 0 and x_CMdown > 0 and x_EAMdown > 0):
                        # Sjekker case 2_1
                        if x_DA == 0:
                            case_2_1_count += 1
                            
                            P_EAMup_2_1_sum += P["EAM_up", w]
                            P_EAMdown_2_1_sum += P["EAM_down", w]

                        # Sjekker case 2_2
                        if x_DA > 0:
                            case_2_2_count += 1
#
                            P_EAMup_2_2_sum += P["EAM_up", w]
                            P_EAMdown_2_2_sum += P["EAM_down", w]
                        
                    

                    num_scenarios += 1

        case_1 = case_1_count/num_scenarios
        print("case_1: ", case_1)
        
        case_2_1 = case_2_1_count/num_scenarios
        print("")
        print("case 2_1: ", case_2_1)
        if case_2_1_count > 0:    
            mean_P_EAMup_2_1 = P_EAMup_2_1_sum/case_2_1_count
            mean_P_EAMdown_2_1 = P_EAMdown_2_1_sum/case_2_1_count
            
            mean_P_EAMup_2_1_global.append(mean_P_EAMup_2_1)
            mean_P_EAMdown_2_1_global.append(mean_P_EAMdown_2_1)

            print("Mean EAMup price: ", mean_P_EAMup_2_1)
            print("Mean EAMdown price: ", mean_P_EAMdown_2_1)

        case_2_2 = case_2_2_count/num_scenarios
        print("") 
        print("case 2_2: ", case_2_2)
        if case_2_2_count > 0:
            mean_P_EAMup_2_2 = P_EAMup_2_2_sum/case_2_2_count
            mean_P_EAMdown_2_2 = P_EAMdown_2_2_sum/case_2_2_count

            mean_P_EAMup_2_2_global.append(mean_P_EAMup_2_2)
            mean_P_EAMdown_2_2_global.append(mean_P_EAMdown_2_2)
            print("Mean EAMup price: ", mean_P_EAMup_2_2)
            print("Mean EAMdown price: ", mean_P_EAMdown_2_2)


        case_1_count_global += case_1

        case_2_1_count_global += case_2_1

        case_2_2_count_global += case_2_2




        num_timestrs += 1
        
                    
        current_time += timedelta(hours=1)
    
    print("")
    print("GLOBAL MEAN")
    print("")
    print("Case 1 count: ", case_1_count_global/num_timestrs)
    print("")
    print("Case 2_1 count: ", case_2_1_count_global/num_timestrs)
    print("Mean EAMup price: ", np.mean(mean_P_EAMup_2_1_global))
    print("EAMup price [min, max] ", np.min(mean_P_EAMup_2_1_global), np.max(mean_P_EAMup_2_1_global))
    print("Mean EAMdown price: ", np.mean(mean_P_EAMdown_2_1_global))
    print("EAMdown price [min, max] ", np.min(mean_P_EAMdown_2_1_global), np.max(mean_P_EAMdown_2_1_global))

    print("")
    print("Case 2_2 count: ", case_2_2_count_global/num_timestrs)
    print("Mean EAMup price: ", np.mean(mean_P_EAMup_2_2_global))
    print("EAMup price [min, max] ", np.min(mean_P_EAMup_2_2_global), np.max(mean_P_EAMup_2_2_global))
    print("Mean EAMdown price: ", np.mean(mean_P_EAMdown_2_2_global))
    print("EAMdown price [min, max] ", np.min(mean_P_EAMdown_2_2_global), np.max(mean_P_EAMdown_2_2_global))


                    

                


        # Det som skal måles er:

        # Case 1: I hvor stor andel av tilfellene gjør modellen følgende:

        # 1. Byr høyt i CM opp, 0 i DA, og høyt i EAM opp
        # 2. Byr høyt i CM ned, høyt i DA, og høyt i EAM ned.
        # Punkt 1 og 2 inkluderer gangene der man byr høyt i både opp og ned samtidig 
        # (dvs når prisene i ett av markedene med sikkerhet er negative, og EAM budet med r=0 ikke blir aktivert)
        # punkt 1 og 2 registreres i strategy_count
        # 3. Gjør noe annet enn dette

        # Case 2: Spesifikt for casene der modellen byr høyt i både opp og ned-regulering:
        # Hvor ofte gjør modellen følgende, og hva er gjennomsnittsprisene da:

        # 2.1. Høyt i både opp og ned, men DA lik 0. Registreres i DA_low_count
        # 2.2. Høyt i både opp og ned, men DA høy. Registreres i DA_high_count












if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-05 09:00:00+00:00"
    n = 4
    verbose = False
    seed = 10
    number_of_runs = 21
    #run_model(time_str, n, seed, verbose=verbose)
    # run_robustness_experiment(time_str, n, number_of_runs, 5)
    #run_deterministic_benchmark(time_str, n, seed)
    get_strategy_stats(n, seed, False, 1)
