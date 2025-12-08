from model import run_model
from model import run_robustness_experiment
from benchmark import run_deterministic_benchmark
import datetime


def get_strategy_stats(n, seed, verbose, runs_per_time_str):
    start_time = datetime.fromisoformat("2025-10-03 00:00:00+00:00")
    end_time = datetime.fromisoformat("2025-10-14 00:00:00+00:00")

    current_time = start_time

    # Lister med samleverdier for hvert tidspunkt. Lengde er lik antall tidspunkt fra start til slutt.
    strategy_proportion_list = 0

    DA_low_proportion_list = 0
    mean_P_EAMup_DA_low_list = 0
    mean_P_EAMdown_DA_low_list = 0

    DA_high_count = 0
    mean_P_EAMup_DA_high = 0
    mean_P_EAMdown_DA_high = 0

    while current_time < end_time:
        
        time_str = current_time.isoformat()
        
        output_dict = run_model(time_str, n, seed, verbose)

        x = output_dict["x"]
        r = output_dict["r"]
        a = output_dict["a"]
        d = output_dict["d"]
        U = output_dict["U"]
        V = output_dict["V"]
        W = output_dict["W"]

        # Modellen byr høyt i CM, og et tilsvarende bud i EAM, og DA er tilpasset om det er opp eller nedregulering som  er forventet. 
        # Her inngår også situasjoner der CM opp og ned bys samtidig (når pris for opp eller ned med sikkerhet er negativ)
        strategy_count = 0

        DA_low_count = 0
        mean_P_EAMup_DA_low = 0
        mean_P_EAMdown_DA_low = 0

        DA_high_count = 0
        mean_P_EAMup_DA_high = 0
        mean_P_EAMdown_DA_high = 0

        for u in U:
            for v in V(u):
                for w in W(v):

                    x_CMup = x["CM_up", u].X
                    r_CMup = r["CM_up", u].X
                    a_CMup = a["CM_up", u].X
                    d_CMup = d["CM_up", u].X

                    x_CMdown = x["CM_down", u].X
                    r_CMdown = r["CM_down", u].X
                    a_CMdown = a["CM_down", u].X
                    d_CMdown = d["CM_down", u].X

                    x_DA = x["DA", v].X
                    r_DA = r["DA", v].X
                    a_DA = a["DA", v].X
                    d_DA = d["DA", v].X

                    x_EAMup = x["EAM_up", w].X
                    r_EAMup = r["EAM_up", w].X
                    a_EAMup = a["EAM_up", w].X
                    d_EAMup = d["EAM_up", w].X

                    x_EAMdown = x["EAM_down", w].X
                    r_EAMdown = r["EAM_down", w].X
                    a_EAMdown = a["EAM_down", w].X
                    d_EAMdown = d["EAM_down", w].X
                    
                    # Sjekker punkt 1 og 2
                    if (x_CMup > 0 and x_DA == 0 and x_EAMup > 0) or (x_CMdown > 0 and x_DA > 0 and x_EAMdown > 0):


                    if (x_CMup > 0 and x_EAMup > 0 and x_CMdown > 0 and x_EAMdown > 0 and x_DA == 0):

                    
                    if (x_CMup > 0 and x_EAMup > 0 and x_CMdown > 0 and x_EAMdown > 0 and x_DA >= 0):





                    

                


        # Det som skal måles er:

        # I hvor stor andel av tilfellene gjør modellen følgende:

        # 1. Byr høyt i CM opp, 0 i DA, og høyt i EAM opp
        # 2. Byr høyt i CM ned, høyt i DA, og høyt i EAM ned.
        # Punkt 1 og 2 inkluderer gangene der man byr høyt i både opp og ned samtidig 
        # (dvs når prisene i ett av markedene med sikkerhet er negative, og EAM budet med r=0 ikke blir aktivert)
        # punkt 1 og 2 registreres i strategy_count

        # 3. Gjør noe annet enn dette

        # Spesifikt for sakene der modellen byr høyt i både opp og ned-regulering:
        # Hvor ofte gjør modellen følgende, og hva er gjennomsnittsprisene da:

        # 1. Høyt i både opp og ned, men DA lik 0. Registreres i DA_low_count
        # 2. Høyt i både opp og ned, men DA høy. Registreres i DA_high_count




        #current_time += timedelta(hours=1)







if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-09 13:00:00+00:00"
    n = 3
    verbose = True
    seed = 15
    number_of_runs = 20
    # run_model(time_str, n, seed, verbose=verbose)
    # run_robustness_experiment(time_str, n, number_of_runs, 5)
    run_deterministic_benchmark(time_str, n, seed)