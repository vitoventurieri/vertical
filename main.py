import os
import pandas as pd

from functions.data_functions import get_input_data
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import plots
from functions.utils import get_plot_dir, export_excel


if __name__ == '__main__':

    analysis = 'Confidence Interval' #'Rt' # 'Single Run' # 'Sensitivity' #
    # Confidence Interval for a lognormal distribution
    # Single Run
    # Sensitivity: r0 varies with 0.1 intervals
    # Rt: adjust for basic reproduction number for a city over time

    fit_analysis = True  # False #
    runs = 30
    days_to_run = 180
    initial_deaths_to_fit = 50
    city_name = "Rio de Janeiro/RJ" #"São Paulo/SP" #'Manaus/AM'# "Belém/PA" # 'Manaus/AM' # 'Fortaleza/CE'  #

    estimation =  'Sivep'  # 'Verity' #
    
    covid_parameters, model_parameters, output_parameters = get_input_data(analysis,
            fit_analysis, estimation, runs, days_to_run, initial_deaths_to_fit, city_name)

    results = run_SEIR_ODE_model(covid_parameters, model_parameters)

    plot_dir = get_plot_dir(covid_parameters, model_parameters)
   
    plots(results, covid_parameters, model_parameters, plot_dir)

    export_excel(results, output_parameters, covid_parameters, model_parameters, plot_dir)
