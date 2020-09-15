import os
import pandas as pd

from functions.data_functions import get_input_data
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import auxiliar_names, plots
from functions.report_functions import generate_report, generate_results_percentiles
from functions.utils import get_output_dir


if __name__ == '__main__':

    analysis = 'Confidence Interval' #  'Single Run' # 'Sensitivity' # 'Rt' #
    fit_analysis = True  # False #
    runs = 1
    days_to_run = 180
    initial_deaths_to_fit = 50
    city_name = 'Fortaleza/CE'  # "São Paulo/SP" #

    estimation =  'Sivep'  # 'Verity' #
    
    covid_parameters, model_parameters, output_parameters = get_input_data(analysis,
            fit_analysis, estimation, runs, days_to_run, initial_deaths_to_fit, city_name)

    results = run_SEIR_ODE_model(covid_parameters, model_parameters)

    filename = auxiliar_names(covid_parameters, model_parameters)
    plot_dir = os.path.join(get_output_dir(), f"{filename + city_name[:-2]}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    output_parameters.to_excel(os.path.join(plot_dir, 'parameters_' + filename +'.xlsx'))

    plots(results, covid_parameters, model_parameters, plot_dir)

    # IC_analysis ==  1 -> CONFIDENCE INTERVAL for a lognormal distribution
    # IC_analysis == 2: -> SINGLE RUN
    # IC_analysis == 3 ->  r0 Sensitivity analysis ->
    # 	Calculate an array for r0 to a sensitivity analysis with 0.1 intervals

    if analysis == 'Single Run':
        report = generate_report(results, model_parameters)
        report.to_excel(os.path.join(plot_dir, 'report_' + filename + '.xlsx'), index=False)
        results.to_excel(os.path.join(plot_dir, 'results_' + filename + '.xlsx'), index=False)

    elif model_parameters.IC_analysis == 'Confidence Interval' or 'Rt':

        results_percentiles = generate_results_percentiles(results, model_parameters)

        results_percentiles[0][0].to_excel(os.path.join(plot_dir, 'results_medians_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_no_isolation' + filename + '.xlsx'), index=False)

        results_percentiles[1][0].to_excel(os.path.join(plot_dir, 'results_medians_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_vertical_' + filename + '.xlsx'), index=False)
