from functions.data_functions import get_input_data
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import auxiliar_names, plots
from functions.report_functions import generate_report
from functions.utils import *

import os

if __name__ == '__main__':

    # cities_ibge = {'Fortaleza': 230440,
    #                'São Paulo': 355030,
    #                'Maceió': 270430,
    #                'São Luís': 2111300,
    #                "Manaus": 130260,
    #                "Rio de Janeiro": 330455,
    #                "Florianópolis": 420540}
    #
    # city_code = cities_ibge["Florianópolis"]

    covid_parameters, model_parameters, output_parameters = get_input_data(IC_analysis=4, city=130260)

    results = run_SEIR_ODE_model(covid_parameters, model_parameters)

    filename = auxiliar_names(covid_parameters, model_parameters)
    plot_dir = os.path.join(get_output_dir(), f"{filename}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    output_parameters.to_excel(os.path.join(plot_dir, 'parameters_' + filename + '.xlsx'))

    plots(results, covid_parameters, model_parameters, plot_dir)

    # IC_analysis ==  1 -> CONFIDENCE INTERVAL for a lognormal distribution
    # IC_analysis == 2: -> SINGLE RUN
    # IC_analysis == 3 ->  r0 Sensitivity analysis ->
    # 	Calculate an array for r0 to a sensitivity analysis with 0.1 intervals

    if model_parameters.IC_analysis == 2:  # SINGLE RUN
        report = generate_report(results, model_parameters)
        report.to_excel(os.path.join(plot_dir, 'report_' + filename + '.xlsx'), index=False)
        results.to_excel(os.path.join(plot_dir, 'results_' + filename + '.xlsx'), index=False)
