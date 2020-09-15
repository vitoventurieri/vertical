from functions.data_functions import get_input_data, define_city, fix_city_code, fix_city_name
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import auxiliar_names, plots
from functions.report_functions import generate_report, generate_results_percentiles
from functions.utils import *
import os

if __name__ == '__main__':


    city = define_city().cidade

    nome_cidade = define_city().city_name

    IC_analysis = define_city().icanalisis
    

    covid_parameters, model_parameters, output_parameters = get_input_data(IC_analysis=IC_analysis, city=city)

    results = run_SEIR_ODE_model(covid_parameters, model_parameters)

    filename = auxiliar_names(covid_parameters, model_parameters)
    plot_dir = os.path.join(get_output_dir(), f"{filename + '_' + nome_cidade}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    output_parameters.to_excel(os.path.join(plot_dir, 'parameters_' + filename +'.xlsx'))

    plots(results, covid_parameters, model_parameters, plot_dir)

    # IC_analysis ==  1 -> CONFIDENCE INTERVAL for a lognormal distribution
    # IC_analysis == 2: -> SINGLE RUN
    # IC_analysis == 3 ->  r0 Sensitivity analysis ->
    # 	Calculate an array for r0 to a sensitivity analysis with 0.1 intervals

    if model_parameters.IC_analysis == 2:  # SINGLE RUN
        report = generate_report(results, model_parameters)
        report.to_excel(os.path.join(plot_dir, 'report_' + filename + '.xlsx'), index=False)
        results.to_excel(os.path.join(plot_dir, 'results_' + filename + '.xlsx'), index=False)

    elif model_parameters.IC_analysis == 1 or 4:

        results_percentiles = generate_results_percentiles(results, model_parameters)

        results_percentiles[0][0].to_excel(os.path.join(plot_dir, 'results_medians_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_no_isolation' + filename + '.xlsx'), index=False)

        results_percentiles[1][0].to_excel(os.path.join(plot_dir, 'results_medians_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_vertical_' + filename + '.xlsx'), index=False)

        #report = generate_report(results_percentiles[0], model_parameters)
        #report.to_excel(os.path.join(plot_dir, 'report_' + filename + '.xlsx'), index=False)
