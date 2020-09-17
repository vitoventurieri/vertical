from pathlib import Path
import os
from .report_functions import generate_report, generate_results_percentiles
from datetime import datetime


def get_root_dir():
    return Path(__file__).parent.parent


def get_output_dir():
    return os.path.join(get_root_dir(), 'output')


def auxiliar_names(covid_parameters, model_parameters):
    """
    Provides filename with timestamp and IC_analysis type
    (2: Single Run, 1: Confidence Interval, 3: Sensitivity Analysis)
    as string for files

    :param covid_parameters:
    :param model_parameters:
    :return:
    """

    time = datetime.today()
    time = time.strftime('%Y%m%d%H%M')

    if model_parameters.IC_analysis == 2:  # SINGLE RUN

        beta = covid_parameters.beta  # infectiviy_rate
        gamma = covid_parameters.gamma  # contamination_rate

        basic_reproduction_number = beta / gamma
        r0 = basic_reproduction_number

        filename = (time
                    + '_single_run'
                    + '_r' + ("%.1f" % r0)[0] + '_' + ("%.1f" % r0)[2]
                    + '__g' + ("%.1f" % gamma)[0] + '_' + ("%.1f" % gamma)[2]
                    )
    elif model_parameters.IC_analysis == 1:  # CONFIDENCE INTERVAL
        filename = (time + '_confidence_interval')
    elif model_parameters.IC_analysis == 3:  # SENSITIVITY_ANALYSIS
        filename = (time + '_sensitivity_analysis')
    else:  # Rt analysis
        filename = (time + '_Rt')
    return filename


def get_plot_dir(covid_parameters, model_parameters):
    filename = auxiliar_names(covid_parameters, model_parameters)
    plot_dir = os.path.join(get_output_dir(), f"{filename + model_parameters.city_name[:-2]}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return plot_dir


def export_excel(results, output_parameters, covid_parameters, model_parameters, plot_dir):
    filename = auxiliar_names(covid_parameters, model_parameters)
    output_parameters.to_excel(os.path.join(plot_dir, 'parameters_' + filename +'.xlsx'))
    if model_parameters.analysis == 'Single Run':
        report = generate_report(results, model_parameters)
        report.to_excel(os.path.join(plot_dir, 'report_' + filename + '.xlsx'), index=False)
        results.to_excel(os.path.join(plot_dir, 'results_' + filename + '.xlsx'), index=False)

    elif model_parameters.analysis == 'Confidence Interval' or 'Rt':

        results_percentiles = generate_results_percentiles(results, model_parameters)

        results_percentiles[0][0].to_excel(os.path.join(plot_dir, 'results_medians_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_no_isolation' + filename + '.xlsx'), index=False)

        results_percentiles[1][0].to_excel(os.path.join(plot_dir, 'results_medians_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_vertical_' + filename + '.xlsx'), index=False)
    
    pass
