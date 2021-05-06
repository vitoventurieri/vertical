from pathlib import Path
import os
import pandas as pd
import numpy as np
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
        
        resultsTemp = generate_results_percentiles(results, model_parameters)
        results_percentiles = resultsTemp[0]
        Hmax = resultsTemp[1]
        Umax = resultsTemp[2]

        #results_percentiles = generate_results_percentiles(results, model_parameters)

        results_percentiles[0][0].to_excel(os.path.join(plot_dir, 'results_medians_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_no_isolation_' + filename + '.xlsx'), index=False)
        results_percentiles[0][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_no_isolation' + filename + '.xlsx'), index=False)

        results_percentiles[1][0].to_excel(os.path.join(plot_dir, 'results_medians_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][1].to_excel(os.path.join(plot_dir, 'results_percentile_05_vertical_' + filename + '.xlsx'), index=False)
        results_percentiles[1][2].to_excel(os.path.join(plot_dir, 'results_percentile_95_vertical_' + filename + '.xlsx'), index=False)

        list_of_last_day_results_no_isolation = [results_percentiles[0][0].iloc[-1:,:],results_percentiles[0][1].iloc[-1:,:],results_percentiles[0][2].iloc[-1:,:]]
        last_day_results_no_isolation = pd.concat(list_of_last_day_results_no_isolation)
        last_day_results_no_isolation.to_excel(os.path.join(plot_dir, 'results_last_day_median_05_95_no_isolation_' + filename + '.xlsx'), index=False)

        list_of_last_day_results_vertical = [results_percentiles[1][0].iloc[-1:,:],results_percentiles[1][1].iloc[-1:,:],results_percentiles[1][2].iloc[-1:,:]]
        last_day_results_vertical = pd.concat(list_of_last_day_results_vertical)
        last_day_results_vertical.to_excel(os.path.join(plot_dir, 'results_last_day_median_05_95_vertical_' + filename + '.xlsx'), index=False)

        outcome_comparisson = {}
        outcome_comparisson['vertical'] = last_day_results_vertical
        outcome_comparisson['no_isolation'] = last_day_results_no_isolation

        compartment_list_outcome = ["Ri", 'Rj', 'Mi', 'Mj']
        compartment_dict_outcome = {}
        reduction = {}

        # Do some comparisons between no_isolation vs vertical

        with open(os.path.join(plot_dir, 'comparison_vertical_vs_no_isolation.txt'), 'a') as f:
            cont = 1
            for i in outcome_comparisson:

                print(i, file=f)
                print('', file=f)

                for a in compartment_list_outcome:
                    compartment_dict_outcome[a] = round(outcome_comparisson[i].loc[:, a].median()), round(
                        outcome_comparisson[i].loc[:, a].min()), round(outcome_comparisson[i].loc[:, a].max())

                    print(a + ' median ' + str(compartment_dict_outcome[a][0]) + ' (05th-95th percentile: ' + str(
                        compartment_dict_outcome[a][1]) + '-' + str(compartment_dict_outcome[a][2]) + ')', file=f)
                    reduction[a, i] = np.array(compartment_dict_outcome[a])
                
                print('H peak ' + str(round(Hmax[cont][0])) + ' (05th-95th percentile: ' + str(
                        round(Hmax[cont][1])) + '-' + str(round(Hmax[cont][2])) + ')', file=f)
                print('U peak median ' + str(round(Umax[cont][0])) + ' (05th-95th percentile: ' + str(
                        round(Umax[cont][1])) + '-' + str(round(Umax[cont][2])) + ')', file=f)
                cont = cont - 1
                
                print('', file=f)
            print('Reduction in cases by age group , median, 05th percentile, 95th percentile', file=f)
            for a in compartment_list_outcome:
                print(str(a) + ' ' + str(np.round((1 - reduction[a, 'vertical'] / reduction[a, 'no_isolation']), 4)),
                      file=f)

            print('', file=f)
            print('Reduction in total cases median', file=f)

            removed_reduction = round(1 - (
                        outcome_comparisson['vertical'].loc[:, 'Ri'].median() + outcome_comparisson['vertical'].loc[:,
                                                                                'Rj'].median()) / (
                                                  outcome_comparisson['no_isolation'].loc[:, 'Ri'].median() +
                                                  outcome_comparisson['no_isolation'].loc[:, 'Rj'].median()), 4)
            death_reduction = round(1 - (
                        outcome_comparisson['vertical'].loc[:, 'Mi'].median() + outcome_comparisson['vertical'].loc[:,
                                                                                'Mj'].median()) / (
                                                outcome_comparisson['no_isolation'].loc[:, 'Mi'].median() +
                                                outcome_comparisson['no_isolation'].loc[:, 'Mj'].median()), 4)
            
            
            # reduction['vertical']/reduction['no_isolation']

            print('', file=f)

            print('Removed ' + str(removed_reduction), file=f)
            print('Deaths ' + str(death_reduction), file=f)
            print('', file=f)

            print('Reduction in total cases 05th percentile', file=f)
            print('', file=f)

            removed_reduction = round(1 - (
                        outcome_comparisson['vertical'].loc[:, 'Ri'].min() + outcome_comparisson['vertical'].loc[:,
                                                                             'Rj'].min()) / (
                                                  outcome_comparisson['no_isolation'].loc[:, 'Ri'].min() +
                                                  outcome_comparisson['no_isolation'].loc[:, 'Rj'].min()), 4)
            death_reduction = round(1 - (
                        outcome_comparisson['vertical'].loc[:, 'Mi'].min() + outcome_comparisson['vertical'].loc[:,
                                                                             'Mj'].min()) / (
                                                outcome_comparisson['no_isolation'].loc[:, 'Mi'].min() +
                                                outcome_comparisson['no_isolation'].loc[:, 'Mj'].min()), 4)
            # reduction['vertical']/reduction['no_isolation']

            print('Removed ' + str(removed_reduction), file=f)
            print('Deaths ' + str(death_reduction), file=f)
            print('', file=f)

            print('Reduction in total cases 95th percentile', file=f)
            print('', file=f)

            removed_reduction = round(1 - (
                        outcome_comparisson['vertical'].loc[:, 'Ri'].max() + outcome_comparisson['vertical'].loc[:,
                                                                             'Rj'].max()) / (
                                                  outcome_comparisson['no_isolation'].loc[:, 'Ri'].max() +
                                                  outcome_comparisson['no_isolation'].loc[:, 'Rj'].max()), 4)
            death_reduction = round(1 - (
                        outcome_comparisson['vertical'].loc[:, 'Mi'].max() + outcome_comparisson['vertical'].loc[:,
                                                                             'Mj'].max()) / (
                                                outcome_comparisson['no_isolation'].loc[:, 'Mi'].max() +
                                                outcome_comparisson['no_isolation'].loc[:, 'Mj'].max()), 4)
            # reduction['vertical']/reduction['no_isolation']

            print('Removed ' + str(removed_reduction), file=f)
            print('Deaths ' + str(death_reduction), file=f)

    pass
