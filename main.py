import os
import pandas as pd

from functions.data_functions import get_input_data
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import plots
from functions.utils import get_plot_dir, export_excel


### FOR ARTICLE RESULTS USE: analysis = 'Confidence Interval'
# fit_analysis = False
# runs = 1000
# days_to_run = 250
# initial_deaths_to_fit = ##any value as fit analisys is false
# estimation = 'Sivep'

# For article graphs for beds, remove the constraints on beds on model_functions, by commenting/uncommenting the lines for each scenario (Bed demand - WITH constraints or NO constraints)


if __name__ == '__main__':

    city_list = ['Porto Velho/RO',
                'Manaus/AM',
                'Rio Branco/AC',
                'Campo Grande/MS ',
                'Macapá/AP',
                'Brasília/DF',
                'Boa Vista/RR',
                'Cuiabá/MT',
                'Palmas/TO',
                'São Paulo/SP',
                'Teresina/PI',
                'Rio de Janeiro/RJ',
                'Belém/PA',
                'Goiânia/GO',
                'Salvador/BA',
                'Florianópolis/SC',
                'São Luís/MA',
                'Maceió/AL',
                'Porto Alegre/RS ',
                'Curitiba/PR',
                'Belo Horizonte/MG',
                'Fortaleza/CE',
                'Recife/PE',
                'João Pessoa/PB',
                'Aracaju/SE',
                'Natal/RN',
                'Vitória/ES']

    for chosen_city in city_list:

        analysis ='Confidence Interval' #'Single Run'#'Rt'  # 'Sensitivity' #

        # Confidence Interval for a lognormal distribution
        # Single Run
        # Sensitivity: r0 varies with 0.1 intervals
        # Rt: adjust for basic reproduction number for a city over time

        fit_analysis =  False #True
        runs = 1000
        days_to_run = 250
        initial_deaths_to_fit = 1
        city_name = chosen_city

        estimation = 'Sivep'  # 'Verity'

        covid_parameters, model_parameters, output_parameters = get_input_data(analysis,
                fit_analysis, estimation, runs, days_to_run, initial_deaths_to_fit, city_name)

        results = run_SEIR_ODE_model(covid_parameters, model_parameters)

        plot_dir = get_plot_dir(covid_parameters, model_parameters)

        plots(results, covid_parameters, model_parameters, plot_dir)

        export_excel(results, output_parameters, covid_parameters, model_parameters, plot_dir)

