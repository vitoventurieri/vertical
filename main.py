from functions.data_functions import get_input_data
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import auxiliar_names, plots
from functions.report_functions import generate_report
from functions.utils import *

import os 

if __name__ == '__main__':

	covid_parameters, model_parameters = get_input_data()

	results = run_SEIR_ODE_model(covid_parameters, model_parameters)
	report = generate_report(results, model_parameters)

	filename = auxiliar_names(covid_parameters, model_parameters)
	report.to_excel(os.path.join(get_output_dir(), 'report_' + filename + '.xlsx'), index=False)
	results.to_excel(os.path.join(get_output_dir(), 'results_' + filename + '.xlsx'), index=False) 
	
	plot_dir = os.path.join(get_output_dir(), f"{filename}_plots")

	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)

	plots(results, model_parameters, plot_dir)
