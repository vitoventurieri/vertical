from functions.data_functions import get_input_data, get_ibge_code
from functions.model_functions import run_SEIR_ODE_model
from functions.plot_functions import auxiliar_names, plots
from functions.report_functions import generate_report
from functions.utils import *
import numpy as np
import matplotlib.pyplot as plt

import os 

if __name__ == '__main__':

	city = 'Fortaleza'

	state = 'CE'

	removed_beginnings = np.arange(5,25)   # range de porcentagens de imunidade cruzada

	dists = []

	city_code = get_ibge_code(city, state)

	i_dist = 0

	for removed_initial in removed_beginnings:

		covid_parameters, model_parameters, output_parameters = get_input_data(IC_analysis = 1, city=city_code, removed_init=removed_initial/100)

		results = run_SEIR_ODE_model(covid_parameters, model_parameters)

		filename = auxiliar_names(covid_parameters, model_parameters)
		plot_dir = os.path.join(get_output_dir(), f"{filename}")
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)

		output_parameters.to_excel(os.path.join(plot_dir, 'parameters_' + filename + '.xlsx'))

		dist = plots(results, covid_parameters, model_parameters, plot_dir, place=city+'/'+state)
		print(dist)
		dists.append(dist)
		print(dists)
		i_dist +=1

		#IC_analysis ==  1 -> CONFIDENCE INTERVAL for a lognormal distribution
		#IC_analysis == 2: -> SINGLE RUN
		#IC_analysis == 3 ->  r0 Sensitivity analysis ->
		# 	Calculate an array for r0 to a sensitivity analysis with 0.1 intervals

		if model_parameters.IC_analysis == 2: # SINGLE RUN
			report = generate_report(results, model_parameters)
			report.to_excel(os.path.join(plot_dir, 'report_' + filename + '.xlsx'), index=False)
			results.to_excel(os.path.join(plot_dir, 'results_' + filename + '.xlsx'), index=False)

	print(model_parameters.contact_reduction_elderly)
	#mudança
	dists = np.array(dists)
	plt.plot(removed_beginnings, dists)
	plt.xlabel("Porcentagem Imune Inicialmente (%)")
	plt.ylabel("Distância Entre Predito e Observado")
	plt.legend(model_parameters.contact_reduction_elderly)
	plt.show()

