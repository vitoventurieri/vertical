import pandas as pd
import numpy as np

def generate_results_percentiles(results, model_parameters):
	results_percentiles = {}
	for i in range(len(model_parameters.isolation_level)):
		isolation_name_i = model_parameters.isolation_level[i]

		Si = np.zeros((len(results), model_parameters.t_max))
		Sj = np.zeros((len(results), model_parameters.t_max))
		Ei = np.zeros((len(results), model_parameters.t_max))
		Ej = np.zeros((len(results), model_parameters.t_max))
		Ii = np.zeros((len(results), model_parameters.t_max))
		Ij = np.zeros((len(results), model_parameters.t_max))
		Ri = np.zeros((len(results), model_parameters.t_max))
		Rj = np.zeros((len(results), model_parameters.t_max))
		Hi = np.zeros((len(results), model_parameters.t_max))
		Hj = np.zeros((len(results), model_parameters.t_max))
		Ui = np.zeros((len(results), model_parameters.t_max))
		Uj = np.zeros((len(results), model_parameters.t_max))
		Mi = np.zeros((len(results), model_parameters.t_max))
		Mj = np.zeros((len(results), model_parameters.t_max))

		dHi = np.zeros((len(results), model_parameters.t_max))
		dHj = np.zeros((len(results), model_parameters.t_max))
		dUi = np.zeros((len(results), model_parameters.t_max))
		dUj = np.zeros((len(results), model_parameters.t_max))
		pHi = np.zeros((len(results), model_parameters.t_max))
		pHj = np.zeros((len(results), model_parameters.t_max))
		pUi = np.zeros((len(results), model_parameters.t_max))
		pUj = np.zeros((len(results), model_parameters.t_max))
		pMi = np.zeros((len(results), model_parameters.t_max))
		pMj = np.zeros((len(results), model_parameters.t_max))

		WARD_survive_i = np.zeros((len(results), model_parameters.t_max))
		WARD_survive_j = np.zeros((len(results), model_parameters.t_max))
		WARD_death_i = np.zeros((len(results), model_parameters.t_max))
		WARD_death_j = np.zeros((len(results), model_parameters.t_max))
		ICU_survive_i = np.zeros((len(results), model_parameters.t_max))
		ICU_survive_j = np.zeros((len(results), model_parameters.t_max))
		ICU_death_i = np.zeros((len(results), model_parameters.t_max))
		ICU_death_j = np.zeros((len(results), model_parameters.t_max))
		WARD_discharged_ICU_survive_i = np.zeros((len(results), model_parameters.t_max))
		WARD_discharged_ICU_survive_j = np.zeros((len(results), model_parameters.t_max))


		for ii in range(len(results)):
			query_condition = 'isolamento == @isolation_name_i'

			Si[ii,] = results[ii].query(query_condition)['Si']
			Sj[ii,] = results[ii].query(query_condition)['Sj']
			Ei[ii,] = results[ii].query(query_condition)['Ei']
			Ej[ii,] = results[ii].query(query_condition)['Ej']
			Ii[ii,] = results[ii].query(query_condition)['Ii']
			Ij[ii,] = results[ii].query(query_condition)['Ij']
			Ri[ii,] = results[ii].query(query_condition)['Ri']
			Rj[ii,] = results[ii].query(query_condition)['Rj']
			Hi[ii,] = results[ii].query(query_condition)['Hi']
			Hj[ii,] = results[ii].query(query_condition)['Hj']
			Ui[ii,] = results[ii].query(query_condition)['Ui']
			Uj[ii,] = results[ii].query(query_condition)['Uj']
			Mi[ii,] = results[ii].query(query_condition)['Mi']
			Mj[ii,] = results[ii].query(query_condition)['Mj']
			pHi[ii,] = results[ii].query(query_condition)['pHi']

			dHi[ii,] = results[ii].query(query_condition)['dHi']
			dHj[ii,] = results[ii].query(query_condition)['dHj']
			dUi[ii,] = results[ii].query(query_condition)['dUi']
			dUj[ii,] = results[ii].query(query_condition)['dUj']
			pHi[ii,] = results[ii].query(query_condition)['pHi']
			pHj[ii,] = results[ii].query(query_condition)['pHj']
			pUi[ii,] = results[ii].query(query_condition)['pUi']
			pUj[ii,] = results[ii].query(query_condition)['pUj']
			pMi[ii,] = results[ii].query(query_condition)['pMi']
			pMj[ii,] = results[ii].query(query_condition)['pMj']

			WARD_survive_i[ii,] = results[ii].query(query_condition)['WARD_survive_i']
			WARD_survive_j[ii,] = results[ii].query(query_condition)['WARD_survive_j']
			WARD_death_i[ii,] = results[ii].query(query_condition)['WARD_death_i']
			WARD_death_j[ii,] = results[ii].query(query_condition)['WARD_death_j']
			ICU_survive_i[ii,] = results[ii].query(query_condition)['ICU_survive_i']
			ICU_survive_j[ii,] = results[ii].query(query_condition)['ICU_survive_j']
			ICU_death_i[ii,] = results[ii].query(query_condition)['ICU_death_i']
			ICU_death_j[ii,] = results[ii].query(query_condition)['ICU_death_j']
			WARD_discharged_ICU_survive_i[ii,] = results[ii].query(query_condition)['WARD_discharged_ICU_survive_i']
			WARD_discharged_ICU_survive_j[ii,] = results[ii].query(query_condition)['WARD_discharged_ICU_survive_j']

		compartiments_list = [Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj, dHi, dHj, dUi, dUj, pHi, pHj, pUi, pUj, pMi, pMj, WARD_survive_i, WARD_survive_j, WARD_death_i, WARD_death_j, ICU_survive_i, ICU_survive_j, ICU_death_i, ICU_death_j, WARD_discharged_ICU_survive_i, WARD_discharged_ICU_survive_j]
		compartiments_strings_list = ['Si', 'Sj', 'Ei', 'Ej', 'Ii', 'Ij', 'Ri', 'Rj', 'Hi', 'Hj', 'Ui', 'Uj', 'Mi', 'Mj', 'dHi', 'dHj', 'dUi', 'dUj', 'pHi', 'pHj', 'pUi', 'pUj', 'pMi', 'pMj', 'WARD_survive_i', 'WARD_survive_j', 'WARD_death_i', 'WARD_death_j', 'ICU_survive_i', 'ICU_survive_j', 'ICU_death_i', 'ICU_death_j', 'WARD_discharged_ICU_survive_i', 'WARD_discharged_ICU_survive_j']
		compartiment_medians = {}
		compartiment_lower_quantile = {}
		compartiment_upper_quantile = {}

		for res, compartment in enumerate(compartiments_list):
			compartiment_medians[res] = np.median(compartment, axis=0)
			compartiment_lower_quantile[res] = np.quantile(compartment, 0.05, axis=0)
			compartiment_upper_quantile[res] = np.quantile(compartment, 0.95, axis=0)

		median_results = pd.DataFrame(compartiment_medians)
		lower_quantile_results = pd.DataFrame(compartiment_lower_quantile)
		upper_quantile_results = pd.DataFrame(compartiment_upper_quantile)

		median_results.columns = compartiments_strings_list
		lower_quantile_results.columns = compartiments_strings_list
		upper_quantile_results.columns = compartiments_strings_list
		results_percentiles[i] = [median_results, lower_quantile_results, upper_quantile_results]

	return results_percentiles

	median_results.to_excel('Resultados_medianas.xlsx')
	lower_quantile_results.to_excel('Resultados_percentil_05.xlsx')
	upper_quantile_results.to_excel('Resultados_percentil_95.xlsx')

def peak_capacity(results, capacity, column):

    days = (results
        .query(f'{column} > @capacity')
        .assign(dummy=1)
        ['dummy']
        .count())

    return days 

def generate_report(results, model_parameters):
	
	results = (results
		.assign(hospitalizados=results['Hi'] + results['Hj'])
		.assign(UTI=results['Ui'] + results['Uj']))
	
	report = []
	CAPACITY_ICU = model_parameters.bed_icu #32304
	CAPACITY_WARD = model_parameters.bed_ward #298791
	
	# 1: without; 2: vertical; 3: horizontal isolation
	print (model_parameters.isolation_level)
	for i in range(len(model_parameters.isolation_level)): # 2: paper

		omega_i = model_parameters.isolation_level[i]
		omega_j = model_parameters.isolation_level[i]
		
		for availability in model_parameters.lotation:

			results_filtered_omega = results.query('omega_i == @omega_i & omega_j == @omega_j')

			capacity_ward = round(CAPACITY_WARD * availability)
			capacity_icu = round(CAPACITY_ICU * availability)

			metrics = {
				'capacidade_enfermaria': capacity_ward,
				'capacidade_icu': capacity_icu,
				'duracao_pico_enfemaria': peak_capacity(results_filtered_omega, capacity_ward, 'hospitalizados'),
				'duracao_pico_uti': peak_capacity(results_filtered_omega, capacity_icu, 'UTI'),
				'mortes_jovens': round(results_filtered_omega['Mj'].max()),
				'mortes_idosos': round(results_filtered_omega['Mi'].max()),
				'pico_necessidade_enfermaria_idosos': round(results_filtered_omega['Hi'].max()),
				'pico_necessidade_enfermaria_jovens': round(results_filtered_omega['Hj'].max()),
				'pico_necessidade_enfermaria': round(results_filtered_omega['hospitalizados'].max()),
				'pico_necessidade_uti_idosos': round(results_filtered_omega['Ui'].max()),
				'pico_necessidade_uti_jovens': round(results_filtered_omega['Uj'].max()),
				'pico_necessidade_uti': round(results_filtered_omega['UTI'].max()),
			}

			metrics['mortes'] = metrics['mortes_jovens'] + metrics['mortes_idosos']
			metrics['demanda_proporcional_enfermaria_pico'] = round((metrics['pico_necessidade_enfermaria'] / capacity_ward) * 100)
			metrics['demanda_proporcional_uti_pico'] = round((metrics['pico_necessidade_uti'] / capacity_icu) * 100)
			metrics['disponibilidade'] = round(availability * 100)
			metrics['omega_i'] = omega_i
			metrics['omega_j'] = omega_j

			report.append(metrics)
	
	return pd.DataFrame(report).rename(columns=
		{
			"capacidade_enfermaria": "Capacidade CNES Enfermaria",
			"capacidade_icu": "Capacidade CNES UTI",
			"duracao_pico_enfemaria": "Duração do pico acima da capacidade enfermaria (dias)",
			"duracao_pico_uti": "Duração do pico  acima da capacidade UTI (dias)",
			"mortes_jovens": "Mortes jovens  (total) -sem contar por falta de assistencia",
			"mortes_idosos":"Mortes Idosos  (total)-sem contar por falta de assistencia",
			"mortes":"Mortes Idosos + jovens  (total)-sem contar por falta de assistencia",
			"pico_necessidade_enfermaria_idosos": "Pico necessidade de leitos para Idosos com COVID enfermaria",
			"pico_necessidade_enfermaria_jovens": "Pico necessidade de leitos para Jovens com COVID enfermaria",
			"pico_necessidade_enfermaria" : "Pico necessidade de leitos para (idosos + Jovens) com COVID enfermaria",
			"pico_necessidade_uti_idosos": "Pico necessidade de leitos para Idosos com COVID UTI",
			"pico_necessidade_uti_jovens": "Pico necessidade de leitos para Jovens com COVID UTI",
			"pico_necessidade_uti": "Pico necessidade de leitos para (idosos + Jovens) com COVID UTI",
			"demanda_proporcional_enfermaria_pico": "% demada em função da capacidade instalada de enfermarias no pico",
			"demanda_proporcional_uti_pico": "% demada em função da capacidade instalada de UTIs no pico",
			"disponibilidade": "% de leitos do CNES disponiveis para atender pacientes com covid",
			"omega_i": "Omega utilizado para idosos",
			"omega_j": "Omega utilizado para jovens"
		}
	)
