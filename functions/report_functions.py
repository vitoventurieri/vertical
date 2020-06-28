import pandas as pd



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
	for i in range(len(model_parameters.contact_reduction_elderly)): # 2: paper
		omega_i = model_parameters.contact_reduction_elderly[i]
		omega_j = model_parameters.contact_reduction_young[i]
		
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
