import pandas as pd

CAPACITY_ICU = 32304
CAPACITY_WARD = 298791

def peak_capacity(results, capacity, column):

    days = (results
        .query(f'{column} > @capacity')
        .assign(dummy=1)
        ['dummy']
        .count())

    return days 

def generate_report(results):

    results = (results
        .assign(hospitalizados=results['Hi'] + results['Hj'])
        .assign(UTI=results['Ui'] + results['Uj']))
    
    report = []

    for availability in [1., .7, .5 , .3, .2]:
        
        capacity_ward = round(CAPACITY_WARD * availability)
        capacity_icu = round(CAPACITY_ICU * availability)

        metrics = {
            'capacidade_enfermaria': capacity_ward,
            'capacidade_icu': capacity_icu,
            'duracao_pico_enfemaria': peak_capacity(results, capacity_ward, 'hospitalizados'),
            'duracao_pico_uti': peak_capacity(results, capacity_icu, 'UTI'),
            'mortes_jovens': round(results['Mj'].sum()),
            'mortes_idosos': round(results['Mi'].sum()),
            'pico_necessidade_enfermaria_idosos': round(results['Hi'].max()),
            'pico_necessidade_enfermaria_jovens': round(results['Hj'].max()),
            'pico_necessidade_enfermaria': round(results['hospitalizados'].max()),
            'pico_necessidade_uti_idosos': round(results['Ui'].max()),
            'pico_necessidade_uti_jovens': round(results['Uj'].max()),
            'pico_necessidade_uti': round(results['UTI'].max()),
        }

        metrics['mortes'] = metrics['mortes_jovens'] + metrics['mortes_idosos']
        metrics['demanda_proporcional_enfermaria_pico'] = round((metrics['pico_necessidade_enfermaria'] / capacity_ward) * 100)
        metrics['demanda_proporcional_uti_pico'] = round((metrics['pico_necessidade_uti'] / capacity_icu) * 100)
        metrics['disponibilidade'] = round(availability * 100)

        report.append(metrics)

    return pd.DataFrame(report).rename(columns=
        {"capacidade_enfermaria": "Capacidade CNES Enfermaria",
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
         "disponibilidade": "% de leitos do CNES disponiveis para atender pacientes com covid"
         })