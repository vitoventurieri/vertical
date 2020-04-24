import pandas as pd

CAPACITY_ICU = 32304
CAPACITY_WARD = 298791

def peak_capacity(results, capacity):

    days = (results
        .query('hospitalizados > @capacity')
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
        
        capacity_ward = CAPACITY_WARD * availability
        capacity_icu = CAPACITY_ICU * availability

        metrics = {
            'capacidade_enfermaria': capacity_ward,
            'capacidade_icu': capacity_icu,
            'duracao_pico_enfemaria': peak_capacity(results, capacity_ward),
            'duracao_pico_uti': peak_capacity(results, capacity_icu),
            'mortes_jovens': round(results['Mj'].sum()),
            'mortes_idosos': round(results['Mi'].sum()),
            'pico_necessidade_enfermaria_idosos': results['Hi'].max(),
            'pico_necessidade_enfermaria_jovens': results['Hj'].max(),
            'pico_necessidade_enfermaria': results['hospitalizados'].max(),
            'pico_necessidade_uti_idosos': results['Ui'].max(),
            'pico_necessidade_uti_jovens': results['Uj'].max(),
            'pico_necessidade_uti': results['UTI'].max(),
        }

        metrics['mortes'] = metrics['mortes_jovens'] + metrics['mortes_idosos']
        metrics['demanda_proporcional_enfermaria_pico'] = metrics['pico_necessidade_enfermaria'] / capacity_ward
        metrics['demanda_proporcional_uti_pico'] = metrics['pico_necessidade_uti'] / capacity_icu
        metrics['disponibilidade'] = availability

        report.append(metrics)

    return pd.DataFrame(report)