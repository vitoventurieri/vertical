from collections import namedtuple


def get_input_data():
    """
    Provides the inputs for the simulation
    :return: tuples for the demograph_parameters, covid_parameters and model_parameters
    """

    demograph_parameters = namedtuple('Demograph_Parameters',
                                      ['population',                # N
                                       'population_rate_elderly',   # percentual_pop_idosa
                                       'bed_ward',                  # capacidade_leitos
                                       'bed_icu'                    # capacidade_UTIs
                                       ])

    demograph_parameters = demograph_parameters(
        # Brazilian Population
        population=210000000,             # 210 millions, 2020 forecast, Source: IBGE's app
        # Brazilian old people proportion (age: 55+)
        population_rate_elderly=0.2,      # 20%, 2020 forecast, Source: IBGE's app
        # Brazilian places
        bed_ward=295083,                  # regular bed, Source: CNES, 13/04/2020
        bed_icu=32329,                    # bed UTIs, Source: CNES, 13/04/2020
    )

    # Basic Reproduction Number # ErreZero
    basic_reproduction_number = 2.3     # 0.8 / 1.3 / 1.8 / 2.3 / 2.8
    # Infectivity Period (in days)      # tempo_de_infecciosidade
    infectivity_period = 10             # 5 / 7.5 / 10 / 12.5 / 15
    # Incubation Period (in days)
    incubation_period = 5               # 1 / 2.5 / 5 / 7.5 / 10 / 12.5 / 15

    # Variaveis de apoio
    incubation_rate = 1 / incubation_period
    infectiviy_rate = 1 / infectivity_period
    contamination_rate = basic_reproduction_number / infectivity_period

    covid_parameters = namedtuple('Covid_Parameters',
                                  ['alpha',                             # incubation rate
                                   'beta',                              # contamination rate
                                   'gamma',                             # infectivity rate
                                   'mortality_rate_elderly',            # taxa_mortalidade_i
                                   'mortality_rate_young',              # taxa_mortalidade_j
                                   'los_ward',                          # los_leito
                                   'los_icu',                           # los_uti
                                   'delay_ward',                        # los_leito
                                   'delay_icu',                         # los_uti
                                   'internation_rate_ward_elderly',     # tax_int_i
                                   'internation_rate_icu_elderly',      # tax_uti_i
                                   'internation_rate_ward_young',       # tax_int_j
                                   'internation_rate_icu_young'         # tax_uti_j
                                   ])

    covid_parameters = covid_parameters(
        # Incubation rate (1/day)
        alpha=incubation_rate,
        # Contamination rate (1/day)
        beta=contamination_rate,
        # Infectivity rate (1/day)
        gamma=infectiviy_rate,
        # Mortality Rates, Source: Verity, et al
        mortality_rate_elderly=0.03495,         # old ones: 60+ years
        mortality_rate_young=0.00127,           # young ones: 0-59 years
        # Length of Stay (in days)
        los_ward=8.9,                         # regular, Source: Wuhan
        los_icu=8,                            # UTI, Source: Wuhan
        # Delay (in days)
        delay_ward=2,                         #
        delay_icu=3,                          #
        # Internation Rate by type and age, Source for hospitalization verity et al; Proportion those need ICU: Severe Outcomes Among Patients with Coronavirus Disease 2019 CDC
        internation_rate_ward_elderly=0.1026,  # regular for old ones: 60+ years
        internation_rate_icu_elderly=0.0395,   # UTI for old ones: 60+ years
        internation_rate_ward_young=0.0209,    # regular for young ones: 0-59 years
        internation_rate_icu_young=0.0052       # UTI for young ones: 0-59 years
    )

    model_parameters = namedtuple('Model_Parameters',
                                  ['contact_reduction_elderly',     # omega_i
                                   'contact_reduction_young',       # omega_j
                                   'lotation',                      # lotacao
                                   'init_exposed_elderly',          # Ei0
                                   'init_exposed_young',            # Ej0
                                   'init_infected_elderly',         # Ii0
                                   'init_infected_young',           # Ij0
                                   'init_removed_elderly',          # Ri0
                                   'init_removed_young',            # Rj0
                                   't_max'                          # t_max
                                   ])

    model_parameters = model_parameters(
        # Social contact reduction factor
        contact_reduction_elderly=1.0,      # 0.2#0.4#0.6#0.8#1.0# # old ones: 55+ years
        contact_reduction_young=1.0,        # 0.2#0.4#0.6#0.8#1.0# # young ones: 0-54 years
        # Scenaries for health system colapse
        lotation=(0.3, 0.5, 0.8, 1),        # 30, 50, 80, 100% capacity
        init_exposed_elderly=20000,         # initial exposed population old ones: 55+ years
        init_exposed_young=20000,           # initial exposed population young ones: 0-54 years
        init_infected_elderly=5520,         # initial infected population old ones: 55+ years
        init_infected_young=10000,          # initial infected population young ones: 0-54 years
        init_removed_elderly=3000,          # initial removed population old ones: 55+ years
        init_removed_young=6240,            # initial removed population young ones: 0-54 years
        t_max=2*365 	                    # number of days to run
    )

    return demograph_parameters, covid_parameters, model_parameters
