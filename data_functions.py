from collections import namedtuple
import numpy as np
import numpy.random as npr
import pandas as pd
import sys


# from .utils import *


# from datetime import datetime as dt

class Conditions:

    def __init__(self,
                 I0,
                 E0,
                 R0,
                 M0,
                 population,
                 fit_analysis,
                 covid_parameters,
                 bed_ward=298_855,
                 bed_icu=32_380,
                 elderly_proportion=.1425) -> None:

        self.I0 = I0
        self.E0 = E0
        self.R0 = R0
        self.M0 = M0
        self.population = population
        self.bed_icu = bed_icu
        self.bed_ward = bed_ward
        self.elderly_proportion = elderly_proportion

        self.Ei0 = self.E0 * self.elderly_proportion
        self.Ii0 = self.I0 * self.elderly_proportion
        self.Ri0 = self.R0 * self.elderly_proportion

        self.Ej0 = self.E0 * (1 - self.elderly_proportion)
        self.Ij0 = self.I0 * (1 - self.elderly_proportion)
        self.Rj0 = self.R0 * (1 - self.elderly_proportion)

        # Leitos normais demandados
        self.Hi0 = self.Ii0 * covid_parameters.internation_rate_ward_elderly
        self.Hj0 = self.Ij0 * covid_parameters.internation_rate_ward_young
        # Leitos UTIs demandados
        self.Ui0 = self.Ii0 * covid_parameters.internation_rate_icu_elderly
        self.Uj0 = self.Ij0 * covid_parameters.internation_rate_icu_young
        # Excesso de demanda para leitos
        self.dHi0 = 0
        self.dHj0 = 0
        self.dUi0 = 0
        self.dUj0 = 0
        # Obitos
        # M_0 = 3_000
        if fit_analysis != 1:
            self.Mi0 = self.Ri0 * covid_parameters.mortality_rate_elderly
            self.Mj0 = self.Rj0 * covid_parameters.mortality_rate_young
        else:
            self.Mi0 = self.M0 * self.elderly_proportion
            self.Mj0 = self.Rj0 * (1 - self.elderly_proportion)
        # Perhaps initial conditions will change to match deaths at the present date


def make_lognormal_params_95_ci(lb, ub):
    """
    Provides mean and standard deviation of a lognormal distribution for 95% confidence interval

    :param lb: lower bound
    :param ub: lower bound and upper bound
    :return: mean
    :return: std: standard deviation
    """
    mean = (ub * lb) ** (1 / 2)
    std = (ub / lb) ** (1 / 4)

    # http://broadleaf.com.au/resource-material/lognormal-distribution-summary/
    # v1 = ub
    # v2 = lb
    # z1 = 1.96
    # z2 = 1.96
    # std = log( v1 / v2 ) / (z1 - z2)
    # mu = ( z2 * log(v1) - z1 * log(v2)) / (z2 - z1)

    return mean, std


def parameter_for_rt_fit_analisys(city, est_incubation_period, est_infectious_period, expected_mortality, expected_initial_rt):
    """

    :param incubation_period: 2 days
    :param expected_initial_rt: 2  # estimated basic reproduction number
    :param expected_mortality:  0.0065  # Verity mortality rate
    :param city:
    :return:
    """
    df_wcota = pd.read_csv('data\cases-brazil-cities-time.csv', sep=',')
    # dataset source : https://github.com/wcota/covid19br/blob/master/README.md
    # W. Cota, “Monitoring the number of COVID-19 cases and deaths in brazil at municipal and federative units level”,
    # SciELOPreprints:362 (2020), 10.1590/scielopreprints.362 - license (CC BY-SA 4.0) acess 30/07/2020
    #
    df_ibge = pd.read_csv(r'data\populacao_ibge.csv', sep=';', encoding="ISO-8859-1")
    # source #http://tabnet.datasus.gov.br/cgi/tabcgi.exe?ibge/cnv/poptbr.def

    codigo_da_cidade_ibge = city  # 355030

    def fix_city_name(row):
        row = row[7:]
        return row

    def fix_city_code(row):
        row = str(row)[:6]
        if row == 'Total':
            row = 000000
        row = int(row)
        return row

    # fix strings on datasets
    df_ibge['city_name_fixed'] = df_ibge['Município'].map(fix_city_name)
    df_ibge['city_code_fixed'] = df_ibge['Município'].map(fix_city_code)
    df_wcota['ibge_code_trimmed'] = df_wcota['ibgeID'].map(fix_city_code)

    # select datasets in the city with rows only with > x deaths
    df_cidade = df_wcota.loc[
        (df_wcota.ibge_code_trimmed == codigo_da_cidade_ibge) & (df_wcota.deaths >= 50)].reset_index()
    pop_cidade = df_ibge['População_estimada'].loc[df_ibge.city_code_fixed == codigo_da_cidade_ibge].values

    # starting_day = 20

    round_infectious_period = np.rint(est_infectious_period)

    I0_fit = (df_cidade.loc[round_infectious_period, 'deaths'] - df_cidade.loc[0, 'deaths'])*(est_infectious_period/round_infectious_period) / expected_mortality
    E0_fit = (I0_fit * expected_initial_rt * est_incubation_period )/ est_infectious_period
    R0_fit = df_cidade.loc[0, 'deaths'] / expected_mortality
    M0_fit = df_cidade.loc[0, 'deaths']
    population_fit = int(pop_cidade)

    return E0_fit, I0_fit, R0_fit, M0_fit, population_fit


# def fit_curve(state_name = 'Rio de Janeiro (UF)',
#              metodo = 'subreport',
#              sub_report = 10):
#    '''
#    provides comparison to reported data
#
#    Returns
#    -------
#    dfMS : dataframe
#        raw data from xlsx file provided dayly by the Brazillian
#        Health Ministerium at https://covid.saude.gov.br/
#    startdate : string
#        simulation initial date
#    state_name : string
#         name of the state of interest
#    population : int
#        population of the state of interest
#    sub_report: int
#        subnotification factor
#    E0 : int
#        Exposed at startdate, defined arbitrarily as 80% of the
#        Infected ones
#    I0 : int
#        Infected at startdate, defined as the difference between
#        the cumulative infected at startdate - infected at backdate (in which
#        backdate was considered arbitrarily as 13 days before startdate)
#    R0 : int
#        Removed at startdate, taken as the infected + deceased at backdate 
#    M0 : int
#        Deceased at startdate
#    r0 : float
#        basic reproduction number confidence interval
#
#    PROCEDIMENTO
# 1) Selecionar local de interesse
# 2) Buscar intervalo de dias em que r está relativamente constante
# obs: para Brasil, não foi visto evolução no tempo, simplesmente pegado IC do
# Imperial College
# 3) Define data em que r começa a ficar constante: startdate (extrai M0 como
# óbitos reportados nesta data)
# 4) Define período de infecciosidade: backdate = startdate - 13 dias (fit ARBITRÁRIO)
# obs: modelo considera 10 dias
# 5) Contabiliza número acumulado de Infectados no período entre backdate e
#                                                             startdate: Infect
# 6) Define um fator de subnotificação: sub_report
# obs: ARBITRÁRIO (15 para PE, SP, Brasil; 3 para SC)
# 7) Multiplica Infect pelo fator de subnotificação: Infect = Infect * sub_report
# 8) Define I0 (infectados em startdate) como diferença do número acumulado
# de infectados entre startdate e backdate, corrigida com subnotificação
# 9) Define R0 (removidos em startdate) como número acumulado de infectados +
#                                                             óbitos em backdate
# 10) Define E0 (expostos em startdate) como 80% de I0
# obs: 80% ARBITRÁRIO
#
#    '''
#
#
#    # INPUT
#    # state_name = 'Rio de Janeiro (UF)' # 'Pernambuco' #'Brasil' # 'Santa Catarina' #
#    # metodo = "subreport" # "fator_verity" #
#    # sub_report = 10
#
#    print('Para: ' + state_name + ' com o metodo ' + metodo)
#
#    # IMPORT DATA
#    print('Importando arquivo do Ministerio da Saude')
#    print('Este processo demora ...')
#
#    #url = 'https://github.com/viniciusriosfuck/vertical/blob/master/data/HIST_PAINEL_COVIDBR_24jun2020.xlsx?raw=true'
#    #df = pd.read_excel(url)
#
#    data_folder = os.path.join(get_root_dir(), 'data')
#    filename = 'HIST_PAINEL_COVIDBR_24jun2020.xlsx'
#    df = pd.read_excel(f'data/{filename}')
#
#    print('Importado')
#
# # Brazilian states data
# states_json = [
#     {"coduf": 76, "state_name": "Brasil", "populationTCU2019": 210_147_125},
#     {"coduf": 11, "state_name": "Rondônia", "populationTCU2019": 1_777_225},
#     {"coduf": 12, "state_name": "Acre", "populationTCU2019": 881_935},
#     {"coduf": 13, "state_name": "Amazonas", "populationTCU2019": 4_144_597},
#     {"coduf": 14, "state_name": "Roraima", "populationTCU2019": 605_761},
#     {"coduf": 15, "state_name": "Pará", "populationTCU2019": 8_602_865},
#     {"coduf": 16, "state_name": "Amapá", "populationTCU2019": 845_731},
#     {"coduf": 17, "state_name": "Tocantins", "populationTCU2019": 1_572_866},
#     {"coduf": 21, "state_name": "Maranhão", "populationTCU2019": 7_075_181},
#     {"coduf": 22, "state_name": "Piauí", "populationTCU2019": 3_273_227},
#     {"coduf": 23, "state_name": "Ceará", "populationTCU2019": 9_132_078},
#     {"coduf": 24, "state_name": "Rio Grande do Norte", "populationTCU2019": 3_506_853},
#     {"coduf": 25, "state_name": "Paraíba", "populationTCU2019": 4_018_127},
#     {"coduf": 26, "state_name": "Pernambuco", "populationTCU2019": 9_557_071},
#     {"coduf": 27, "state_name": "Alagoas", "populationTCU2019": 3_337_357},
#     {"coduf": 28, "state_name": "Sergipe", "populationTCU2019": 2_298_696},
#     {"coduf": 29, "state_name": "Bahia", "populationTCU2019": 14_873_064},
#     {"coduf": 31, "state_name": "Minas Gerais", "populationTCU2019": 21_168_791},
#     {"coduf": 32, "state_name": "Espiríto Santo", "populationTCU2019": 4_018_650},
#     {"coduf": 33, "state_name": "Rio de Janeiro (UF)", "populationTCU2019": 17_264_943},
#     {"coduf": 35, "state_name": "São Paulo (UF)", "populationTCU2019": 45_919_049},
#     {"coduf": 41, "state_name": "Paraná", "populationTCU2019": 11_433_957},
#     {"coduf": 42, "state_name": "Santa Catarina", "populationTCU2019": 7_164_788},
#     {"coduf": 43, "state_name": "Rio Grande do Sul", "populationTCU2019": 11_377_239},
#     {"coduf": 50, "state_name": "Mato Grosso do Sul", "populationTCU2019": 2_778_986},
#     {"coduf": 51, "state_name": "Mato Grosso", "populationTCU2019": 3_484_466},
#     {"coduf": 52, "state_name": "Goiás", "populationTCU2019": 7_018_354},
#     {"coduf": 53, "state_name": "Distrito Federal", "populationTCU2019": 3_015_268}
# ]
# states_df = pd.DataFrame(states_json,
#                          columns=['coduf', 'state_name', 'populationTCU2019'])


#    elif state_name == 'Brasil':
#        startdate = '2020-04-26'
#        r0 = (2.25, 3.57) # fonte r0: Brasil: Imperial College
#        # coduf = 76
#        # population = 210_147_125
#        sub_report = 8        
# https://www1.folha.uol.com.br/equilibrioesaude/2020/04/brasil-tem-maior-taxa-de-contagio-por-coronavirus-do-mundo-aponta-estudo.shtml
#    elif state_name == 'Rio de Janeiro (UF)':
#        startdate = '2020-03-23' #
#        r0 = (1.28, 1.60) # (1.15, 1.32) # (3.4, 5.3) # Imperial College 21
#
#    
#    states_set = states[states['state_name'] == state_name ]
#    
#    population = states_set['populationTCU2019'].values[0]
#    coduf = states_set['coduf'].values[0]                 
#    
#    dfMS = df[df['coduf'] == coduf ]
#    dfMS = dfMS[dfMS['codmun'].isnull()] # only rows without city
#    
#    dfMS['data'] = pd.to_datetime(dfMS['data'])
#    dfMS['obitosAcumulado'] = pd.to_numeric(dfMS['obitosAcumulado'])
#    
#    M0_MS = dfMS['obitosAcumulado'].max() # most recent cumulative deaths
#    R0_MS = dfMS['Recuperadosnovos'].max() + M0_MS  # most recent removed
#    I0_MS = dfMS['emAcompanhamentoNovos'].max() # most recent active reported cases
#    
#    # IDENTIFY 13 DAYS AGO
#    # Hypothesis: one takes 13 days to recover
#    backdate = pd.to_datetime(startdate) - pd.DateOffset(days=3)
#    #backdate = pd.to_datetime(backdate.date())#pd.to_datetime(backdate).dt.date#.normalize()
#    # DECEASED
#    M0 = dfMS['obitosAcumulado'][dfMS['data'] == startdate].values[0]
#    # CUMULATIVE INFECTED FROM THE PREVIOUS 13 DAYS
#    Infect = dfMS['casosAcumulado'][dfMS['data'].
#                                 between(backdate,startdate,inclusive = True)]
#    #print(backdate)
#    #print(Infect)
#    #print(dfMS['data'])    
#    # ESTIMATED INITIAL CONDITIONS
#    if metodo == "subreport":
#        Infect = Infect * sub_report
#        # INFECTED
#        I0 = max(Infect) - min(Infect)
#        # RECOVERED
#        R0 = min(Infect) + dfMS['obitosAcumulado'][dfMS['data'] == backdate].values[0]# max(Infect) - I0
#    elif metodo == "fator_verity":
#        I0 = M0 * 165 # estimated from Verity to Brazil: country, state, city
#        R0 = I0 * 0.6 # Hypothesis: Removed correspond to 60% of the Infected
#    # EXPOSED
#    E0 = 0.8 * I0  # Hypothesis: Exposed correspond to 80% of the Infected   
#    
#    
#    #E0, I0, R0 = 5, 10, 0
#    
#    return dfMS, startdate, state_name, population, sub_report, E0, I0, R0, M0, r0


def get_input_data(IC_analysis, city):
    """
    Provides the inputs for the simulation
    :return: tuples for the demograph_parameters, covid_parameters 
    and model_parameters
    
    
    
    Degrees of isolation (i)
    no isolation, vertical, horizontal
        
    IC_Analysis
    1: Confidence Interval; 2: Single Run; 3: Sensitivity Analysis
    
    1: CONFIDENCE INTERVAL for a lognormal distribution
    2: SINGLE RUN
    3: r0 Sensitivity analysis: Calculate an array for r0 
    to perform a sensitivity analysis with 0.1 intervals
    
    """

    IC_analysis = IC_analysis  # 1 # 2 # 3 
    if IC_analysis == 1:
        print('Confidence Interval Analysis (r0, gamma and alpha, lognormal distribution)')
    elif IC_analysis == 2:
        print('Single Run Analysis')
    elif IC_analysis == 3:
        print('Sensitivity r0 Analysis')
    elif IC_analysis == 4:
        print('Time variable inputted Rt analysis with confidence interval')
    else:
        sys.exit('ERROR: Not programmed such Analysis, please enter 1, 2, 3 or 4')

    runs = 10  # 1_000 # number of runs for Confidence Interval analysis

    dfMS, startdate, state_name, sub_report, r0_fit = [], [], [], [], []

    fit_analysis = 0  # 0 # 1 #

    if fit_analysis == 1:
        if IC_analysis == 1:
            pass
            #  print('With fit analysis')
            # [dfMS, startdate, state_name, population_fit, sub_report,
            #  E0_fit, I0_fit, R0_fit, M0_fit, r0_fit] = fit_curve()
        else:
            sys.exit('ERROR: Not programmed fit analysis for other case than Confidence Interval')

            # CONFIDENCE INTERVAL AND SENSITIVITY ANALYSIS
    # 95% Confidence interval bounds or range for sensitivity analysis
    # Basic Reproduction Number # ErreZero
    basic_reproduction_number = (2.7, 2.9)  # (2.4, 3.3)     # 1.4 / 2.2 / 3.9
    if fit_analysis == 1:
        basic_reproduction_number = r0_fit

    # SINGLE RUN AND SENSITIVITY ANALYSIS
    # Incubation Period (in days)
    incubation_period = 2  # (4.1, 7.0) #
    # Infectivity Period (in days)      # tempo_de_infecciosidade
    infectivity_period = 3
    infection_to_death_period = 17
    pI = 0.1425  # 962/7600 #  Proportion of persons aged 60+ in Brazil,
    # 2020 forecast, Source: IBGE's app
    contact_matrix = [None] * 3
    contact_matrix[0] = np.array([[16.24264923, 0.54534103], [3.2788393, 0.72978211]])
    contact_matrix[1] = np.array([[16.24264923, 0.2391334], [1.43777922, 0.38361959]])
    contact_matrix[2] = np.array([[7.47115329, 0.2391334], [1.43777922, 0.38361959]])

    Population_proportion = np.array([1 - pI, pI])
    Normalization_constant = np.zeros(3)
    for k in range(3):
        M_matrix = np.zeros((len(contact_matrix[k]), len(contact_matrix[k][0])))

        for i in range(len(contact_matrix[k])):
            for j in range(len(contact_matrix[k][0])):
                M_matrix[i][j] = contact_matrix[k][i][j] * Population_proportion[i] / Population_proportion[j]
        Temp, _ = np.linalg.eig(M_matrix)
        Normalization_constant[k] = max(Temp.real)
    if IC_analysis == 1:  # CONFIDENCE INTERVAL for a lognormal distribution

        # PARAMETERS ARE ARRAYS

        # 95% Confidence interval bounds for Covid parameters
        # Incubation Period (in days)
        incubation_period = (3.3, 3.3)  # (2.1, 7) # (4.1, 7.0)

        # Infectivity Period (in days)   # tempo_de_infecciosidade
        # infectivity_period = (7.0, 12.0) #    3 days or 7 days
        infectivity_period = (3.5, 3.5)  # 3 days or 7 days
        # Woelfel et al 22 (eCDC: 7-12 days @ 19/4/20, 
        # https://www.ecdc.europa.eu/en/covid-19/questions-answers)
        infection_to_death_period = (16.9, 17.1)

        # Computes mean and std for a lognormal distribution
        alpha_inv_params = make_lognormal_params_95_ci(*incubation_period)
        gamma_inv_params = make_lognormal_params_95_ci(*infectivity_period)
        delta_inv_params = make_lognormal_params_95_ci(*infection_to_death_period)
        R0__params = make_lognormal_params_95_ci(*basic_reproduction_number)

        # samples for a lognormal distribution (Monte Carlo Method)
        # alpha
        incubation_rate = 1 / npr.lognormal(*map(np.log, alpha_inv_params), runs)
        # gamma
        infectivity_rate = 1 / npr.lognormal(*map(np.log, gamma_inv_params), runs)
        # beta = r0 * gamma
        contamination_rate = npr.lognormal(*map(np.log, R0__params), runs) * infectivity_rate
        infection_to_death_rate = 1 / npr.lognormal(*map(np.log, delta_inv_params), runs)

    elif IC_analysis == 2:  # SINGLE RUN

        # PARAMETERS ARE FLOATS
        basic_reproduction_number = 2.2
        # 2.2 is from Li Q, Guan X, Wu P et al. 
        # Early Transmission Dynamics in Wuhan, China, 
        # of Novel Coronavirus–Infected Pneumonia.
        # New England Journal of Medicine. 2020 Mar 26;382(13):1199–207.
        # DOI: 10.1056/NEJMoa2001316.
        # alpha
        incubation_rate = 1 / incubation_period
        # gamma
        infectivity_rate = 1 / infectivity_period
        # delta
        infection_to_death_rate = 1 / infection_to_death_period
        # beta = r0 * gamma
        contamination_rate = basic_reproduction_number * infectivity_rate
    elif IC_analysis == 3:
        # PARAMETERS ARE ARRAYS
        # Calculate array for r0 to a sensitivity analysis
        R0_array = np.arange(*basic_reproduction_number, 0.1)  # step 0.1 for r0
        # alpha with length
        incubation_rate = np.repeat(1 / incubation_period, len(R0_array))
        # gamma
        infectivity_rate = np.repeat(1 / infectivity_period, len(R0_array))
        # delta
        infection_to_death_rate = np.repeat(1 / infection_to_death_period, len(R0_array))
        # beta = r0 * gamma
        contamination_rate = R0_array * infectivity_rate

    else:  # r0 Sensitivity analysis
        # PARAMETERS ARE ARRAYS

        # 95% Confidence interval bounds for Covid parameters
        # Incubation Period (in days)
        incubation_period = (4.1 - 3.0, 7.1 - 0.8)  # (4.1, 7.0)

        # Infectivity Period (in days)   # tempo_de_infecciosidade
        # infectivity_period = (7.0, 12.0) #    3 days or 7 days
        infectivity_period = (2.92, 3.22)  # 3 days or 7 days
        # Woelfel et al 22 (eCDC: 7-12 days @ 19/4/20, 
        # https://www.ecdc.europa.eu/en/covid-19/questions-answers)
        infection_to_death_period = (16.9, 17.1)

        # Computes mean and std for a lognormal distribution
        alpha_inv_params = make_lognormal_params_95_ci(*incubation_period)
        gamma_inv_params = make_lognormal_params_95_ci(*infectivity_period)
        delta_inv_params = make_lognormal_params_95_ci(*infection_to_death_period)
        R0__params = make_lognormal_params_95_ci(*basic_reproduction_number)

        # samples for a lognormal distribution (Monte Carlo Method)
        # alpha
        incubation_rate = 1 / npr.lognormal(*map(np.log, alpha_inv_params), runs)
        # gamma
        infectivity_rate = 1 / npr.lognormal(*map(np.log, gamma_inv_params), runs)
        # beta = r0 * gamma
        contamination_rate = npr.lognormal(*map(np.log, R0__params), runs) * infectivity_rate
        infection_to_death_rate = 1 / npr.lognormal(*map(np.log, delta_inv_params), runs)

    covid_parameters = namedtuple('Covid_Parameters',
                                  ['alpha',  # incubation rate
                                   'beta',  # contamination rate
                                   'gamma',  # infectivity rate
                                   'delta',  # infection to death rate
                                   'mortality_rate_elderly',  # taxa_mortalidade_i
                                   'mortality_rate_young',  # taxa_mortalidade_j
                                   'los_ward',  # los_leito
                                   'los_icu',  # los_uti
                                   'infection_to_hospitalization',  # infection to hospitalization period
                                   'infection_to_icu',  # infection to icu period
                                   'internation_rate_ward_elderly',  # tax_int_i
                                   'internation_rate_ward_young',  # tax_int_j
                                   'internation_rate_icu_elderly',  # tax_uti_i
                                   'internation_rate_icu_young',  # tax_uti_j
                                   'pH',
                                   'pU'
                                   ])

    covid_parameters = covid_parameters(
        # Incubation rate (1/day)
        alpha=incubation_rate,
        # Contamination rate (1/day)
        beta=contamination_rate,
        # Infectivity rate (1/day)
        gamma=infectivity_rate,
        delta=infection_to_death_rate,
        # Mortality Rates, Source: Verity, et al,
        # adjusted with population distribution IBGE 2020
        mortality_rate_elderly=0.03495,  # old ones: 60+ years
        mortality_rate_young=0.00127,  # young ones: 0-59 years
        pH=0.6,  # probability of death for someone that needs a ward bed and does not receive it
        pU=0.9,  # probability of death for someone that needs an ICU bed and does not receive it
        # Length of Stay (in days), Source: Wuhan
        los_ward=8.9,  # regular
        los_icu=8,  # UTI
        infection_to_hospitalization=10,
        infection_to_icu=10,
        # Internation Rate by type and age, 
        # Source for hospitalization verity et al;
        # Proportion those need ICU:
        # Severe Outcomes Among Patients with Coronavirus Disease 2019 CDC
        internation_rate_ward_elderly=0.1026,  # regular for old ones: 60+ years
        internation_rate_ward_young=0.0209,  # regular for young ones: 0-59 years
        internation_rate_icu_elderly=0.0395,  # UTI for old ones: 60+ years
        internation_rate_icu_young=0.0052  # UTI for young ones: 0-59 years
    )

    model_parameters = namedtuple('Model_Parameters',
                                  ['isolation_level',  # niveis_isolamento
                                   #  'contact_reduction_elderly',  # omega_i
                                   #  'contact_reduction_young',  # omega_j
                                   'lotation',  # lotacao
                                   'init_exposed_elderly',  # Ei0
                                   'init_exposed_young',  # Ej0
                                   'init_infected_elderly',  # Ii0
                                   'init_infected_young',  # Ij0
                                   'init_removed_elderly',  # Ri0
                                   'init_removed_young',  # Rj0
                                   'init_hospitalized_ward_elderly',  # Hi0
                                   'init_hospitalized_ward_young',  # Hj0
                                   'init_hospitalized_ward_elderly_excess',  # dHi0
                                   'init_hospitalized_ward_young_excess',  # dHj0
                                   'init_hospitalized_icu_elderly',  # Ui0
                                   'init_hospitalized_icu_young',  # Uj0
                                   'init_hospitalized_icu_elderly_excess',  # dUi0
                                   'init_hospitalized_icu_young_excess',  # dUj0
                                   'init_deceased_elderly',  # Mi0
                                   'init_deceased_young',  # Mj0
                                   't_max',  # t_max
                                   'population',  # N
                                   'population_rate_elderly',  # percentual_pop_idosa
                                   'bed_ward',  # capacidade_leitos
                                   'bed_icu',  # capacidade_UTIs
                                   'IC_analysis',  # Type of analysis
                                   'dfMS',  # dataframe_Min_Saude_data
                                   'startdate',  # start date of the fit and simulation
                                   'state_name',  # state simulated
                                   'r0_fit',  # range of r0
                                   'sub_report',  # sub_report factor
                                   'contact_matrix',  # contact matrix
                                   'Normalization_constant',  # normalization constant for contact matrix
                                   'city'
                                   ])

    # N = 12_000_000  # 211_755_692 # 211 millions, 2020 forecast, Source: IBGE's app
    # 7_600_000_000, #

    # INITIAL CONDITIONS
    # E0 = 2300  # 64 #260_000 #basic_reproduction_number * I0
    # I0 = 1800  # 100#304_000 #  (a total of 20943 cases in the last 10 days
    # within a total of 38654 cumulative confirmed cases in 
    # 19/04/2020 17:00 GMT-3 - source https://covid.saude.gov.br/)
    # R0 = 3000  # 407#472_000 #

    expected_mortality = covid_parameters.mortality_rate_elderly * pI + (1-pI) * covid_parameters.mortality_rate_young
    expected_initial_rt = np.mean(basic_reproduction_number)  # botar pra fora??
    est_infectious_period = np.mean(infectivity_period)
    est_incubation_period = np.mean(incubation_period)


    # Caso queira usar parametros personalizados, escreva cidade em parameter_for_rt_fit_analisys('sao_paulo')
    E0, I0, R0, M0, N = parameter_for_rt_fit_analisys(city, est_incubation_period, est_infectious_period, expected_mortality, expected_initial_rt)

    ### Criando objeto com status iniciais, juntando todas as infos que mudam o inicio
    ### Os parametros padrao podem ser mudados, como cama/UTI por cidade
    conditions = Conditions(I0, E0, R0, M0, N, fit_analysis, covid_parameters)

    # Mantive aqui para questões de não estragar o restante, mas o normal seria usar o objeto mesmo
    # para o resto dos problemas.
    model_parameters = model_parameters(
        # Social contact reduction factor (without, vertical, horizontal) isolation
        isolation_level=[" Without isolation", " Elderly isolation"],
        # contact_reduction_elderly=(1., .4),  # young ones: 0-59 years
        # contact_reduction_young=(1., 1.),  # old ones: 60+ years
        # Scenaries for health system colapse
        lotation=(0.3, 0.5, 0.8, 1),  # 30, 50, 80, 100% capacity
        init_exposed_elderly=conditions.Ei0,  # initial exposed population old ones: 60+ years
        init_exposed_young=conditions.Ej0,  # initial exposed population young ones: 0-59 years
        init_infected_elderly=conditions.Ii0,  # initial infected population old ones: 60+ years
        init_infected_young=conditions.Ij0,  # initial infected population young ones: 0-59 years
        init_removed_elderly=conditions.Ri0,  # initial removed population old ones: 60+ years
        init_removed_young=conditions.Rj0,  # initial removed population young ones: 0-59 years
        init_hospitalized_ward_elderly=conditions.Hi0,  # initial ward hospitalized old ones: 60+ years
        init_hospitalized_ward_young=conditions.Hj0,  # initial ward hospitalized young ones: 0-59 years
        init_hospitalized_ward_elderly_excess=conditions.dHi0,
        # initial ward hospitalized demand excess old ones: 60+ years
        init_hospitalized_ward_young_excess=conditions.dHj0,
        # initial ward hospitalized demand excess young ones: 0-59 years
        init_hospitalized_icu_elderly=conditions.Ui0,  # initial icu hospitalized old ones: 60+ years
        init_hospitalized_icu_young=conditions.Uj0,  # initial icu hospitalized young ones: 0-59 years
        init_hospitalized_icu_elderly_excess=conditions.dUi0,
        # initial iCU hospitalized demand excess old ones: 60+ years
        init_hospitalized_icu_young_excess=conditions.dUj0,
        # initial iCU hospitalized demand excess young ones: 0-59 years
        init_deceased_elderly=conditions.Mi0,  # initial deceased population old ones: 60+ years
        init_deceased_young=conditions.Mj0,  # initial deceased population young ones: 0-59 years
        t_max=180,  # # number of days to run
        # Brazilian Population
        population=N,
        # Brazilian old people proportion (age: 60+), 2020 forecast
        population_rate_elderly=pI,
        # Proportion of persons aged 60+ in Brazil, Source: IBGE's app
        # Brazilian bed places , Source: CNES, 05/05/2020
        # http://cnes2.datasus.gov.br/Mod_Ind_Tipo_Leito.asp?VEstado=00
        bed_ward=conditions.bed_ward,  # bed ward
        bed_icu=conditions.bed_icu,  # bed ICUs
        IC_analysis=IC_analysis,  # flag for run type
        # 1: confidence interval, 2: single run, 3: r0 sensitivity analysis
        dfMS=dfMS,  # dataframe_Min_Saude_data
        startdate=startdate,  # start date of the fit and simulation
        state_name=state_name,  # state simulated
        r0_fit=r0_fit,  # range of r0 fitted
        sub_report=sub_report,  # sub_report factor
        contact_matrix=contact_matrix,
        Normalization_constant=Normalization_constant,
        city=city
    )

    parametros = {'incubation_period = 1/alpha': [incubation_period],
                  'basic_reproduction_number = beta/gamma': [basic_reproduction_number],
                  'infectivity_period = 1/gamma': [infectivity_period],
                  'runs': [runs],
                  'isolation_level': [model_parameters.isolation_level],
                  #  'contact_reduction_elderly': [model_parameters.contact_reduction_elderly],
                  #  'contact_reduction_young': [model_parameters.contact_reduction_young],
                  'lotation': [model_parameters.lotation],
                  'init_exposed_elderly': [conditions.Ei0],
                  'init_exposed_young': [conditions.Ej0],
                  'init_infected_elderly': [conditions.Ii0],
                  'init_infected_young': [conditions.Ij0],
                  'init_removed_elderly': [conditions.Ri0],
                  'init_removed_young': [conditions.Rj0],
                  'init_hospitalized_ward_elderly': [conditions.Hi0],
                  'init_hospitalized_ward_young': [conditions.Hj0],
                  'init_hospitalized_icu_elderly': [conditions.Ui0],
                  'init_hospitalized_icu_young': [conditions.Uj0],
                  'init_deceased_elderly': [conditions.Mi0],
                  'init_deceased_young': [conditions.Mj0],
                  't_max': [model_parameters.t_max],
                  'population': [N],
                  'population_rate_elderly': [pI],
                  'bed_ward': [model_parameters.bed_ward],
                  'bed_icu': [model_parameters.bed_icu],
                  'IC_analysis': [IC_analysis],
                  'fit_analysis': [fit_analysis],
                  'startdate': [startdate],
                  'state_name': [state_name],
                  'r0_fit': [r0_fit],
                  'sub_report': [sub_report],
                  'city': [model_parameters.city]}

    output_parameters = pd.DataFrame(parametros).T
    print(output_parameters)
    print('')

    return covid_parameters, model_parameters, output_parameters
