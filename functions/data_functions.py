import numpy as np
import numpy.random as npr
import pandas as pd
import sys
import os
from .utils import get_root_dir

np.random.seed(seed=1)


def city_name_to_code(city_name):
    city = { 'Porto Velho/RO': 110020,
        'Manaus/AM': 130260,
        'Rio Branco/AC': 120040,
        'Campo Grande/MS ': 500270,
        'Macapá/AP': 160030,
        'Brasília/DF': 530010,
        'Boa Vista/RR': 140010,
        'Cuiabá/MT': 510340,
        'Palmas/TO': 172100,
        'São Paulo/SP': 355030,
        'Teresina/PI': 221100,
        'Rio de Janeiro/RJ': 330455,
        'Belém/PA': 150140,
        'Goiânia/GO': 520870,
        'Salvador/BA': 292740,
        'Florianópolis/SC': 420540,
        'São Luís/MA': 211130,
        'Maceió/AL': 270430,
        'Porto Alegre/RS ': 431490,
        'Curitiba/PR': 410690,
        'Belo Horizonte/MG': 310620,
        'Fortaleza/CE': 230440,
        'Recife/PE': 261160,
        'João Pessoa/PB': 250750,
        'Aracaju/SE': 280030,
        'Natal/RN': 240810,
        'Vitória/ES': 320530}
    city_code = city[city_name]
    
    return city_code


class Conditions:

    def __init__(self,
                 E0,
                 I0,
                 R0,
                 M0,
                 population,
                 fit_analysis,
                 IC_analysis,
                 covid_parameters,
                 elderly_proportion) -> None:

        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.M0 = M0
        self.population = population
        self.elderly_proportion = elderly_proportion

        self.Ei0 = self.E0 * self.elderly_proportion
        self.Ii0 = self.I0 * self.elderly_proportion
        self.Ri0 = self.R0 * self.elderly_proportion

        self.Ej0 = self.E0 * (1 - self.elderly_proportion)
        self.Ij0 = self.I0 * (1 - self.elderly_proportion)
        self.Rj0 = self.R0 * (1 - self.elderly_proportion)

        if IC_analysis == 2:  # TODO check  # 'Single Run'
            self.Hi0 = 0#self.Ii0 * covid_parameters.internation_rate_ward_elderly
            self.Hj0 = 0#self.Ij0 * covid_parameters.internation_rate_ward_young
            # Leitos UTIs demandados
            self.Ui0 = 0#self.Ii0 * covid_parameters.internation_rate_icu_elderly
            self.Uj0 = 0#self.Ij0 * covid_parameters.internation_rate_icu_young
        else:
            # Leitos normais demandados
            self.Hi0 = 0#self.Ii0 * covid_parameters.internation_rate_ward_elderly.mean()
            self.Hj0 = 0#self.Ij0 * covid_parameters.internation_rate_ward_young.mean()
            # Leitos UTIs demandados
            self.Ui0 = 0#self.Ii0 * covid_parameters.internation_rate_icu_elderly.mean()
            self.Uj0 = 0#self.Ij0 * covid_parameters.internation_rate_icu_young.mean()

        # Excesso de demanda para leitos
        self.WARD_excess_i0 = 0
        self.WARD_excess_j0 = 0
        self.ICU_excess_i0 = 0
        self.ICU_excess_j0 = 0
        # Obitos
        if not fit_analysis: #0: #  TODO: Check if is there a not or no/ check if dis IF clause is necessary
            self.Mi0 = self.Ri0 * np.mean(covid_parameters.mortality_rate_elderly)
            self.Mj0 = self.Rj0 * np.mean(covid_parameters.mortality_rate_young)
        else:
            self.Mi0 = self.Ri0 * self.elderly_proportion * np.mean(covid_parameters.mortality_rate_elderly)
            self.Mj0 = self.Rj0 * (1 - self.elderly_proportion)* np.mean(covid_parameters.mortality_rate_young)
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
    # z2 = -1.96
    # std = log( v1 / v2 ) / (z1 - z2)
    # mu = ( z2 * log(v1) - z1 * log(v2)) / (z2 - z1)

    return mean, std

def fix_city_name(row):
    row = row[7:]
    return row

def fix_city_code(row):
    row = str(row)[:6]
    if row == 'Total':
        row = 000000
    row = int(row)
    return row


def import_MS_cases():
    """ Import data from MS
    # dataset source : https://covid.saude.gov.br/  ## https://covid.saude.gov.br/b056bbc5-07bf-4209-afc1-efb36354f158


    Returns:
        df_MS_CASES: dataframe
    """
    df_MS_CASES = pd.read_csv(os.path.join(get_root_dir(),
                    'data', 'HIST_PAINEL_COVIDBR_04dez2020.csv'), sep=';', encoding = "ISO-8859-1")
    # fix strings on datasets
    df_MS_CASES['ibge_code_trimmed'] = df_MS_CASES['codmun']#.map(fix_city_code)
    return df_MS_CASES

def import_ibge():
    """ Import data from IBGE

    # source #http://tabnet.datasus.gov.br/cgi/tabcgi.exe?ibge/cnv/poptbr.def

    Returns:
        df_ibge: dataframe
    """
    df_ibge = pd.read_csv(os.path.join(get_root_dir(),
            'data', 'populacao_ibge.csv'), sep=';', encoding="ISO-8859-1")
    
    # fix strings on datasets
    df_ibge['city_name_fixed'] = df_ibge['Município'].map(fix_city_name)
    df_ibge['city_code_fixed'] = df_ibge['Município'].map(fix_city_code)
    return df_ibge


def get_rt_by_city(city_name):
    """
    Return a dataframe with r (basic reproduction number) over time t for a city

    :param city: string
    :return: dataframe with the selected city
    """
    cities = {'Fortaleza/CE': 'Fortaleza',
              'São Paulo/SP': 'SaoPaulo',
              'Maceió/AL': 'Maceio',
              'São Luís/MA': 'SaoLuis'}

    return pd.read_csv(os.path.join(get_root_dir(),
                    'data',f"Re_{cities[city_name]}.csv"), sep=',')


def import_cnes(city_code):
    df_cnes = pd.read_csv(os.path.join(get_root_dir(), 'data',
                            'cnes_simplificado_02-2020.csv'), sep=';')  # source
    # leitos = {"BUCO MAXILO FACIAL": 1,
    #          "CARDIOLOGIA": 2,
    #          "CIRURGIA GERAL": 3,
    #          "ENDOCRINOLOGIA": 4,
    #          "GASTROENTEROLOGIA": 5,
    #          "GINECOLOGIA": 6,
    #          "CIRURGICO/DIAGNOSTICO/TERAPEUTICO": 7,
    #          "NEFROLOGIAUROLOGIA": 8,
    #          "NEUROCIRURGIA": 9,
    #          "OBSTETRICIA CIRURGICA": 10,
    #          "OFTALMOLOGIA": 11,
    #          "ONCOLOGIA": 12,
    #          "ORTOPEDIATRAUMATOLOGIA": 13,
    #          "OTORRINOLARINGOLOGIA": 14,
    #          "PLASTICA": 15,
    #          "TORACICA": 16,
    #          "AIDS": 31,
    #          "CARDIOLOGIA": 32,
    #          "CLINICA GERAL": 33,
    #          "CRONICOS": 34,
    #          "DERMATOLOGIA": 35,
    #          "GERIATRIA": 36,
    #          "HANSENOLOGIA": 37,
    #          "HEMATOLOGIA": 38,
    #          "NEFROUROLOGIA": 40,
    #          "NEONATOLOGIA": 41,
    #          "NEUROLOGIA": 42,
    #          "OBSTETRICIA CLINICA": 43,
    #          "ONCOLOGIA": 44,
    #          "PEDIATRIA CLINICA": 45,
    #          "PNEUMOLOGIA": 46,
    #          "PSIQUIATRIA": 47,
    #          "REABILITACAO": 48,
    #          "PNEUMOLOGIA SANITARIA": 49,
    #          "UNIDADE INTERMEDIARIA": 64,
    #          "UNIDADE INTERMEDIARIA NEONATAL": 65,
    #          "UNIDADE ISOLAMENTO": 66,
    #          "TRANSPLANTE": 67,
    #          "PEDIATRIA CIRURGICA": 68,
    #          "AIDS": 69,
    #          "FIBROSE CISTICA": 70,
    #          "INTERCORRENCIA POS-TRANSPLANTE": 71,
    #          "GERIATRIA": 72,
    #          "SAUDE MENTAL": 73,
    #          "UTI ADULTO - TIPO I": 74,
    #          "UTI ADULTO - TIPO II": 75,
    #          "UTI ADULTO - TIPO III": 76,
    #          "UTI PEDIATRICA - TIPO I": 77,
    #          "UTI PEDIATRICA - TIPO II": 78,
    #          "UTI PEDIATRICA - TIPO III": 79,
    #          "UTI NEONATAL - TIPO I": 80,
    #          "UTI NEONATAL - TIPO II": 81,
    #          "UTI NEONATAL - TIPO III": 82,
    #          "UTI DE QUEIMADOS": 83,
    #          "ACOLHIMENTO NOTURNO": 84,
    #          "UTI CORONARIANA TIPO II - UCO TIPO II": 85,
    #          "UTI CORONARIANA TIPO III - UCO TIPO III": 86,
    #          "SAUDE MENTAL": 87,
    #          "QUEIMADO ADULTO": 88,
    #          "QUEIMADO PEDIATRICO": 89,
    #          "QUEIMADO ADULTO": 90,
    #          "QUEIMADO PEDIATRICO": 91,
    #          "UNIDADE DE CUIDADOS INTERMEDIARIOS NEONATAL CONVENCIONAL": 92,
    #          "UNIDADE DE CUIDADOS INTERMEDIARIOS NEONATAL CANGURU": 93,
    #          "UNIDADE DE CUIDADOS INTERMEDIARIOS PEDIATRICO": 94,
    #          "UNIDADE DE CUIDADOS INTERMEDIARIOS ADULTO": 95
    #          } # not used, dict is here for reference

   
    """ TODO: Fix Duplicate Keys
        Duplicate key 'CARDIOLOGIA' in dictionarypylint(duplicate-key)
        Duplicate key 'ONCOLOGIA' in dictionarypylint(duplicate-key)
        Duplicate key 'AIDS' in dictionarypylint(duplicate-key)
        Duplicate key 'GERIATRIA' in dictionarypylint(duplicate-key)
        Duplicate key 'SAUDE MENTAL' in dictionarypylint(duplicate-key)
        Duplicate key 'QUEIMADO ADULTO' in dictionarypylint(duplicate-key)
        Duplicate key 'QUEIMADO PEDIATRICO' in dictionarypylint(duplicate-key)
    """
    wards = {"BUCO MAXILO FACIAL": 1,
             "CARDIOLOGIA": 2,  # TODO: Fix Duplicate Keys 2, 32
             "CIRURGIA GERAL": 3,
             "ENDOCRINOLOGIA": 4,
             "GASTROENTEROLOGIA": 5,
             "GINECOLOGIA": 6,
             "CIRURGICO/DIAGNOSTICO/TERAPEUTICO": 7,
             "NEFROLOGIAUROLOGIA": 8,
             "NEUROCIRURGIA": 9,
             "OBSTETRICIA CIRURGICA": 10,
             "OFTALMOLOGIA": 11,
             "ONCOLOGIA": 12,  # TODO: Fix Duplicate Keys 12, 44
             "ORTOPEDIATRAUMATOLOGIA": 13,
             "OTORRINOLARINGOLOGIA": 14,
             "PLASTICA": 15,
             "TORACICA": 16,
             "AIDS": 31,  # TODO: Fix Duplicate Keys 31, 69
             "CARDIOLOGIA": 32, # TODO: Fix Duplicate Keys 2, 32
             "CLINICA GERAL": 33,
             "CRONICOS": 34,
             "DERMATOLOGIA": 35,
             "GERIATRIA": 36,  # TODO: Fix Duplicate Keys 36, 72
             "HANSENOLOGIA": 37,
             "HEMATOLOGIA": 38,
             "NEFROUROLOGIA": 40,
             "NEONATOLOGIA": 41,
             "NEUROLOGIA": 42,
             "OBSTETRICIA CLINICA": 43,
             "ONCOLOGIA": 44,  # TODO: Fix Duplicate Keys 12, 44
             "PEDIATRIA CLINICA": 45,
             "PNEUMOLOGIA": 46,
             "PSIQUIATRIA": 47,
             "REABILITACAO": 48,
             "PNEUMOLOGIA SANITARIA": 49,
             "UNIDADE ISOLAMENTO": 66,
             "TRANSPLANTE": 67,
             "PEDIATRIA CIRURGICA": 68,
             "AIDS": 69,  # TODO: Fix Duplicate Keys 31, 69
             "FIBROSE CISTICA": 70,
             "INTERCORRENCIA POS-TRANSPLANTE": 71,
             "GERIATRIA": 72,  # TODO: Fix Duplicate Keys 36, 72
             "SAUDE MENTAL": 73,  # TODO: Fix Duplicate Keys 73, 87
             "ACOLHIMENTO NOTURNO": 84,
             "SAUDE MENTAL": 87,  # TODO: Fix Duplicate Keys 73, 87
             "QUEIMADO ADULTO": 88,  # TODO: Fix Duplicate Keys 88, 90
             "QUEIMADO PEDIATRICO": 89,
             "QUEIMADO ADULTO": 90,  # TODO: Fix Duplicate Keys 88, 90
             "QUEIMADO PEDIATRICO": 91
             }
    icus = {
        "UNIDADE INTERMEDIARIA": 64,
        "UNIDADE INTERMEDIARIA NEONATAL": 65,
        "UTI ADULTO - TIPO I": 74,
        "UTI ADULTO - TIPO II": 75,
        "UTI ADULTO - TIPO III": 76,
        "UTI PEDIATRICA - TIPO I": 77,
        "UTI PEDIATRICA - TIPO II": 78,
        "UTI PEDIATRICA - TIPO III": 79,
        "UTI NEONATAL - TIPO I": 80,
        "UTI NEONATAL - TIPO II": 81,
        "UTI NEONATAL - TIPO III": 82,
        "UTI DE QUEIMADOS": 83,
        "UTI CORONARIANA TIPO II - UCO TIPO II": 85,
        "UTI CORONARIANA TIPO III - UCO TIPO III": 86,
        "UNIDADE DE CUIDADOS INTERMEDIARIOS NEONATAL CONVENCIONAL": 92,
        "UNIDADE DE CUIDADOS INTERMEDIARIOS NEONATAL CANGURU": 93,
        "UNIDADE DE CUIDADOS INTERMEDIARIOS PEDIATRICO": 94,
        "UNIDADE DE CUIDADOS INTERMEDIARIOS ADULTO": 95
    }
    df_wards = df_cnes.loc[df_cnes.Tipo_de_leito_traducao.isin(wards), :].groupby(
        df_cnes.CO_MUNICIPIO_GESTOR).sum().sort_values('QT_EXIST')
    df_icus = df_cnes.loc[df_cnes.Tipo_de_leito_traducao.isin(icus), :].groupby(
        df_cnes.CO_MUNICIPIO_GESTOR).sum().sort_values('QT_EXIST')
    df_leitos = df_wards.join(df_icus, lsuffix='_ward', rsuffix='_icus').fillna(0)

    bed_ward = df_leitos.at[city_code, 'QT_EXIST_ward']*0.5
    bed_icu = df_leitos.at[city_code, 'QT_EXIST_icus']*0.5
    
    return bed_ward, bed_icu


def parameter_for_rt_fit_analisys(city_code, 
                                  est_incubation_period,
                                  est_infectious_period,
                                  expected_mortality,
                                  expected_initial_rt,
                                  initial_deaths=50):
    """

    :param incubation_period: 2 days
    :param expected_initial_rt: 2  # estimated basic reproduction number
    :param expected_mortality:  0.0065  # Verity mortality rate
    :param city:
    :return:
    """
    df_MS_CASES = import_MS_cases()
    df_ibge = import_ibge()

    codigo_da_cidade_ibge = city_code  # 355030
    # select datasets in the city with rows only with > x deaths
    df_cidade = df_MS_CASES.loc[
        (df_MS_CASES.ibge_code_trimmed == codigo_da_cidade_ibge)
        & (df_MS_CASES.obitosAcumulado >= initial_deaths)].reset_index()

    pop_cidade = df_ibge['População_estimada'].loc[df_ibge.city_code_fixed == codigo_da_cidade_ibge].values

    round_infectious_period = 3 #np.ceil(est_infectious_period)
    # deaths_delay_post_infection =  2 #infection_to_death_period.mean()
    # deaths_delay_minus_infectious_period = deaths_delay_post_infection - round_infectious_period
    #infection_to_death_period.mean() / (est_incubation_period + est_infectious_period)
    I0_fit = (df_cidade.loc[round_infectious_period, 'obitosAcumulado'] - df_cidade.loc[0, 'obitosAcumulado']) * (est_infectious_period / round_infectious_period) / expected_mortality
    #I0_fit = (df_cidade.loc[deaths_delay_post_infection, 'obitosAcumulado'] - df_cidade.loc[deaths_delay_minus_infectious_period, 'obitosAcumulado'])*(est_infectious_period/round_infectious_period) / expected_mortality
    E0_fit = (I0_fit * expected_initial_rt * est_incubation_period ) / est_infectious_period
    R0_fit = (df_cidade.loc[0, 'obitosAcumulado'] / expected_mortality)
    M0_fit = df_cidade.loc[0, 'obitosAcumulado']
    population_fit = int(pop_cidade)

    E0_fit, I0_fit, R0_fit, M0_fit = 300, 200, 0, 0

    return E0_fit, I0_fit, R0_fit, M0_fit, population_fit, df_cidade


def analysis_type(analysis):
    """ Maps analysis string to IC_analysis flag

    Args:
        analysis ([str]): 'Confidence Interval', 'Single Run', 'Sensitivity', 'Rt'

    Returns:
        IC_Analysis : int 
    1: CONFIDENCE INTERVAL for a lognormal distribution
    2: SINGLE RUN
    3: r0 Sensitivity analysis: Calculate an array for r0 
    to perform a sensitivity analysis with 0.1 intervals
    4:Time variable inputted Rt analysis with confidence interval
    """   
    if analysis == 'Confidence Interval':
        IC_analysis = 1
        print('Confidence Interval Analysis (r0, gamma and alpha, lognormal distribution)')
    elif analysis == 'Single Run':
        IC_analysis = 2
        print('Single Run Analysis')
    elif analysis == 'Sensitity':
        IC_analysis = 3
        print('Sensitivity')
    elif analysis == 'Rt':
        IC_analysis = 4
        print('Time variable inputted Rt analysis with confidence interval')
    else:
        sys.exit('ERROR: Not programmed such Analysis, please enter 1, 2, 3 or 4')
    
    return IC_analysis


def contact_matrix_params(proportion_elderly):
    contact_matrix = [None] * 3
    contact_matrix[0] = np.array([[16.24264923, 0.54534103], [3.2788393, 0.72978211]])
    contact_matrix[1] = np.array([[16.24264923, 0.2391334], [1.43777922, 0.38361959]])
    contact_matrix[2] = np.array([[7.47115329, 0.2391334], [1.43777922, 0.38361959]])

    Population_proportion = np.array([1 - proportion_elderly, proportion_elderly])
    Normalization_constant = np.zeros(3)

    for k in range(3):
        M_matrix = np.zeros((len(contact_matrix[k]), len(contact_matrix[k][0])))

        for i in range(len(contact_matrix[k])):
            for j in range(len(contact_matrix[k][0])):
                M_matrix[i][j] = contact_matrix[k][i][j] * Population_proportion[i] / Population_proportion[j]
        Temp, _ = np.linalg.eig(M_matrix)
        Normalization_constant[k] = max(Temp.real)
    
    return Normalization_constant, contact_matrix


def sivep_rates(proportion_elderly, runs):
    # source sivep-gripe: INFLUD-31-08-2020.csv 
    # https://opendatasus.saude.gov.br/dataset/bd-srag-2020
    # Find calculations on notebooks calculo_mortalidade_uti
    
    IFR = (0.0034, 0.0034) # source silva et al https://www.medrxiv.org/content/10.1101/2020.05.13.20101253v3 DOI https://doi.org/10.1101/2020.08.28.20180463

    proportion_bed_need_over_deaths_group = {
        "ward_elderly": 1.209922,
        "ward_young": 3.911702,
        "icu_elderly": 0.896583,
        "icu_young": 1.729259
    }

    proportion_elderly_total_deaths = 0.5125

    proportion_elderly_ward_need_over_deaths_in_elderly = proportion_bed_need_over_deaths_group["ward_elderly"]
    proportion_young_ward_need_over_deaths_in_young = proportion_bed_need_over_deaths_group["ward_young"]
    proportion_elderly_icu_need_over_deaths_in_elderly = proportion_bed_need_over_deaths_group["icu_elderly"]
    proportion_young_icu_need_over_deaths_in_young = proportion_bed_need_over_deaths_group["icu_young"]  
    
    proportion_young_total_deaths = (1 - proportion_elderly_total_deaths)
    proportion_young = (1 - proportion_elderly)

    mortality_rate_elderly_intervals =  (IFR[0]*(proportion_elderly_total_deaths/proportion_elderly), IFR[1]*(proportion_elderly_total_deaths/proportion_elderly))#(0.03495*0.59, 0.03495*0.59)  # (0.03495*0.59, 0.03495*2.02)#try to capture verity error CIs (0.03495, 0.03495)
    mortality_rate_young_intervals= (IFR[0]*(proportion_young_total_deaths/proportion_young), IFR[1]*(proportion_young_total_deaths/proportion_young))#(0.00127*0.59, 0.00127*0.59) #(0.00127*0.59, 0.00127*2.02)  # (0.00127, 0.00127)

    icu_rate_elderly_intervals = np.array(mortality_rate_elderly_intervals) * proportion_elderly_icu_need_over_deaths_in_elderly #(0.0395*0.59, 0.0395*2.02) #try to capture verity error CIs (0.03495, 0.03495)
    icu_rate_young_intervals= np.array(mortality_rate_young_intervals) * proportion_young_icu_need_over_deaths_in_young #(0.0052*0.59, 0.0052*2.02) # (0.00127, 0.00127)
    
    ward_rate_elderly_intervals = np.array(mortality_rate_elderly_intervals) * proportion_elderly_ward_need_over_deaths_in_elderly #1.209922, icu_rate_elderly_intervals[1]*1.209922)#(0.1026*0.59, 0.1026*2.02) #try to capture verity error CIs (0.03495, 0.03495)
    ward_rate_young_intervals= np.array(mortality_rate_young_intervals) * proportion_young_ward_need_over_deaths_in_young #(0.0209*0.59, 0.0209*2.02) # (0.00127, 0.00127)

    # mortality_rate_elderly_intervals =  (0.03495*0.59, 0.03495*0.59) #try to capture verity error CIs (0.03495, 0.03495)
    # mortality_rate_young_intervals= (0.00127*0.59, 0.00127*0.59) # (0.00127, 0.00127)
    #
    #
    # ward_rate_elderly_intervals =  (0.1026*0.59, 0.1026*0.59) #try to capture verity error CIs (0.03495, 0.03495)
    # ward_rate_young_intervals= (0.0209*0.59, 0.0209*0.59) # (0.00127, 0.00127)
    #
    # icu_rate_elderly_intervals =  (0.0395*0.59, 0.0395*0.59) #try to capture verity error CIs (0.03495, 0.03495)
    # icu_rate_young_intervals= (0.0052*0.59, 0.0052*0.59) # (0.00127, 0.00127)

    rate_intervals = {
        "ward_elderly": ward_rate_elderly_intervals,
        "ward_young": ward_rate_young_intervals,
        "icu_elderly": icu_rate_elderly_intervals,
        "icu_young": icu_rate_young_intervals,
        "mortality_elderly": mortality_rate_elderly_intervals,
        "mortality_young": mortality_rate_young_intervals
    }

    rate_params = {}
    rate = {}
    for key, val in rate_intervals.items():
        rate_params[key] = make_lognormal_params_95_ci(*val)
        rate[key] = npr.lognormal(*map(np.log, rate_params[key]), runs)

    return rate


def lognormal_samples(incubation_period,
                      infectivity_period,
                      infection_to_death_period,
                      basic_reproduction_number,
                      runs):
  
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
    #infectivity_rate = 1/ npr.gamma(97.1875, 3.7187, runs)
    # beta = r0 * gamma
    contamination_rate = npr.lognormal(*map(np.log, R0__params), runs) * infectivity_rate
    infection_to_death_rate = 1 / npr.lognormal(*map(np.log, delta_inv_params), runs)

    ret = (incubation_rate, infectivity_rate, contamination_rate, infection_to_death_rate)

    return ret


def verity_rates():
    rate = {}
    rate['mortality_elderly'] = 0.03495
    rate['mortality_young'] = 0.00127
    rate['ward_elderly'] = 0.1026  # regular for old ones: 60+ years
    rate['ward_young'] = 0.0209  # regular for young ones: 0-59 years
    rate['icu_elderly'] = 0.0395  # UTI for old ones: 60+ years
    rate['icu_young'] = 0.0052  # UTI for young ones: 0-59 years

    return rate


def mortality_proportion_per_bed_type_and_age_group():
    
    mortality_proportion_bed_group = {
        "ward_elderly": 0.407565,
        "ward_young": 0.357161,
        "icu_elderly": 0.592435,
        "icu_young": 0.642839
    }

    return mortality_proportion_bed_group

def survive_proportion_per_bed_type_and_age_group():

    survive_proportion_bed_group = {
    "ward_elderly": 0.606622,
    "ward_young": 0.892248,
    "icu_elderly": 0.259130,
    "icu_young": 0.566919
    }
    
    return survive_proportion_bed_group  
    
    
def los_per_bed_type_and_age_group():

    los_bed_group = {
    "ward_survive_elderly": 9.394910,
    "ward_survive_young": 7.494861,
    "ward_death_elderly": 9.201503,
    "ward_death_young": 9.475104,
    "icu_survive_elderly": 9.847628,
    "icu_survive_young": 8.736200,
    "icu_death_elderly": 9.868753,
    "icu_death_young": 10.720939,
    "icu_discharged_survive_elderly": 6.572384,
    "icu_discharged_survive_young": 4.594393
    } 
    
    return los_bed_group 


def rates(estimation, proportion_elderly, runs):
    if estimation == 'Verity':
        return verity_rates()
    elif estimation == 'Sivep':
        return sivep_rates(proportion_elderly, runs)
    else:
        sys.exit('ERROR: Not programmed such Rate, please enter Verity or Sivep')


def get_input_data(analysis, fit_analysis, estimation,
                    runs, days_to_run, initial_deaths_to_fit, city_name):
    """
    Provides the inputs for the simulation
    
    Args:
        analysis ([type]): [description]
        fit_analysis ([type]): [description]
        estimation ([type]): [description]
        runs ([type]): [description]
        days_to_run ([type]): [description]
        initial_deaths_to_fit ([type]): [description]
        city_name ([type]): [description]

    :return: tuples for the demograph_parameters, covid_parameters 
    and model_parameters

    """
    #  Proportion of persons aged 60+ in Brazil, 2020 forecast, Source: IBGE's app
    proportion_elderly = 0.1425

    pH = 0.6  # probability of death for someone that needs a ward bed and does not receive it
    pU = 0.9  # probability of death for someone that needs an ICU bed and does not receive it
    
    # Length of Stay (in days), Source: Wuhan
    los_ward = 8.9  # regular  # los_leito
    los_icu = 8  # UTI
    
    infection_to_hospitalization = 5  # days
    infection_to_icu = 5  # days

    city_code = city_name_to_code(city_name)
    IC_analysis = analysis_type(analysis)
    
    Normalization_constant, contact_matrix = contact_matrix_params(proportion_elderly)

    rate = rates(estimation, proportion_elderly, runs)

    mortality_proportion_bed_group = mortality_proportion_per_bed_type_and_age_group()
    survive_proportion_bed_group = survive_proportion_per_bed_type_and_age_group()
    los_bed_group  = los_per_bed_type_and_age_group()

    bed_ward, bed_icu = import_cnes(city_code)

    # Basic Reproduction Number # ErreZero
    basic_reproduction_number_dct = {
        'Confidence Interval': (1.4, 3.9),#(1.4, 3.9),
        #'Confidence Interval': (1.4, 3.9),
        'Single Run': 2.2,
        'Sensitivity': 2.2,
        # 2.2 is from Li Q, Guan X, Wu P et al. 
        # Early Transmission Dynamics in Wuhan, China, 
        # of Novel Coronavirus–Infected Pneumonia.
        # New England Journal of Medicine. 2020 Mar 26;382(13):1199–207.
        # DOI: 10.1056/NEJMoa2001316
        # https://www.nejm.org/doi/full/10.1056/nejmoa2001316      
        'Rt': (1.4, 3.9)
    }
    basic_reproduction_number = basic_reproduction_number_dct[analysis]
    # (2.4, 3.3)
    # 1.4 / 2.2 / 3.9 

    # Incubation Period (in days)
    incubation_period_dct = {
        'Confidence Interval': (2.9, 2.9),
        #'Confidence Interval': (4.37 , 6.02),
        'Single Run': 5.2,
        'Sensitivity': 5.2,
        'Rt': (4.37 , 6.02)
    # https://www.sciencedirect.com/science/article/pii/S2213398420301895?via%3Dihub 
    # ###li et al - nature (3.9 , 9.6) #(4.1 - 3.0, 7.1 - 0.8)  # (4.1, 7.0)
    }
    incubation_period = incubation_period_dct[analysis]

    # Infectivity Period (in days)      # tempo_de_infecciosidade
    infectivity_period_dct = {
        'Confidence Interval': (2.9, 2.9),
        #'Confidence Interval': (0.01, 0.01),
        'Single Run': 3,
        'Sensitivity': 3,
        'Rt': (0.01, 0.01)
    }
    infectivity_period = infectivity_period_dct[analysis]
    # 3 days or 7 days - source nature   
    # (2.92, 3.22)

    # Infection to Death Period (in days)
    infection_to_death_period_dct = {
        'Confidence Interval': (16.9, 17.1),
        'Single Run': 17,
        'Sensitivity': 17,
        'Rt': (16.9, 17.1)
    }
    infection_to_death_period = infection_to_death_period_dct[analysis]

    df_rt_city = None
    if analysis == 'Rt':
        df_rt_city = get_rt_by_city(city_name)

    if (analysis == 'Confidence Interval') or  (analysis == 'Rt'):  
        (incubation_rate, infectivity_rate,
        contamination_rate, infection_to_death_rate) = lognormal_samples(
                incubation_period, infectivity_period, infection_to_death_period,
                basic_reproduction_number, runs)

    elif analysis == 'Single Run':
        # alpha
        incubation_rate = 1 / incubation_period
        # gamma
        infectivity_rate = 1 / infectivity_period
        # delta
        infection_to_death_rate = 1 / infection_to_death_period
        # beta = r0 * gamma
        contamination_rate = basic_reproduction_number * infectivity_rate

    elif analysis == 'Sensitivity':
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

    class Covid_parameters():
        def __init__(self):
            self.alpha=incubation_rate   # (1/day)
            self.beta=contamination_rate # (1/day)
            self.gamma=infectivity_rate  #(1/day)
            self.delta=infection_to_death_rate
            # Mortality Rates, Source: Verity, et al,
            # adjusted with population distribution IBGE 2020
            # taxa_mortalidade_i
            self.mortality_rate_elderly=rate['mortality_elderly']  # 0.03495  # old ones: 60+ years
            # taxa_mortalidade_j
            self.mortality_rate_young=rate['mortality_young']  # 0.00127  # young ones: 0-59 years
            self.pH=pH  # probability of death for someone that needs a ward bed and does not receive it
            self.pU=pU  # probability of death for someone that needs an ICU bed and does not receive it
            # Length of Stay (in days), Source: Wuhan
            self.los_ward=los_ward  # regular  # los_leito
            self.los_icu=los_icu  # UTI
            self.infection_to_hospitalization=infection_to_hospitalization  # days
            self.infection_to_icu=infection_to_icu  # days
            # Internation Rate by type and age, 
            # Source for hospitalization verity et al;
            # Proportion those need ICU:
            # Severe Outcomes Among Patients with Coronavirus Disease 2019 CDC
            # tax_int_i
            self.internation_rate_ward_elderly=rate['ward_elderly']  # 0.1026 # regular for old ones: 60+ years
            # tax_int_j
            self.internation_rate_ward_young=rate['ward_young']  # 0.0209 # regular for young ones: 0-59 years
            # tax_uti_i
            self.internation_rate_icu_elderly=rate['icu_elderly']  # 0.0395 # UTI for old ones: 60+ years
            # tax_uti_j
            self.internation_rate_icu_young=rate['icu_young']  # 0.0052 # UTI for young ones: 0-59 years
            self.ward_mortality_proportion_elderly = mortality_proportion_bed_group["ward_elderly"]
            self.ward_mortality_proportion_young = mortality_proportion_bed_group["ward_young"]
            self.icu_mortality_proportion_elderly = mortality_proportion_bed_group["icu_elderly"]
            self.icu_mortality_proportion_young = mortality_proportion_bed_group["icu_young"]
            self.WARD_survive_proportion_i = survive_proportion_bed_group["ward_elderly"]
            self.WARD_survive_proportion_j = survive_proportion_bed_group["ward_young"]
            self.ICU_survive_proportion_i = survive_proportion_bed_group["icu_elderly"]
            self.ICU_survive_proportion_j = survive_proportion_bed_group["icu_young"]

            self.los_WARD_survive_i = los_bed_group["ward_survive_elderly"]
            self.los_WARD_survive_j = los_bed_group["ward_survive_young"]
            self.los_WARD_death_i = los_bed_group["ward_death_elderly"]
            self.los_WARD_death_j = los_bed_group["ward_death_young"]
            self.los_ICU_survive_i = los_bed_group["icu_survive_elderly"]
            self.los_ICU_survive_j = los_bed_group["icu_survive_young"]
            self.los_ICU_death_i = los_bed_group["icu_death_elderly"]
            self.los_ICU_death_j = los_bed_group["icu_death_young"]
            self.los_discharged_ICU_survive_i = los_bed_group["icu_discharged_survive_elderly"]
            self.los_discharged_ICU_survive_j = los_bed_group["icu_discharged_survive_young"]

    covid_parameters = Covid_parameters()

    expected_mortality = np.mean(rate['mortality_elderly']) * proportion_elderly + (1-proportion_elderly) * np.mean(rate['mortality_young'])
    expected_initial_rt = np.mean(basic_reproduction_number)  # botar pra fora??
    est_infectious_period = np.mean(infectivity_period)
    est_incubation_period = np.mean(incubation_period)

    E0, I0, R0, M0, N0, df_cidade = parameter_for_rt_fit_analisys(city_code,
                                est_incubation_period, est_infectious_period,
                                expected_mortality, expected_initial_rt,
                                initial_deaths_to_fit)

    ### Criando objeto com status iniciais, juntando todas as infos que mudam o inicio
    ### Os parametros padrao podem ser mudados, como cama/UTI por cidade
    conditions = Conditions(E0, I0, R0, M0, N0,
            fit_analysis, IC_analysis, covid_parameters, proportion_elderly)  

    class Model_parameters():
        def __init__(self):
            # Social contact reduction factor (without, vertical, horizontal) isolation
            # niveis_isolamento
            self.isolation_level=[" (sem isolamento)", " (isolamento vertical)"]
            # Scenaries for health system colapse
            # self.lotation=(0.3, 0.5, 0.8, 1)  # 30, 50, 80, 100% capacity
            # self.contact_reduction_elderly = omega_i
            # self.contact_reduction_elderly = omega_j
            self.init_exposed_elderly=conditions.Ei0  # initial exposed population old ones: 60+ years
            self.init_exposed_young=conditions.Ej0  # initial exposed population young ones: 0-59 years
            self.init_infected_elderly=conditions.Ii0  # initial infected population old ones: 60+ years
            self.init_infected_young=conditions.Ij0  # initial infected population young ones: 0-59 years
            self.init_removed_elderly=conditions.Ri0  # initial removed population old ones: 60+ years
            self.init_removed_young=conditions.Rj0  # initial removed population young ones: 0-59 years
            self.init_hospitalized_ward_elderly=conditions.Hi0  # initial ward hospitalized old ones: 60+ years
            self.init_hospitalized_ward_young=conditions.Hj0  # initial ward hospitalized young ones: 0-59 years
            self.init_hospitalized_ward_elderly_excess=conditions.WARD_excess_i0
            # initial ward hospitalized demand excess old ones: 60+ years
            self.init_hospitalized_ward_young_excess=conditions.WARD_excess_j0
            # initial ward hospitalized demand excess young ones: 0-59 years
            self.init_hospitalized_icu_elderly=conditions.Ui0  # initial icu hospitalized old ones: 60+ years
            self.init_hospitalized_icu_young=conditions.Uj0  # initial icu hospitalized young ones: 0-59 years
            self.init_hospitalized_icu_elderly_excess=conditions.ICU_excess_i0
            # initial iCU hospitalized demand excess old ones: 60+ years
            self.init_hospitalized_icu_young_excess=conditions.ICU_excess_j0
            # initial iCU hospitalized demand excess young ones: 0-59 years
            self.init_deceased_elderly=conditions.Mi0  # initial deceased population old ones: 60+ years
            self.init_deceased_young=conditions.Mj0  # initial deceased population young ones: 0-59 years
            self.t_max=days_to_run  # number of days to run
            self.population=N0  # Brazilian Population
            # Brazilian old people proportion (age: 60+), 2020 forecast
            self.population_rate_elderly=proportion_elderly  # percentual_pop_idosa
            # Proportion of persons aged 60+ in Brazil, Source: IBGE's app
            # Brazilian bed places , Source: CNES, 05/05/2020
            # http://cnes2.datasus.gov.br/Mod_Ind_Tipo_Leito.asp?VEstado=00
            self.bed_ward=bed_ward  # capacidade_leitos
            self.bed_icu=bed_icu  # capacidade_UTIs
            self.IC_analysis=IC_analysis  # flag for type of analysis
            self.analysis=analysis
            # 1: confidence interval, 2: single run, 3: r0 sensitivity analysis
            self.contact_matrix=contact_matrix
            self.Normalization_constant=Normalization_constant # for contact matrix
            self.city=city_code
            self.city_name=city_name
            self.fit_analysis=fit_analysis  # 0: without;  1: with
            self.df_cidade = df_cidade
            self.df_rt_city = df_rt_city
            self.initial_deaths_to_fit = initial_deaths_to_fit
            self.runs = runs

    model_parameters = Model_parameters()

    parametros = {'City name': [city_name],
                  'City code IBGE': [model_parameters.city],
                  '1/alpha = (incubation_period in traditional SEIR)': [incubation_period],
                  'basic_reproduction_number = beta/gamma': [basic_reproduction_number],
                  '1/gamma = (infectivity_period in traditional SEIR)': [infectivity_period],
                  'runs': [runs],
                  'isolation_level': [model_parameters.isolation_level],
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
                  'population': [N0],
                  'population_rate_elderly': [proportion_elderly],
                  'bed_ward_number': [model_parameters.bed_ward],
                  'bed_icu_number': [model_parameters.bed_icu],
                  'IC_analysis': [IC_analysis],
                  'fit_analysis': [fit_analysis],
                  'initial_deaths_to_fit': [initial_deaths_to_fit],
                  'Elderly IFR - infection fatality rate (%)': (rate['mortality_elderly'].mean()*100),
                  'Young IFR - infection fatality rate (%)': (rate['mortality_young'].mean()*100)}
    output_parameters = pd.DataFrame(parametros).T
    print(output_parameters)
    print('')
    # print('taxa_mortalidade_i' )
    # print(mortality_rate_elderly_params)
    # print('taxa_uti')
    # print(icu_rate_elderly_params)
    return covid_parameters, model_parameters, output_parameters
