from collections import namedtuple
import numpy as np
import numpy.random as npr
import pandas as pd
import sys

#Santos/SP    354850
#GoiÃ¢nia/GO    520870
#Porto Velho/RO    110020
#Aracaju/SE    280030
#CuiabÃ¡/MT    510340
#Duque de Caxias/RJ    330170
#SÃ£o GonÃ§alo/RJ    330490
#Belo Horizonte/MG    310620
#Osasco/SP    353440
#SÃ£o Bernardo do Campo/SP    354870
#Curitiba/PR    410690
#JoÃ£o Pessoa/PB    250750
#Teresina/PI    221100
#MaceiÃ³/AL    270430
#JaboatÃ£o dos Guararapes/PE    260790
#Campinas/SP    350950
#Natal/RN    240810
#Guarulhos/SP    351880
#SÃ£o LuÃ­s/MA    211130
#BrasÃ­lia/DF    530010
#Salvador/BA    292740
#Manaus/AM    130260
#BelÃ©m/PA    150140
#Recife/PE    261160
#Fortaleza/CE    230440
#Rio de Janeiro/RJ    330455
#SÃ£o Paulo/SP    355030

class define_city:
    """
    Defines city for analisys, analisys type and number of runs;
    Gets number of wards and ICU beds from CNES dataset extracted from http://cnes.datasus.gov.br/pages/downloads/arquivosBaseDados.jsp - BASE_DE_DADOS_CNES_202002.ZIP

    Imported data:
    Ward quantity
    ICU quantity

    """
    def __init__(self):
        self.cidade = 230440
        self.icanalisis = 1
        self.runs = 30

        df_cnes = pd.read_csv(r'..\vertical-master-12_ago\data\cnes_simplificado_02-2020.csv', sep=';')  # source

        leitos = {"BUCO MAXILO FACIAL": 1,
                  "CARDIOLOGIA": 2,
                  "CIRURGIA GERAL": 3,
                  "ENDOCRINOLOGIA": 4,
                  "GASTROENTEROLOGIA": 5,
                  "GINECOLOGIA": 6,
                  "CIRURGICO/DIAGNOSTICO/TERAPEUTICO": 7,
                  "NEFROLOGIAUROLOGIA": 8,
                  "NEUROCIRURGIA": 9,
                  "OBSTETRICIA CIRURGICA": 10,
                  "OFTALMOLOGIA": 11,
                  "ONCOLOGIA": 12,
                  "ORTOPEDIATRAUMATOLOGIA": 13,
                  "OTORRINOLARINGOLOGIA": 14,
                  "PLASTICA": 15,
                  "TORACICA": 16,
                  "AIDS": 31,
                  "CARDIOLOGIA": 32,
                  "CLINICA GERAL": 33,
                  "CRONICOS": 34,
                  "DERMATOLOGIA": 35,
                  "GERIATRIA": 36,
                  "HANSENOLOGIA": 37,
                  "HEMATOLOGIA": 38,
                  "NEFROUROLOGIA": 40,
                  "NEONATOLOGIA": 41,
                  "NEUROLOGIA": 42,
                  "OBSTETRICIA CLINICA": 43,
                  "ONCOLOGIA": 44,
                  "PEDIATRIA CLINICA": 45,
                  "PNEUMOLOGIA": 46,
                  "PSIQUIATRIA": 47,
                  "REABILITACAO": 48,
                  "PNEUMOLOGIA SANITARIA": 49,
                  "UNIDADE INTERMEDIARIA": 64,
                  "UNIDADE INTERMEDIARIA NEONATAL": 65,
                  "UNIDADE ISOLAMENTO": 66,
                  "TRANSPLANTE": 67,
                  "PEDIATRIA CIRURGICA": 68,
                  "AIDS": 69,
                  "FIBROSE CISTICA": 70,
                  "INTERCORRENCIA POS-TRANSPLANTE": 71,
                  "GERIATRIA": 72,
                  "SAUDE MENTAL": 73,
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
                  "ACOLHIMENTO NOTURNO": 84,
                  "UTI CORONARIANA TIPO II - UCO TIPO II": 85,
                  "UTI CORONARIANA TIPO III - UCO TIPO III": 86,
                  "SAUDE MENTAL": 87,
                  "QUEIMADO ADULTO": 88,
                  "QUEIMADO PEDIATRICO": 89,
                  "QUEIMADO ADULTO": 90,
                  "QUEIMADO PEDIATRICO": 91,
                  "UNIDADE DE CUIDADOS INTERMEDIARIOS NEONATAL CONVENCIONAL": 92,
                  "UNIDADE DE CUIDADOS INTERMEDIARIOS NEONATAL CANGURU": 93,
                  "UNIDADE DE CUIDADOS INTERMEDIARIOS PEDIATRICO": 94,
                  "UNIDADE DE CUIDADOS INTERMEDIARIOS ADULTO": 95
                  } # not used, dict is here for reference

        wards = {"BUCO MAXILO FACIAL": 1,
                 "CARDIOLOGIA": 2,
                 "CIRURGIA GERAL": 3,
                 "ENDOCRINOLOGIA": 4,
                 "GASTROENTEROLOGIA": 5,
                 "GINECOLOGIA": 6,
                 "CIRURGICO/DIAGNOSTICO/TERAPEUTICO": 7,
                 "NEFROLOGIAUROLOGIA": 8,
                 "NEUROCIRURGIA": 9,
                 "OBSTETRICIA CIRURGICA": 10,
                 "OFTALMOLOGIA": 11,
                 "ONCOLOGIA": 12,
                 "ORTOPEDIATRAUMATOLOGIA": 13,
                 "OTORRINOLARINGOLOGIA": 14,
                 "PLASTICA": 15,
                 "TORACICA": 16,
                 "AIDS": 31,
                 "CARDIOLOGIA": 32,
                 "CLINICA GERAL": 33,
                 "CRONICOS": 34,
                 "DERMATOLOGIA": 35,
                 "GERIATRIA": 36,
                 "HANSENOLOGIA": 37,
                 "HEMATOLOGIA": 38,
                 "NEFROUROLOGIA": 40,
                 "NEONATOLOGIA": 41,
                 "NEUROLOGIA": 42,
                 "OBSTETRICIA CLINICA": 43,
                 "ONCOLOGIA": 44,
                 "PEDIATRIA CLINICA": 45,
                 "PNEUMOLOGIA": 46,
                 "PSIQUIATRIA": 47,
                 "REABILITACAO": 48,
                 "PNEUMOLOGIA SANITARIA": 49,
                 "UNIDADE ISOLAMENTO": 66,
                 "TRANSPLANTE": 67,
                 "PEDIATRIA CIRURGICA": 68,
                 "AIDS": 69,
                 "FIBROSE CISTICA": 70,
                 "INTERCORRENCIA POS-TRANSPLANTE": 71,
                 "GERIATRIA": 72,
                 "SAUDE MENTAL": 73,
                 "ACOLHIMENTO NOTURNO": 84,
                 "SAUDE MENTAL": 87,
                 "QUEIMADO ADULTO": 88,
                 "QUEIMADO PEDIATRICO": 89,
                 "QUEIMADO ADULTO": 90,
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

        self.bed_ward = df_leitos.at[self.cidade, 'QT_EXIST_ward']
        self.bed_icu = df_leitos.at[self.cidade, 'QT_EXIST_icus']

        df_ibge = pd.read_csv(r'data\populacao_ibge.csv', sep=';', encoding="ISO-8859-1")
        df_ibge['city_name_fixed'] = df_ibge['Município'].map(fix_city_name)
        df_ibge['city_code_fixed'] = df_ibge['Município'].map(fix_city_code)

        self.city_name= str(np.asscalar(df_ibge['city_name_fixed'].loc[df_ibge.city_code_fixed == self.cidade].values))


class Conditions:

    def __init__(self,
                 I0,
                 E0,
                 R0,
                 M0,
                 population,
                 fit_analysis,
                 covid_parameters,
                 elderly_proportion=.1425) -> None:

        self.I0 = I0
        self.E0 = E0
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
        if define_city().icanalisis == 2:
            self.Hi0 = self.Ii0 * covid_parameters.internation_rate_ward_elderly
            self.Hj0 = self.Ij0 * covid_parameters.internation_rate_ward_young
            # Leitos UTIs demandados
            self.Ui0 = self.Ii0 * covid_parameters.internation_rate_icu_elderly
            self.Uj0 = self.Ij0 * covid_parameters.internation_rate_icu_young
        else:
            # Leitos normais demandados
            self.Hi0 = self.Ii0 * covid_parameters.internation_rate_ward_elderly.mean()
            self.Hj0 = self.Ij0 * covid_parameters.internation_rate_ward_young.mean()
            # Leitos UTIs demandados
            self.Ui0 = self.Ii0 * covid_parameters.internation_rate_icu_elderly.mean()
            self.Uj0 = self.Ij0 * covid_parameters.internation_rate_icu_young.mean()

        # Excesso de demanda para leitos
        self.dHi0 = 0
        self.dHj0 = 0
        self.dUi0 = 0
        self.dUj0 = 0
        # Obitos
        # M_0 = 3_000
        if fit_analysis != 1:
            self.Mi0 = self.Ri0 * np.mean(covid_parameters.mortality_rate_elderly)
            self.Mj0 = self.Rj0 * np.mean(covid_parameters.mortality_rate_young)
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

def fix_city_name(row):
    row = row[7:]
    return row

def fix_city_code(row):
    row = str(row)[:6]
    if row == 'Total':
        row = 000000
    row = int(row)
    return row

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

    # fix strings on datasets
    df_ibge['city_name_fixed'] = df_ibge['Município'].map(fix_city_name)
    df_ibge['city_code_fixed'] = df_ibge['Município'].map(fix_city_code)
    df_wcota['ibge_code_trimmed'] = df_wcota['ibgeID'].map(fix_city_code)

    # select datasets in the city with rows only with > x deaths
    df_cidade = df_wcota.loc[
        (df_wcota.ibge_code_trimmed == codigo_da_cidade_ibge) & (df_wcota.deaths >= 50)].reset_index()

    pop_cidade = df_ibge['População_estimada'].loc[df_ibge.city_code_fixed == codigo_da_cidade_ibge].values


    round_infectious_period = np.ceil(est_infectious_period)
    deaths_delay_post_infection =  2 #infection_to_death_period.mean()
    deaths_delay_minus_infectious_period = deaths_delay_post_infection - round_infectious_period

    #infection_to_death_period.mean() / (est_incubation_period + est_infectious_period)

    I0_fit = (df_cidade.loc[round_infectious_period, 'deaths'] - df_cidade.loc[0, 'deaths']) * (est_infectious_period / round_infectious_period) / expected_mortality

    #I0_fit = (df_cidade.loc[deaths_delay_post_infection, 'deaths'] - df_cidade.loc[deaths_delay_minus_infectious_period, 'deaths'])*(est_infectious_period/round_infectious_period) / expected_mortality
    E0_fit = (I0_fit * expected_initial_rt * est_incubation_period ) / est_infectious_period
    R0_fit = (df_cidade.loc[0, 'deaths'] / expected_mortality)
    M0_fit = df_cidade.loc[0, 'deaths']
    population_fit = int(pop_cidade)



    return E0_fit, I0_fit, R0_fit, M0_fit, population_fit


def get_input_data(IC_analysis, city):
    """
    Provides the inputs for the simulation
    :return: tuples for the demograph_parameters, covid_parameters 
    and model_parameters
    
    
    
    Degrees of isolation (i)
    no isolation, vertical, horizontal
        
    IC_Analysis
    1: Confidence Interval; 2: Single Run; 3: Sensitivity Analysis :4:Time variable inputted Rt analysis with confidence interval
    
    1: CONFIDENCE INTERVAL for a lognormal distribution
    2: SINGLE RUN
    3: r0 Sensitivity analysis: Calculate an array for r0 
    to perform a sensitivity analysis with 0.1 intervals
    4:Time variable inputted Rt analysis with confidence interval
    
    """

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

    runs = define_city().runs # 1_000 # number of runs for Confidence Interval analysis

    dfMS, startdate, state_name, sub_report, r0_fit = [], [], [], [], []

    fit_analysis = 0  # 0 # 1 #

    # if fit_analysis == 1:
    #     if IC_analysis == 1:
    #         pass
    #         #  print('With fit analysis')
    #         # [dfMS, startdate, state_name, population_fit, sub_report,
    #         #  E0_fit, I0_fit, R0_fit, M0_fit, r0_fit] = fit_curve()
    #     else:
    #         sys.exit('ERROR: Not programmed fit analysis for other case than Confidence Interval')

            # CONFIDENCE INTERVAL AND SENSITIVITY ANALYSIS
    # 95% Confidence interval bounds or range for sensitivity analysis
    # Basic Reproduction Number # ErreZero


    basic_reproduction_number = (1.4, 2.7)  # (2.4, 3.3)     # 1.4 / 2.2 / 3.9


    # if fit_analysis == 1:
    #     basic_reproduction_number = r0_fit

    # SINGLE RUN AND SENSITIVITY ANALYSIS
    # Incubation Period (in days)


    incubation_period = 5.2  # (4.1, 7.0) #
    # Infectivity Period (in days)      # tempo_de_infecciosidade
    infectivity_period = 3
    infection_to_death_period = 17
    pI = 0.1425  #  Proportion of persons aged 60+ in Brazil, 2020 forecast, Source: IBGE's app

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

    if IC_analysis == 1 or 4:  # CONFIDENCE INTERVAL for a lognormal distribution

        # PARAMETERS ARE ARRAYS

        # 95% Confidence interval bounds for Covid parameters
        # Incubation Period (in days)
        incubation_period = (4.79 , 6.79)#(3.9 , 9.6) #(4.1 - 3.0, 7.1 - 0.8)  # (4.1, 7.0) source li et al - nature

        # Infectivity Period (in days)

        infectivity_period = (0.01, 0.01)  # 3 days or 7 days - source nature

        infection_to_death_period = (16.9, 17.1)

        proportion_elderly = .1425
        proportion_young = (1 - proportion_elderly)

        IFR = (0.002, 0.006) # source estudo maranhao

        # source sivep-gripe: INFLUD-31-08-2020.csv https://opendatasus.saude.gov.br/dataset/bd-srag-2020 - Find calculations on notebooks calculo_mortalidade_uti
        proportion_elderly_total_deaths = 0.5125
        proportion_young_total_deaths = (1 - proportion_elderly_total_deaths)

        mortality_rate_elderly_intervals =  (IFR[0]*(proportion_elderly_total_deaths/proportion_elderly), IFR[1]*(proportion_elderly_total_deaths/proportion_elderly))#(0.03495*0.59, 0.03495*0.59)  # (0.03495*0.59, 0.03495*2.02)#try to capture verity error CIs (0.03495, 0.03495)
        mortality_rate_young_intervals= (IFR[0]*(proportion_young_total_deaths/proportion_young), IFR[1]*(proportion_young_total_deaths/proportion_young))#(0.00127*0.59, 0.00127*0.59) #(0.00127*0.59, 0.00127*2.02)  # (0.00127, 0.00127)

        proportion_elderly_icu_need_over_deaths_in_elderly = 0.896583
        proportion_young_icu_need_over_deaths_in_young = 1.729259

        icu_rate_elderly_intervals = np.array(mortality_rate_elderly_intervals) * proportion_elderly_icu_need_over_deaths_in_elderly #(0.0395*0.59, 0.0395*2.02) #try to capture verity error CIs (0.03495, 0.03495)
        icu_rate_young_intervals= np.array(mortality_rate_young_intervals) *proportion_young_icu_need_over_deaths_in_young #(0.0052*0.59, 0.0052*2.02) # (0.00127, 0.00127)

        proportion_elderly_ward_need_over_deaths_in_elderly = 1.209922
        proportion_young_ward_need_over_deaths_in_young = 3.911702

        ward_rate_elderly_intervals =  np.array(mortality_rate_elderly_intervals) * proportion_elderly_ward_need_over_deaths_in_elderly #1.209922, icu_rate_elderly_intervals[1]*1.209922)#(0.1026*0.59, 0.1026*2.02) #try to capture verity error CIs (0.03495, 0.03495)
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

        # Computes mean and std for a lognormal distribution
        mortality_rate_elderly_params = make_lognormal_params_95_ci(*mortality_rate_elderly_intervals)
        mortality_rate_young_params = make_lognormal_params_95_ci(*mortality_rate_young_intervals)

        alpha_inv_params = make_lognormal_params_95_ci(*incubation_period)
        gamma_inv_params = make_lognormal_params_95_ci(*infectivity_period)
        delta_inv_params = make_lognormal_params_95_ci(*infection_to_death_period)

        R0__params = make_lognormal_params_95_ci(*basic_reproduction_number)

        ward_rate_elderly_params = make_lognormal_params_95_ci(*ward_rate_elderly_intervals)
        ward_rate_young_params = make_lognormal_params_95_ci(*ward_rate_young_intervals)

        icu_rate_elderly_params = make_lognormal_params_95_ci(*icu_rate_elderly_intervals)
        icu_rate_young_params = make_lognormal_params_95_ci(*icu_rate_young_intervals)


        # samples for a lognormal distribution (Monte Carlo Method)
        # alpha
        incubation_rate = 1 / npr.lognormal(*map(np.log, alpha_inv_params), runs)
        # gamma
        infectivity_rate = 1 / npr.lognormal(*map(np.log, gamma_inv_params), runs)
        #infectivity_rate = 1/ npr.gamma(97.1875, 3.7187, runs)
        # beta = r0 * gamma
        contamination_rate = npr.lognormal(*map(np.log, R0__params), runs) * infectivity_rate
        infection_to_death_rate = 1 / npr.lognormal(*map(np.log, delta_inv_params), runs)
        mortality_rate_elderly_params = npr.lognormal(*map(np.log, mortality_rate_elderly_params), runs)
        mortality_rate_young_params = npr.lognormal(*map(np.log, mortality_rate_young_params), runs)

        ward_rate_elderly_params = npr.lognormal(*map(np.log, ward_rate_elderly_params), runs)
        ward_rate_young_params = npr.lognormal(*map(np.log, ward_rate_young_params), runs)

        icu_rate_elderly_params = npr.lognormal(*map(np.log, icu_rate_elderly_params), runs)
        icu_rate_young_params = npr.lognormal(*map(np.log, icu_rate_young_params), runs)




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

        mortality_rate_elderly_params = 0.03495
        mortality_rate_young_params = 0.00127

        ward_rate_elderly_params=0.1026  # regular for old ones: 60+ years
        ward_rate_young_params=0.0209  # regular for young ones: 0-59 years
        icu_rate_elderly_params=0.0395  # UTI for old ones: 60+ years
        icu_rate_young_params=0.0052  # UTI for young ones: 0-59 years

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

        mortality_rate_elderly_params = 0.03495
        mortality_rate_young_params = 0.00127

        ward_rate_elderly_params=0.1026,  # regular for old ones: 60+ years
        ward_rate_young_params=0.0209,  # regular for young ones: 0-59 years
        icu_rate_elderly_params=0.0395,  # UTI for old ones: 60+ years
        icu_rate_young_params=0.0052  # UTI for young ones: 0-59 years

    else:  # r0 Sensitivity analysis
        # PARAMETERS ARE ARRAYS

        # 95% Confidence interval bounds for Covid parameters
        # Incubation Period (in days)
        incubation_period = (4.1 - 3.0, 7.1 - 0.8)  # (4.1, 7.0)

        # Infectivity Period (in days)   # tempo_de_infecciosidade

        infectivity_period = (2.92, 3.22)  # 3 days or 7 days

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

        mortality_rate_elderly_params = 0.03495
        mortality_rate_young_params = 0.00127

        ward_rate_elderly_params=0.1026,  # regular for old ones: 60+ years
        ward_rate_young_params=0.0209,  # regular for young ones: 0-59 years
        icu_rate_elderly_params=0.0395,  # UTI for old ones: 60+ years
        icu_rate_young_params=0.0052  # UTI for young ones: 0-59 years

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
        mortality_rate_elderly=mortality_rate_elderly_params,  # 0.03495,  # old ones: 60+ years
        mortality_rate_young=mortality_rate_young_params,  # 0.00127,  # young ones: 0-59 years
        #mortality_rate_elderly=0.03495,  # old ones: 60+ years
        #mortality_rate_young=0.00127,  # young ones: 0-59 years
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
        internation_rate_ward_elderly=ward_rate_elderly_params,  # regular for old ones: 60+ years
        internation_rate_ward_young=ward_rate_young_params,  # regular for young ones: 0-59 years
        internation_rate_icu_elderly=icu_rate_elderly_params,  # UTI for old ones: 60+ years
        internation_rate_icu_young=icu_rate_young_params  # UTI for young ones: 0-59 years
        # internation_rate_ward_elderly=0.1026,  # regular for old ones: 60+ years
        # internation_rate_ward_young=0.0209,  # regular for young ones: 0-59 years
        # internation_rate_icu_elderly=0.0395,  # UTI for old ones: 60+ years
        # internation_rate_icu_young=0.0052  # UTI for young ones: 0-59 years
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

    expected_mortality = np.mean(covid_parameters.mortality_rate_elderly) * pI + (1-pI) * np.mean(covid_parameters.mortality_rate_young)
    expected_initial_rt = np.mean(basic_reproduction_number)  # botar pra fora??
    est_infectious_period = np.mean(infectivity_period)
    est_incubation_period = np.mean(incubation_period)

    #E0, I0, R0, M0, N0 = 50, 25, 0, 0, 6000000

    E0, I0, R0, M0, N0 = parameter_for_rt_fit_analisys(city, est_incubation_period, est_infectious_period, expected_mortality, expected_initial_rt)
    N0 =5_500_000


    ### Criando objeto com status iniciais, juntando todas as infos que mudam o inicio
    ### Os parametros padrao podem ser mudados, como cama/UTI por cidade
    conditions = Conditions(I0, E0, R0, M0, N0, fit_analysis, covid_parameters)
    city_params = define_city()

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
        population=N0,
        # Brazilian old people proportion (age: 60+), 2020 forecast
        population_rate_elderly=pI,
        # Proportion of persons aged 60+ in Brazil, Source: IBGE's app
        # Brazilian bed places , Source: CNES, 05/05/2020
        # http://cnes2.datasus.gov.br/Mod_Ind_Tipo_Leito.asp?VEstado=00
        bed_ward=city_params.bed_ward,  # bed ward
        bed_icu=city_params.bed_icu,  # bed ICUs
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

    parametros = {'City name': [city_params.city_name],
                  'City code IBGE': [model_parameters.city],
                  'incubation_period = 1/alpha': [incubation_period],
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
                  'population': [N0],
                  'population_rate_elderly': [pI],
                  'bed_ward': [model_parameters.bed_ward],
                  'bed_icu': [model_parameters.bed_icu],
                  'IC_analysis': [IC_analysis],
                  'fit_analysis': [fit_analysis],
                  'startdate': [startdate],
                  'state_name': [state_name],
                  'r0_fit': [r0_fit],
                  'sub_report': [sub_report]}

    output_parameters = pd.DataFrame(parametros).T
    print(output_parameters)
    print('')
    # print('taxa_mortalidade_i' )
    # print(mortality_rate_elderly_params)
    # print('taxa_uti')
    # print(icu_rate_elderly_params)
    return covid_parameters, model_parameters, output_parameters
