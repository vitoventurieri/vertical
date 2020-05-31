from collections import namedtuple
import numpy as np
import numpy.random as npr
import pandas as pd
#from datetime import datetime as dt


def make_lognormal_params_95_ci(lb, ub):
	'''
	Provides mean and standard deviation of a lognormal distribution
	input: lower bound and upper bound of the 95% confidence interval
	return: mean and std
	'''
	mean = (ub*lb)**(1/2)
	std = (ub/lb)**(1/4)
	
	
	# http://broadleaf.com.au/resource-material/lognormal-distribution-summary/
	# v1 = ub
	# v2 = lb
	# z1 = 1.96
	# z2 = 1.96
	# std = log( v1 / v2 ) / (z1 - z2)
	# mu = ( z2 * log(v1) - z1 * log(v2)) / (z2 - z1)
	
	return mean, std


def fit_curve():
	'''
	provides comparison to reported data

	Returns
	-------
	dfMS : dataframe
		raw data from xlsx file provided dayly by the Brazillian
		Health Ministerium at https://covid.saude.gov.br/
	startdate : string
		simulation initial date
	state_name : string
		 name of the state of interest
	population : int
		population of the state of interest
	sub_report: int
		subnotification factor
	E0 : int
		Exposed at startdate, defined arbitrarily as 80% of the
		Infected ones
	I0 : int
		Infected at startdate, defined as the difference between
		the cumulative infected at startdate - infected at backdate (in which
		backdate was considered arbitrarily as 13 days before startdate)
	R0 : int
		Removed at startdate, taken as the infected + deceased at backdate 
	M0 : int
		Deceased at startdate
	r0 : float
		basic reproduction number confidence interval

	PROCEDIMENTO
1) Selecionar local de interesse
2) Buscar intervalo de dias em que r está relativamente constante
obs: para Brasil, não foi visto evolução no tempo, simplesmente pegado IC do 
Imperial College
3) Define data em que r começa a ficar constante: startdate (extrai M0 como 
óbitos reportados nesta data)
4) Define período de infecciosidade: backdate = startdate - 13 dias (fit ARBITRÁRIO)
obs: modelo considera 10 dias
5) Contabiliza número acumulado de Infectados no período entre backdate e 
															 startdate: Infect
6) Define um fator de subnotificação: sub_report
obs: ARBITRÁRIO (15 para PE, SP, Brasil; 3 para SC)
7) Multiplica Infect pelo fator de subnotificação: Infect = Infect * sub_report
8) Define I0 (infectados em startdate) como diferença do número acumulado 
de infectados entre startdate e backdate, corrigida com subnotificação
9) Define R0 (removidos em startdate) como número acumulado de infectados + 
															 óbitos em backdate
10) Define E0 (expostos em startdate) como 80% de I0
obs: 80% ARBITRÁRIO

	'''
	
	
	# INPUT
	state_name = 'Rio de Janeiro (UF)' # 'Pernambuco' #'Brasil' # 'Santa Catarina' # 
	metodo = "subreport" # "fator_verity" # 
	sub_report = 10	
	
	# IMPORT DATA
	url = 'https://github.com/viniciusriosfuck/vertical/blob/fit_reported_data/HIST_PAINEL_COVIDBR_29mai2020.xlsx?raw=true'
	# df = pd.read_excel(r'C:\Users\Fuck\Downloads\HIST_PAINEL_COVIDBR_21mai2020.xlsx')
	df = pd.read_excel(url)
	# data	semanaEpi	populacaoTCU2019	casosAcumulado	obitosAcumulado	Recuperadosnovos	emAcompanhamentoNovos
	states = { 'coduf': [76, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53],
			'state_name': ['Brasil','Rondônia','Acre','Amazonas','Roraima','Pará','Amapá','Tocantins','Maranhão','Piauí','Ceará','Rio Grande do Norte','Paraíba','Pernambuco','Alagoas','Sergipe','Bahia','Minas Gerais','Espiríto Santo','Rio de Janeiro (UF)','São Paulo (UF)','Paraná','Santa Catarina','Rio Grande do Sul','Mato Grosso do Sul','Mato Grosso','Goiás','Distrito Federal'],
			'populationTCU2019': [210_147_125, 1_777_225, 881_935, 4_144_597, 605_761, 8_602_865, 845_731, 1_572_866, 7_075_181, 3_273_227, 9_132_078, 3_506_853, 4_018_127, 9_557_071, 3_337_357, 2_298_696, 14_873_064, 21_168_791, 4_018_650, 17_264_943, 45_919_049, 11_433_957, 7_164_788, 11_377_239, 2_778_986, 3_484_466, 7_018_354, 3_015_268]}
	states = pd.DataFrame(states, columns = ['coduf', 'state_name', 'populationTCU2019'])
	
	# INITIAL DATE
	# fonte r0: UF e cidades Brasil: nosso projeto
	# http://coletivocovid19.site/modelo/
	if state_name == 'Pernambuco':
		startdate = '2020-05-02'
		r0 = (1.1, 1.3)
		# coduf = 26 
		# population = 9_557_071
		sub_report = 13
	elif state_name == 'Santa Catarina':
		startdate = '2020-05-10'
		r0 = (1.1, 1.2)
		# coduf = 42
		# population = 7_164_788
		sub_report = 2
	elif state_name == 'São Paulo (UF)':
		startdate = '2020-03-15' # '2020-04-29' #
		r0 = (2.5, 3.5) # (1.15, 1.32) # (3.4, 5.3) #
		# coduf = 35
		# population = 45_919_049
		sub_report = 8
	elif state_name == 'Brasil':
		startdate = '2020-04-26'
		r0 = (2.25, 3.57) # fonte r0: Brasil: Imperial College
		# coduf = 76
		# population = 210_147_125
		sub_report = 8        
		# https://www1.folha.uol.com.br/equilibrioesaude/2020/04/brasil-tem-maior-taxa-de-contagio-por-coronavirus-do-mundo-aponta-estudo.shtml
	elif state_name == 'Rio de Janeiro (UF)':
		startdate = '2020-03-23' #
		r0 = (1.28, 1.60) # (1.15, 1.32) # (3.4, 5.3) # Imperial College 21
		# coduf = 
		# population = 17_264_943	
	
	
	states_set = states[states['state_name'] == state_name ]
	
	population = states_set['populationTCU2019'].values[0]
	coduf = states_set['coduf'].values[0]                 
	
	dfMS = df[df['coduf'] == coduf ]
	dfMS = dfMS[dfMS['codmun'].isnull()] # only rows without city
	
	dfMS['data'] = pd.to_datetime(dfMS['data'])
	dfMS['obitosAcumulado'] = pd.to_numeric(dfMS['obitosAcumulado'])
	
	M0_MS = dfMS['obitosAcumulado'].max() # most recent cumulative deaths
	R0_MS = dfMS['Recuperadosnovos'].max() + M0_MS  # most recent removed
	I0_MS = dfMS['emAcompanhamentoNovos'].max() # most recent active reported cases
	
	# IDENTIFY 13 DAYS AGO
	# Hypothesis: one takes 13 days to recover
	backdate = pd.to_datetime(startdate) - pd.DateOffset(days=3)
	#backdate = pd.to_datetime(backdate.date())#pd.to_datetime(backdate).dt.date#.normalize()
	# DECEASED
	M0 = dfMS['obitosAcumulado'][dfMS['data'] == startdate].values[0]
	# CUMULATIVE INFECTED FROM THE PREVIOUS 13 DAYS
	Infect = dfMS['casosAcumulado'][dfMS['data'].
								 between(backdate,startdate,inclusive = True)]
	#print(backdate)
	#print(Infect)
	#print(dfMS['data'])	
	# ESTIMATED INITIAL CONDITIONS
	if metodo == "subreport":
		Infect = Infect * sub_report
		# INFECTED
		I0 = max(Infect) - min(Infect)
		# RECOVERED
		R0 = min(Infect) + dfMS['obitosAcumulado'][dfMS['data'] == backdate].values[0]# max(Infect) - I0
	elif metodo == "fator_verity":
		I0 = M0 * 165 # estimated from Verity to Brazil: country, state, city
		R0 = I0 * 0.6 # Hypothesis: Removed correspond to 60% of the Infected
	# EXPOSED
	E0 = 0.8 * I0  # Hypothesis: Exposed correspond to 80% of the Infected   
	
	
	#E0, I0, R0 = 5, 10, 0
	
	return dfMS, startdate, state_name, population, sub_report, E0, I0, R0, M0, r0



def get_input_data():
	"""
	Provides the inputs for the simulation
	:return: tuples for the demograph_parameters, covid_parameters 
	and model_parameters
	
	
	
	Degrees of isolation (i)
	no isolation, vertical, horizontal
		
	IC_Analysis
	1: Confidence Interval; 2: Single Run; 3: Sensitivity Analysis
	
	"""
		
	
	IC_analysis = 2  # 1 # 2 # 3 
	# 1: CONFIDENCE INTERVAL (r0, gamma and alpha, lognormal distribution)
	# 2: SINGLE RUN
	# 3: SENSITIVITY ANALYSIS (r0)
	
	runs = 300 # 1_000 # number of runs for Confidence Interval analysis
	
	dfMS, startdate, state_name, sub_report, r0_fit = [], [], [], [], []
	fit_analysis = 0 # 0 #
	if fit_analysis == 1:
		 [dfMS, startdate, state_name, population_fit, sub_report,
		 E0_fit, I0_fit, R0_fit, M0_fit, r0_fit] = fit_curve()

	# CONFIDENCE INTERVAL AND SENSITIVITY ANALYSIS
	# 95% Confidence interval bounds or range for sensitivity analysis
	# Basic Reproduction Number # ErreZero
	basic_reproduction_number = (2.4, 3.3)     # 1.4 / 2.2 / 3.9  		
	if fit_analysis == 1:
		 basic_reproduction_number = r0_fit

	# SINGLE RUN AND SENSITIVITY ANALYSIS
	# Incubation Period (in days)
	incubation_period = 5.2             # (4.1, 7.0) #	
	# Infectivity Period (in days)      # tempo_de_infecciosidade
	infectivity_period = 10.0
	pI = 0.1425 #962/7600 #  Proportion of persons aged 60+ in Brazil,
	# 2020 forecast, Source: IBGE's app
	contact_matrix = np.array([[16.24264923,0.34732121],[ 5.14821886 ,0.72978211]])
	M_matrix = np.zeros((len(contact_matrix),len(contact_matrix[0])))
	Population_proportion = np.array([1-pI,pI])
	for i in range(len(contact_matrix)):
		for j in range(len(contact_matrix[0])):
			M_matrix[i][j] = contact_matrix[i][j]*Population_proportion[i]/Population_proportion[j]
	Normalization_constant,_ = np.linalg.eig(M_matrix)
	Normalization_constant = max(Normalization_constant.real)
	if IC_analysis == 1: # CONFIDENCE INTERVAL for a lognormal distribution
		
		# PARAMETERS ARE ARRAYS
		
		# 95% Confidence interval bounds for Covid parameters
		# Incubation Period (in days)
		incubation_period = (1.9, 2.1) # (4.1, 7.0)
	
		# Infectivity Period (in days)   # tempo_de_infecciosidade
		#infectivity_period = (7.0, 12.0) #	3 days or 7 days
		infectivity_period = (2.9, 3.1) #	3 days or 7 days
		# Woelfel et al 22 (eCDC: 7-12 days @ 19/4/20, 
		# https://www.ecdc.europa.eu/en/covid-19/questions-answers)
		
		# Computes mean and std for a lognormal distribution
		alpha_inv_params = make_lognormal_params_95_ci(*incubation_period)
		gamma_inv_params = make_lognormal_params_95_ci(*infectivity_period)
		R0__params = make_lognormal_params_95_ci(*basic_reproduction_number)
	
		# samples for a lognormal distribution (Monte Carlo Method)
		# alpha
		incubation_rate = 1/npr.lognormal(*map(np.log, alpha_inv_params),runs)
		# gamma
		infectivity_rate = 1/npr.lognormal(*map(np.log, gamma_inv_params),runs)
		# beta = r0 * gamma
		contamination_rate = npr.lognormal(*map(np.log, R0__params), runs) * infectivity_rate
		
		
	elif IC_analysis == 2: # SINGLE RUN
	
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
		# beta = r0 * gamma
		contamination_rate = basic_reproduction_number * infectivity_rate
		
	else: # r0 Sensitivity analysis
		
		# PARAMETERS ARE ARRAYS
		# Calculate array for r0 to a sensitivity analysis
		R0_array = np.arange(*basic_reproduction_number, 0.1) # step 0.1 for r0
		# alpha with length
		incubation_rate = np.repeat(1 / incubation_period, len(R0_array))
		# gamma
		infectivity_rate = np.repeat(1 / infectivity_period, len(R0_array)) 
		# beta = r0 * gamma
		contamination_rate = R0_array * infectivity_rate 
	contamination_rate = contamination_rate/Normalization_constant
	print(contamination_rate)

	covid_parameters = namedtuple('Covid_Parameters',
								['alpha',                             # incubation rate
								'beta',                              # contamination rate
								'gamma',                             # infectivity rate
								'mortality_rate_elderly',            # taxa_mortalidade_i
								'mortality_rate_young',              # taxa_mortalidade_j
								'los_ward',                          # los_leito
								'los_icu',                           # los_uti
								'internation_rate_ward_elderly',     # tax_int_i
								'internation_rate_ward_young',       # tax_int_j
								'internation_rate_icu_elderly',      # tax_uti_i
								'internation_rate_icu_young'         # tax_uti_j
								])
	
	covid_parameters = covid_parameters(
		# Incubation rate (1/day)
		alpha = incubation_rate,
		# Contamination rate (1/day)
		beta = contamination_rate,
		# Infectivity rate (1/day)
		gamma = infectivity_rate,
		# Mortality Rates, Source: Verity, et al,
		# adjusted with population distribution IBGE 2020
		mortality_rate_elderly = 0.0079,#0.03495,         # old ones: 60+ years
		mortality_rate_young = 0.0079,#0.00127,           # young ones: 0-59 years
		# Length of Stay (in days), Source: Wuhan
		los_ward = 8.9,                         # regular
		los_icu = 8,                            # UTI
		# Internation Rate by type and age, 
		# Source for hospitalization verity et al;
		# Proportion those need ICU:
		# Severe Outcomes Among Patients with Coronavirus Disease 2019 CDC
		internation_rate_ward_elderly = 0.1026,  # regular for old ones: 60+ years
		internation_rate_ward_young = 0.0209,    # regular for young ones: 0-59 years
		internation_rate_icu_elderly = 0.0395,   # UTI for old ones: 60+ years
		internation_rate_icu_young = 0.0052      # UTI for young ones: 0-59 years
	)
	
	model_parameters = namedtuple('Model_Parameters',
								['contact_reduction_elderly',     	# omega_i
								'contact_reduction_young',       	# omega_j
								'lotation',                      	# lotacao
								'init_exposed_elderly',          	# Ei0
								'init_exposed_young',            	# Ej0
								'init_infected_elderly',         	# Ii0
								'init_infected_young',           	# Ij0
								'init_removed_elderly',          	# Ri0
								'init_removed_young',            	# Rj0
								'init_hospitalized_ward_elderly',	# Hi0
								'init_hospitalized_ward_young',     # Hj0
								'init_hospitalized_icu_elderly',    # Ui0
								'init_hospitalized_icu_young',      # Uj0
								'init_deceased_elderly',         	# Mi0
								'init_deceased_young',           	# Mj0
								't_max',                         	# t_max
								'population',                		# N
								'population_rate_elderly',   		# percentual_pop_idosa
								'bed_ward',                  		# capacidade_leitos
								'bed_icu',                    		# capacidade_UTIs
								'IC_analysis',						# Type of analysis
								'dfMS',                    		    # dataframe_Min_Saude_data
								'startdate',                    	# start date of the fit and simulation
								'state_name',                    	# state simulated
								'r0_fit',                           # range of r0
								'sub_report',                       # sub_report factor
								'contact_matrix'					# contact matrix
								])
	
	N = 211_755_692 # 211 millions, 2020 forecast, Source: IBGE's app
	#7_600_000_000, #
	
	# INITIAL CONDITIONS
	E0 = 0 #64 #260_000 #basic_reproduction_number * I0
	I0 = 1 #100#304_000 #  (a total of 20943 cases in the last 10 days 
	# within a total of 38654 cumulative confirmed cases in 
	# 19/04/2020 17:00 GMT-3 - source https://covid.saude.gov.br/)
	R0 = 0 #407#472_000 # 
	
	if fit_analysis == 1:
		 E0, I0, R0, M0, N = E0_fit, I0_fit, R0_fit, M0_fit, population_fit
	
	
	Ei0 = E0 * pI
	Ii0 = I0 * pI
	Ri0 = R0 * pI
	
	Ej0 = E0 * (1 - pI)
	Ij0 = I0 * (1 - pI)
	Rj0 = R0 * (1 - pI)
	
	# Leitos normais demandados
	Hi0 = Ii0 * covid_parameters.internation_rate_ward_elderly
	Hj0 = Ij0 * covid_parameters.internation_rate_ward_young
	# Leitos UTIs demandados
	Ui0 = Ii0 * covid_parameters.internation_rate_icu_elderly
	Uj0 = Ij0 * covid_parameters.internation_rate_icu_young
	
	# Obitos
	#M_0 = 3_000
	Mi0 = Ri0 * covid_parameters.mortality_rate_elderly
	Mj0 = Rj0 * covid_parameters.mortality_rate_young
	
	if fit_analysis == 1:
		 Mi0 = M0 * pI
		 Mj0 = M0 * (1 - pI)    
	
	
	# Perhaps initial conditions will change to match deaths at the present date
	
	model_parameters = model_parameters(
		# Social contact reduction factor (without, vertical, horizontal) isolation
		contact_reduction_elderly = (1., .4, .4), # young ones: 0-59 years
		contact_reduction_young = (1., 1., .6), # old ones: 60+ years	
		# Scenaries for health system colapse
		lotation = (0.3, 0.5, 0.8, 1),        	# 30, 50, 80, 100% capacity
		init_exposed_elderly = Ei0,    			# initial exposed population old ones: 60+ years
		init_exposed_young = Ej0,       		# initial exposed population young ones: 0-59 years
		init_infected_elderly = Ii0,    		# initial infected population old ones: 60+ years 
		init_infected_young = Ij0,      		# initial infected population young ones: 0-59 years
		init_removed_elderly = Ri0,        		# initial removed population old ones: 60+ years
		init_removed_young = Rj0,           	# initial removed population young ones: 0-59 years
		init_hospitalized_ward_elderly = Hi0,   # initial ward hospitalized old ones: 60+ years
		init_hospitalized_ward_young = Hj0,     # initial ward hospitalized young ones: 0-59 years
		init_hospitalized_icu_elderly = Ui0,   	# initial icu hospitalized old ones: 60+ years
		init_hospitalized_icu_young = Uj0,     	# initial icu hospitalized young ones: 0-59 years
		init_deceased_elderly = Mi0,        	# initial deceased population old ones: 60+ years
		init_deceased_young = Mj0,           	# initial deceased population young ones: 0-59 years
		t_max = 2 * 365, #	        # number of days to run
		# Brazilian Population
		population = N,             
		# Brazilian old people proportion (age: 60+), 2020 forecast
		population_rate_elderly = pI,
		# Proportion of persons aged 60+ in Brazil, Source: IBGE's app
		# Brazilian bed places , Source: CNES, 05/05/2020
		# http://cnes2.datasus.gov.br/Mod_Ind_Tipo_Leito.asp?VEstado=00
		bed_ward = 298_855,                     # bed ward
		bed_icu = 32_380,                       # bed ICUs
		IC_analysis = IC_analysis,			    # flag for run type
		# 1: confidence interval, 2: single run, 3: r0 sensitivity analysis
		dfMS = dfMS, #dataframe_Min_Saude_data
		startdate = startdate, # start date of the fit and simulation
		state_name = state_name, # state simulated
		r0_fit = r0_fit,                        # range of r0 fitted
		sub_report = sub_report,                 # sub_report factor
		contact_matrix = contact_matrix
	)
	
	return covid_parameters, model_parameters
