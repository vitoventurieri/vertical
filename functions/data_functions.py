from collections import namedtuple
import numpy as np
import numpy.random as npr



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
    
	runs = 1_000 # number of runs for Confidence Interval analysis

    # CONFIDENCE INTERVAL AND SENSITIVITY ANALYSIS
	# 95% Confidence interval bounds or range for sensitivity analysis
	# Basic Reproduction Number # ErreZero
	basic_reproduction_number = (2.4, 3.3)     # 1.4 / 2.2 / 3.9  		
	

    # SINGLE RUN AND SENSITIVITY ANALYSIS
    # Incubation Period (in days)
	incubation_period = 5.2             # (4.1, 7.0) #	
	# Infectivity Period (in days)      # tempo_de_infecciosidade
	infectivity_period = 10.0
    
	if IC_analysis == 1: # CONFIDENCE INTERVAL for a lognormal distribution
        
        # PARAMETERS ARE ARRAYS
        
		# 95% Confidence interval bounds for Covid parameters
		# Incubation Period (in days)
		incubation_period = (4.1, 7.0)
	
		# Infectivity Period (in days)   # tempo_de_infecciosidade
		infectivity_period = (7.0, 12.0) #	3 days or 7 days	Woelfel et al 22 (eCDC: 7-12 days @ 19/4/20, https://www.ecdc.europa.eu/en/covid-19/questions-answers)
		
		# Computes mean and std for a lognormal distribution
		alpha_inv_params = make_lognormal_params_95_ci(*incubation_period)
		gamma_inv_params = make_lognormal_params_95_ci(*infectivity_period)
		R0__params = make_lognormal_params_95_ci(*basic_reproduction_number)
	
		# samples for a lognormal distribution (Monte Carlo Method)
        # alpha
		incubation_rate = 1 / npr.lognormal(*map(np.log, alpha_inv_params), runs)
        # gamma
		infectivity_rate = 1 / npr.lognormal(*map(np.log, gamma_inv_params), runs)
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
		mortality_rate_elderly = 0.03495,         # old ones: 60+ years
		mortality_rate_young = 0.00127,           # young ones: 0-59 years
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
								'IC_analysis'						# Type of analysis
								])
	
	pI = 0.1425 #962/7600 #  Proportion of persons aged 60+ in Brazil, 2020 forecast, Source: IBGE's app
	
	# INITIAL CONDITIONS
	E0 = 0 #64 #260_000 #basic_reproduction_number * I0
	I0 = 1 #100#304_000 #  (a total of 20943 cases in the last 10 days within a total of 38654 cumulative confirmed cases in 19/04/2020 17:00 GMT-3 - source https://covid.saude.gov.br/)
	R0 = 0 #407#472_000 # 
	
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
		t_max = 2 * 365,   # 2 * 365 #	        # number of days to run
		# Brazilian Population
		population = 211_755_692,#7_600_000_000, #             
        # 211 millions, 2020 forecast, Source: IBGE's app
		# Brazilian old people proportion (age: 60+), 2020 forecast
		population_rate_elderly = pI,
        # Proportion of persons aged 60+ in Brazil, Source: IBGE's app
		# Brazilian bed places , Source: CNES, 05/05/2020
        # http://cnes2.datasus.gov.br/Mod_Ind_Tipo_Leito.asp?VEstado=00
		bed_ward = 298_855,                  # bed ward
		bed_icu = 32_380,                    # bed ICUs
		IC_analysis = IC_analysis			# flag for run type
        # 1: confidence interval, 2: single run, 3: r0 sensitivity analysis
	)
	
	return covid_parameters, model_parameters
