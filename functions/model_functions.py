import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import odeint


def run_SEIR_ODE_model(covid_parameters, model_parameters) -> pd.DataFrame:
	"""
	Runs the simulation
    
    output:
        dataframe for SINGLE RUN
        dataframe list for SENSITIVITY ANALYSIS AND CONFIDENCE INTERVAL
	"""
	cp = covid_parameters
	mp = model_parameters
	# Variaveis apresentadas em base diaria
	# A grid of time points (in days)
	t = range(mp.t_max)
		
	# CONDICOES INICIAIS
	# Initial conditions vector
	SEIRHUM_0 = initial_conditions(mp)
	
	
    
	if mp.IC_analysis == 2:
	
		ii = 1
		df = pd.DataFrame()
		
		# 1: without; 2: vertical; 3: horizontal isolation 
		for i in range(3): # different isolation levels
			omega_i = mp.contact_reduction_elderly[i]
			omega_j = mp.contact_reduction_young[i]
		
			# Integrate the SEIR equations over the time grid, t
			# PARAMETROS PARA CALCULAR DERIVADAS
			args = args_assignment(cp, mp, omega_i, omega_j, ii)
			ret = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
			# Update the variables
			Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = ret.T
	
			df = df.append(pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej, 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
									'Hi': Hi, 'Hj': Hj, 'Ui': Ui, 'Uj': Uj, 'Mi': Mi, 'Mj': Mj}, index=t)
								.assign(omega_i = omega_i)
								.assign(omega_j = omega_j))
		DF_list = df
	
	else:
		DF_list = list() # list of data frames
	
		for ii in range(len(cp.alpha)): # sweeps the data frames list
			df = pd.DataFrame()
		
			# 1: without; 2: vertical; 3: horizontal isolation 
			for i in range(3): # different isolation levels
				omega_i = mp.contact_reduction_elderly[i]
				omega_j = mp.contact_reduction_young[i]
			
				# Integrate the SEIR equations over the time grid, t
				# PARAMETROS PARA CALCULAR DERIVADAS
				args = args_assignment(cp, mp, omega_i, omega_j, ii)
				ret = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
				# Update the variables
				Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = ret.T
			
				df = df.append(pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej, 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
										'Hi': Hi, 'Hj': Hj, 'Ui': Ui, 'Uj': Uj, 'Mi': Mi, 'Mj': Mj}, index=t)
									.assign(omega_i = omega_i)
									.assign(omega_j = omega_j))
			DF_list.append(df)
		
	return DF_list

def initial_conditions(mp):
	"""
	Assembly of the initial conditions
	input: model_parameters (namedtuple)
	output: vector SEIRHUM_0 with the variables:
	Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, Ui0, Uj0, Mi0, Mj0
	Suscetible, Exposed, Infected, Removed, Ward Bed demand, ICU bed demand, Death
	i: elderly (idoso, 60+); j: young (jovem, 0-59 years)
	"""	
	
	Ei0 = mp.init_exposed_elderly     		# Ee0
	Ej0 = mp.init_exposed_young       		# Ey0
	Ii0 = mp.init_infected_elderly    		# Ie0
	Ij0 = mp.init_infected_young      		# Iy0
	Ri0 = mp.init_removed_elderly     		# Re0
	Rj0 = mp.init_removed_young       		# Ry0
	Hi0 = mp.init_hospitalized_ward_elderly # He0
	Hj0 = mp.init_hospitalized_ward_young   # Hy0
	Ui0 = mp.init_hospitalized_icu_elderly  # Ue0
	Uj0 = mp.init_hospitalized_icu_young    # Uy0
	Mi0 = mp.init_deceased_elderly    		# Me0
	Mj0 = mp.init_deceased_young    		# My0

	# Suscetiveis
	Si0 = mp.population * mp.population_rate_elderly - Ii0 - Ri0 - Ei0  # Suscetiveis idosos
	Sj0 = mp.population * (1 - mp.population_rate_elderly) - Ij0 - Rj0 - Ej0 # Suscetiveis jovens
	
	SEIRHUM_0 = Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, Ui0, Uj0, Mi0, Mj0
	return SEIRHUM_0


def args_assignment(cp, mp, omega_i, omega_j, ii):
	"""
	Assembly of the derivative parameters
	input: covid_parameters, model_parameters
	output: vector args with the variables:

	N, alpha, beta, gamma,
	los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
	taxa_mortalidade_i, taxa_mortalidade_j,
	omega_i, omega_j
	
	Population, incubation_rate, contact_rate, infectiviy_rate,
	average_length_of_stay (regular and icu beds), internation rates (regular and icu beds, by age)
	i: elderly (idoso, 60+); j: young (jovem, 0-59 years)
	mortality_rate for young and elderly
	"""	
	
	N = mp.population
	
	if mp.IC_analysis == 2: # SINGLE RUN
		alpha = cp.alpha
		beta = cp.beta
		gamma = cp.gamma
	else: # CONFIDENCE INTERVAL OR SENSITIVITY ANALYSIS
		alpha = cp.alpha[ii]
		beta = cp.beta[ii]
		gamma = cp.gamma[ii]
	
	taxa_mortalidade_i = cp.mortality_rate_elderly
	taxa_mortalidade_j = cp.mortality_rate_young
	
	los_leito = cp.los_ward
	los_uti = cp.los_icu
	
	tax_int_i = cp.internation_rate_ward_elderly
	tax_int_j = cp.internation_rate_ward_young
	
	tax_uti_i = cp.internation_rate_icu_elderly
	tax_uti_j = cp.internation_rate_icu_young
	
	args = (N, alpha, beta, gamma,
			los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
			taxa_mortalidade_i, taxa_mortalidade_j,
			omega_i, omega_j)
	return args



def derivSEIRHUM(SEIRHUM, t, N, alpha, beta, gamma,
				los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
				taxa_mortalidade_i, taxa_mortalidade_j,
				omega_i, omega_j):
	"""
	Computes the derivatives

	input: SEIRHUM variables for elderly (i) and young (j), 
    Suscetible, Exposed, Infected, Recovered, Hospitalized, ICU, Deacesed
    time, Brazillian population,
    incubation rate, contamination rate, infectivity rate,
    LOS, hospitalization rates for wards and icu beds,
    death rates
    attenuating factors

	output: vector with the derivatives
	"""	
    
	# Vetor variaveis incognitas
	Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = SEIRHUM
	
	dSidt = - beta * omega_i * Si * (Ii + Ij) / N
	dSjdt = - beta * omega_j * Sj * (Ii + Ij) / N
	dEidt = - dSidt - alpha * Ei
	dEjdt = - dSjdt - alpha * Ej
	dIidt = alpha * Ei - gamma * Ii
	dIjdt = alpha * Ej - gamma * Ij
	dRidt = gamma * Ii
	dRjdt = gamma * Ij
	# Leitos comuns demandados
	dHidt = tax_int_i * alpha * Ei - Hi / los_leito
	dHjdt = tax_int_j * alpha * Ej - Hj / los_leito
	# Leitos UTIs demandados
	dUidt = tax_uti_i * alpha * Ei - Ui / los_uti
	dUjdt = tax_uti_j * alpha * Ej - Uj / los_uti
	# Removidos
	dRidt = gamma * Ii
	dRjdt = gamma * Ij
	# Obitos
	dMidt = taxa_mortalidade_i * dRidt
	dMjdt = taxa_mortalidade_j * dRjdt
	
	return (dSidt, dSjdt, dEidt, dEjdt, dIidt, dIjdt, dRidt, dRjdt,
			dHidt, dHjdt, dUidt, dUjdt, dMidt, dMjdt)