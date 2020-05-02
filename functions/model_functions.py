import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import odeint


def run_SEIR_ODE_model(covid_parameters, model_parameters) -> pd.DataFrame:
	"""
	Runs the simulation
	"""
	cp = covid_parameters
	mp = model_parameters
	# Variaveis apresentadas em base diaria
	# A grid of time points (in days)
	t = range(mp.t_max)
		
	# CONDICOES INICIAIS
	# Initial conditions vector
	SEIRHUM_0 = initial_conditions(cp, mp)
	
	df = pd.DataFrame()
	
	for omega_i in mp.contact_reduction_elderly:
		for omega_j in mp.contact_reduction_young:
	
			# PARAMETROS PARA CALCULAR DERIVADAS
			args = args_assignment(cp, mp, omega_i, omega_j)

			# Integrate the SIR equations over the time grid, t
			ret = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
			# Update the variables
			Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = ret.T
			health_system_colapse_identifier(Hi, Hj, Ui, Uj, Mi, Mj, mp, omega_i, omega_j)
	
			df = df.append(pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej, 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
								'Hi': Hi, 'Hj': Hj, 'Ui': Ui, 'Uj': Uj, 'Mi': Mi, 'Mj': Mj}, index=t)
							.assign(omega_i=omega_i)
							.assign(omega_j=omega_j))
	
	
	return df

def initial_conditions(cp, mp):
	
	N = mp.population
	
	taxa_mortalidade_i = cp.mortality_rate_elderly
	taxa_mortalidade_j = cp.mortality_rate_young
	
	tax_int_i = cp.internation_rate_ward_elderly
	tax_int_j = cp.internation_rate_ward_young
	
	tax_uti_i = cp.internation_rate_icu_elderly
	tax_uti_j = cp.internation_rate_icu_young

	Ei0 = mp.init_exposed_elderly     # Ee0
	Ej0 = mp.init_exposed_young       # Ey0
	Ii0 = mp.init_infected_elderly    # Ie0
	Ij0 = mp.init_infected_young      # Iy0
	Ri0 = mp.init_removed_elderly     # Re0
	Rj0 = mp.init_removed_young       # Ry0
	
	# Suscetiveis
	Si0 = N * mp.population_rate_elderly - Ii0 - Ri0 - Ei0  # Suscetiveis idosos
	Sj0 = N * (1 - mp.population_rate_elderly) - Ij0 - Rj0 - Ej0 # Suscetiveis jovens
	
	# Leitos normais demandados
	Hi0 = Ii0 * tax_int_i
	Hj0 = Ij0 * tax_int_j
	# Leitos UTIs demandados
	Ui0 = Ii0 * tax_uti_i
	Uj0 = Ij0 * tax_uti_j
	# Obitos
	Mi0 = Ri0 * taxa_mortalidade_i
	Mj0 = Rj0 * taxa_mortalidade_j
	
	SEIRHUM_0 = Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, Ui0, Uj0, Mi0, Mj0
	return SEIRHUM_0


def args_assignment(cp, mp, omega_i, omega_j):
	
	N = mp.population
	
	alpha = cp.alpha
	beta = cp.beta
	gamma = cp.gamma
	
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


def health_system_colapse_identifier(Hi, Hj, Ui, Uj, Mi, Mj, mp, omega_i, omega_j):
	"""
	Performs a post_processing analysis,
	forecast the date to a load of the health system for 30,50,80,100 %
	considers the inital date as today.
	"""
	H = Hi + Hj
	U = Ui + Uj
	
	capacidade_leitos = mp.bed_ward
	capacidade_UTIs = mp.bed_icu
	
	lotacao = mp.lotation
	t_max = mp.t_max
	
	# IDENTIFICADOR DE DIAS DE COLAPSOS
	# Dia em que colapsa o sistema de saude: 30, 50, 80, 100% capacidade
	
	datelist = [d.strftime('%d/%m/%Y')
				for d in pd.date_range(datetime.today(), periods = t_max)]
	
	for lotacao_nivel in lotacao:
		
		dias_lotacao, = np.where(H > capacidade_leitos*lotacao_nivel)
	
		if dias_lotacao.size == 0:
			print(f'Não atingiu lotacao com {lotacao_nivel*100}% de capacidade dos leitos comuns (omega_i = {omega_i} e omega_j = {omega_j})')
		else:
			inicio_lotacao =  np.min(dias_lotacao)
			print(f"{inicio_lotacao} dias para atingir {lotacao_nivel*100}% de capacidade dos leitos comuns (omega_i = {omega_i} e omega_j = {omega_j})."
				f" Dia: {datelist[inicio_lotacao]}")
	
	
	for lotacao_nivel in lotacao:
	
		dias_lotacao, = np.where(H > capacidade_UTIs*lotacao_nivel)
	
		if dias_lotacao.size == 0:
			print(f'Não atingiu lotacao com {lotacao_nivel*100}% de capacidade das UTIs (omega_i = {omega_i} e omega_j = {omega_j})')
		else:
			inicio_lotacao =  np.min(dias_lotacao)
			print(f"{inicio_lotacao} dias para atingir {lotacao_nivel*100}% de capacidade da UTI (omega_i = {omega_i} e omega_j = {omega_j})."
				f" Dia: {datelist[inicio_lotacao]}")
	
	
	
	print('Idosos falecidos: %d' % max(Mi))
	print('Jovens falecidos: %d' % max(Mj))
	print('Total de obitos: %d' % (max(Mi)+ max(Mj)))
    # dia_colapso_UTIs_30  = np.min(np.where(U > capacidade_UTIs*lotacao[0]))
    # dia_colapso_UTIs_50  = np.min(np.where(U > capacidade_UTIs*lotacao[1]))
    # dia_colapso_UTIs_80  = np.min(np.where(U > capacidade_UTIs*lotacao[2]))
    # dia_colapso_UTIs_100 = np.min(np.where(U > capacidade_UTIs*lotacao[3]))
    # dia_colapso_UTIs = (dia_colapso_UTIs_30, dia_colapso_UTIs_50,
    #                     dia_colapso_UTIs_80,dia_colapso_UTIs_100)

    # print('Dias para atingir 30, 50, 80, 100% da capacidade de UTIs')
    # print(dia_colapso_UTIs)

    # TimeSeries
    # datelist = [d.strftime('%d/%m/%Y')
    #         for d in pd.date_range(datetime.today(), periods = t_max)]
    #     #for d in pd.date_range(start = '26/2/2020', periods = t_max)]

    # print('Dia em que atinge 30, 50, 80, 100% capacidade de leitos comuns')

    # print(datelist[dia_colapso_leitos[0]])
    # print(datelist[dia_colapso_leitos[1]])
    # print(datelist[dia_colapso_leitos[2]])
    # print(datelist[dia_colapso_leitos[3]])

    # print('Dia em que atinge 30, 50, 80, 100% capacidade de UTIs')

    # print(datelist[dia_colapso_UTIs[0]])
    # print(datelist[dia_colapso_UTIs[1]])
    # print(datelist[dia_colapso_UTIs[2]])
    # print(datelist[dia_colapso_UTIs[3]])

def derivSEIRHUM(SEIRHUM, t, N, alpha, beta, gamma,
				los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
				taxa_mortalidade_i, taxa_mortalidade_j,
				omega_i, omega_j):

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
	# Leitos comunss demandados
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