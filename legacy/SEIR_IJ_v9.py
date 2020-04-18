import numpy as np
import pandas as pd

def get_input_data():
	'''Provides the inputs for the simulation'''
	from collections import namedtuple

	Demograph_Parameters = namedtuple('Demograph_Parameters',
                                   ['population', # N
                                   'population_rate_elderly', # percentual_pop_idosa
                                   'bed_ward', # capacidade_leitos
                                   'bed_icu' # capacidade_UTIs
                                   ]
                                   )
	demograph_parameters = Demograph_Parameters(
	# Brazilian Population
	population = 210000000, # 210 millions, 2020 forecast, Source: IBGE's app
	# Brazilian old people proportion (age: 55+)
	population_rate_elderly = 0.2, # 20%, 2020 forecast, Source: IBGE's app
	# Brazilian places
	bed_ward = 295083, # regular bed, Source: CNES, 13/04/2020
	bed_icu = 32329, # bed UTIs, Source: CNES, 13/04/2020
    )

	# Basic Reproduction Number # ErreZero
	basic_reproduction_number = 2.3 #0.8#1.3#1.8#2.3#2.8#
	# Infectivity Period (in days) # tempo_de_infecciosidade
	infectivity_period = 10 #5#7.5#10#12.5#15
	# Incubation Period (in days)
	incubation_period = 5 #1#2.5#5#7.5#10#12.5#15

	# Variaveis de apoio
	incubation_rate = 1 / incubation_period
	infectiviy_rate = 1 / infectivity_period
	contamination_rate = basic_reproduction_number / infectivity_period

	Covid_Parameters = namedtuple('Covid_Parameters',
                                ['alpha', # incubation rate
                                 'beta', # contamination rate
                                 'gamma', # infectivity rate
                                 'mortality_rate_elderly', # taxa_mortalidade_i
                                 'mortality_rate_young', # taxa_mortalidade_j
                                 'los_ward', # los_leito
                                 'los_icu', # los_uti
								 'delay_ward', # los_leito
                                 'delay_icu', # los_uti
                                 'internation_rate_ward_elderly', # tax_int_i
                                 'internation_rate_icu_elderly', # tax_uti_i
                                 'internation_rate_ward_young', # tax_int_j
                                 'internation_rate_icu_young' # tax_uti_j
                                 ]
                                )
	covid_parameters = Covid_Parameters(
	# Incubation rate (1/day)
	alpha = incubation_rate,
	# Contamination rate (1/day)
	beta = contamination_rate,
	# Infectivity rate (1/day)
	gamma = infectiviy_rate,
	# Mortality Rates, Source: min CDC
	mortality_rate_elderly = 0.034, # old ones: 55+ years
	mortality_rate_young = 0.002, # young ones: 0-54 years
	# Length of Stay (in days)
	los_ward = 8.9, # regular, Source: Wuhan
	los_icu = 8, # UTI, Source: Wuhan
	# Delay (in days)
	delay_ward = 2, #
	delay_icu = 3, #
	# Internation Rate by type and age, Source: min CDC
	internation_rate_ward_elderly = 0.263, # regular for old ones: 55+ years
	internation_rate_icu_elderly = 0.071, # UTI for old ones: 55+ years
	internation_rate_ward_young = 0.154, # regular for young ones: 0-54 years
 	internation_rate_icu_young = 0.03 # UTI for young ones: 0-54 years
    )

	Model_Parameters = namedtuple('Model_Parameters',
									['contact_reduction_elderly', # omega_i
									'contact_reduction_young', # omega_j
									'lotation', # lotacao
									'init_exposed_elderly', # Ei0
									'init_exposed_young', # Ej0
									'init_infected_elderly', # Ii0
									'init_infected_young', # Ij0
									'init_removed_elderly', # Ri0
									'init_removed_young', # Rj0
									't_max' # t_max
                                   ]
                                   )
	model_parameters = Model_Parameters(
	# Social contact reduction factor
	contact_reduction_elderly = 1.0, #0.2#0.4#0.6#0.8#1.0# # old ones: 55+ years
	contact_reduction_young = 1.0,  #0.2#0.4#0.6#0.8#1.0# # young ones: 0-54 years
	# Scenaries for health system colapse
	lotation = (0.3,0.5,0.8,1), # 30, 50, 80, 100% capacity
	init_exposed_elderly = 20000, # initial exposed population old ones: 55+ years
	init_exposed_young = 20000, # initial exposed population young ones: 0-54 years
	init_infected_elderly = 5520, # initial infected population old ones: 55+ years
	init_infected_young = 10000, # initial infected population young ones: 0-54 years
	init_removed_elderly = 3000, # initial removed population old ones: 55+ years
	init_removed_young = 6240, # initial removed population young ones: 0-54 years
	t_max = 2*365 	# number of days to run
	)

	return (demograph_parameters, covid_parameters, model_parameters)

# The SEIR model differential equations.
def Euler(SEIR_0 , t, params):
	'''Computes the Derivatives by Semi Implicit Euler Method'''
	N, alpha, beta, gamma, omega_i, omega_j = params
	# Vetor variaveis incognitas
	Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0 = SEIR_0
	Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj = [Si0], [Sj0], [Ei0], [Ej0], [Ii0], [Ij0], [Ri0], [Rj0]
	dt = t[1] - t[0]
	for _ in t[1:]:
		dSidt = - beta * omega_i * Si[-1] * (Ii[-1] + Ij[-1]) / N
		dSjdt = - beta * omega_j * Sj[-1] * (Ii[-1] + Ij[-1]) / N
		dEidt = - dSidt - alpha * Ei[-1]
		dEjdt = - dSjdt - alpha * Ej[-1]
		dIidt = alpha * Ei[-1] - gamma * Ii[-1]
		dIjdt = alpha * Ej[-1] - gamma * Ij[-1]
		dRidt = gamma * Ii[-1]
		dRjdt = gamma * Ij[-1]
		next_Si = Si[-1] + dSidt * dt
		next_Sj = Sj[-1] + dSjdt * dt
		next_Ei = Ei[-1] + dEidt * dt
		next_Ej = Ej[-1] + dEjdt * dt
		next_Ii = Ii[-1] + dIidt * dt
		next_Ij = Ij[-1] + dIjdt * dt
		next_Ri = Ri[-1] + dRidt * dt
		next_Rj = Rj[-1] + dRjdt * dt
		Si.append(next_Si)
		Sj.append(next_Sj)
		Ei.append(next_Ei)
		Ej.append(next_Ej)
		Ii.append(next_Ii)
		Ij.append(next_Ij)
		Ri.append(next_Ri)
		Rj.append(next_Rj)
	return np.stack([Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj]).T


def run_SEIR_ODE_model(demograph_parameters, covid_parameters, model_parameters) -> pd.DataFrame:
	'''Runs the simulation'''
	#from scipy.integrate import odeint

	dp = demograph_parameters
	cp = covid_parameters
	mp = model_parameters

	N = dp.population

	Ei0 = mp.init_exposed_elderly     # Ee0
	Ej0 = mp.init_exposed_young       # Ey0
	Ii0 = mp.init_infected_elderly    # Ie0
	Ij0 = mp.init_infected_young      # Iy0
	Ri0 = mp.init_removed_elderly     # Re0
	Rj0 = mp.init_removed_young       # Ry0
	t_max = mp.t_max

	# Variaveis apresentadas em base diaria
	# A grid of time points (in days)
	t = range(t_max)
	#dt = .1
	#t = np.linspace(0, t_max, int(t_max/dt) + 1)

	# CONDICOES INICIAIS
	# Suscetiveis
	Si0 = N * dp.population_rate_elderly - Ii0 - Ri0 - Ei0  # Suscetiveis idosos
	Sj0 = N * (1 - dp.population_rate_elderly) - Ij0 - Rj0 - Ej0 # Suscetiveis jovens

	# Initial conditions vector
	SEIR_0 = Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0

	alpha = cp.alpha
	beta = cp.beta
	gamma = cp.gamma

	omega_i = mp.contact_reduction_elderly
	omega_j = mp.contact_reduction_young

	# PARAMETROS PARA CALCULAR DERIVADAS
	args = (N, alpha, beta, gamma, omega_i, omega_j)

	# Integrate the SIR equations over the time grid, t
	#ret = odeint(deriv, y0, t, args)
	# Integrate the SEIR equations over the time grid, t
	ret = Euler(SEIR_0, t, args)
	# Update the variables
	Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj = ret.T
	SEIR = Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj

	# POST PROCESS to obtain the hospital demand (ward and ICUs) and deaths
	HUM = HUM_analysis(SEIR, t, cp)

	Hi, Hj, Ui, Uj, Mi, Mj = HUM

	print(Hi[:10])
	print(max(Mi))
	print(max(Mj))

	#health_system_colapse_identifier(Hi, Hj, Ui, Uj, dp, mp)

	return pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej, 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
	'Hi': Hi, 'Hj': Hj, 'Ui': Ui, 'Uj': Uj, 'Mi': Mi, 'Mj': Mj}, index=t)


def auxiliar_names(covid_parameters, model_parameters):
	'''Provides filename and legend for plots from the sensitivity parameter analysis'''

	alpha = covid_parameters.alpha
	beta = covid_parameters.beta
	gamma = covid_parameters.gamma

	omega_i = model_parameters.contact_reduction_elderly
	omega_j = model_parameters.contact_reduction_young


	# AUTOMATIZACAO PARA ANALISE SENSIBILIDADE PARAMETROS
	psel = 3 #0#1#2#3#
	pvalue = (alpha, beta, gamma, omega_i)
	pname = 'alpha','beta','gamma','omegaI' # parametros
	pInt = ("%.1f" % pvalue[psel])[0] # parte inteira do parametro (1 caractere)
	pDec = ("%.1f" % pvalue[psel])[2] # parte decimal do parametro (1 caractere)
	filename = pname[psel] + pInt + '_' + pDec
	if psel == 3:
		filename = filename + '__omegaJ' + ("%.1f" % omega_j)[0] + '_' + ("%.1f" % omega_j)[2]
	leg = (
	f'SEIR($alpha$={"%0.1f" % alpha})',
	f'SEIR($\\beta$={"%0.1f" % beta})',
	f'SEIR($\\gamma$={"%0.1f" % gamma})',
	f'SEIR($\\omega_I$={"%0.1f" % omega_i}, $\\omega_J$={"%0.1f" % omega_j})'
	)
	legenda = leg[psel]
	return filename, legenda


def HUM_analysis(SEIR,t,covid_parameters):
	'''Provides H (ward) U (ICU) M (deaths) variables, in a post-processment'''
	Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj = SEIR

	cp = covid_parameters

	alpha = cp.alpha
	gamma = cp.gamma

	taxa_mortalidade_i = cp.mortality_rate_elderly
	taxa_mortalidade_j = cp.mortality_rate_young

	los_leito = cp.los_ward
	los_uti = cp.los_icu

	delay_leito = cp.delay_ward
	delay_uti = cp.delay_icu

	tax_int_i = cp.internation_rate_ward_elderly
	tax_int_j = cp.internation_rate_ward_young

	tax_uti_i = cp.internation_rate_icu_elderly
	tax_uti_j = cp.internation_rate_icu_young

	# Leitos normais demandados
	Hi0 = Ii[0] * tax_int_i
	Hj0 = Ij[0] * tax_int_j
	# Leitos UTIs demandados
	Ui0 = Ii[0] * tax_uti_i
	Uj0 = Ij[0] * tax_uti_j
	# Obitos
	Mi0 = Ri[0] * taxa_mortalidade_i
	Mj0 = Rj[0] * taxa_mortalidade_j



	#print(Ei[:10])

	Hi, Hj, Ui, Uj, Mi, Mj = [Hi0], [Hj0], [Ui0], [Uj0], [Mi0], [Mj0]
	# Ei, Ej, Ii, Ij = [Ei], [Ej], [Ii], [Ij]
	dt = t[1] - t[0]
	for i in t[1:]:
		# Leitos Normais demandados
		dHidt = tax_int_i * alpha * Ei[i-1] - Hi[i-1] / (los_leito + delay_leito)
		dHjdt = tax_int_j * alpha * Ej[i-1] - Hj[i-1] / (los_leito + delay_leito)
		# Leitos UTIs demandados
		dUidt = tax_uti_i * alpha * Ei[i-1] - Ui[i-1] / (los_uti + delay_uti)
		dUjdt = tax_uti_j * alpha * Ej[i-1] - Uj[i-1] / (los_uti + delay_uti)
		# Removidos
		dRidt = gamma * Ii[i-1]
		dRjdt = gamma * Ij[i-1]
		# Obitos
		dMidt = taxa_mortalidade_i * dRidt
		dMjdt = taxa_mortalidade_j * dRjdt
		next_Hi = Hi[i-1] + dHidt * dt
		next_Hj = Hj[i-1] + dHjdt * dt
		next_Ui = Ui[i-1] + dUidt * dt
		next_Uj = Uj[i-1] + dUjdt * dt
		next_Mi = dMidt * dt
		next_Mj = dMjdt * dt
		Hi.append(next_Hi)
		Hj.append(next_Hj)
		Ui.append(next_Ui)
		Uj.append(next_Uj)
		Mi.append(next_Mi)
		Mj.append(next_Mj)
	ret = np.stack([Hi, Hj, Ui, Uj, Mi, Mj]).T
	Hi, Hj, Ui, Uj, Mi, Mj = ret.T
	return Hi, Hj, Ui, Uj, Mi, Mj


def health_system_colapse_identifier(Hi, Hj, Ui, Uj, dp, mp):
	from datetime import datetime
	'''Performs a post_processing analysis,
    forecast the date to a load of the health system for 30,50,80,100 %
    considers the inital date as today.'''
	H = Hi + Hj
	U = Ui + Uj

	capacidade_leitos = dp.bed_ward
	capacidade_UTIs = dp.bed_icu

	lotacao = mp.lotation

	t_max = mp.t_max

	# IDENTIFICADOR DE DIAS DE COLAPSOS
	# Dia em que colapsa o sistema de saude: 30, 50, 80, 100% capacidade
	dia_colapso_leitos_30  = np.min(np.where(H > capacidade_leitos*lotacao[0]))
	dia_colapso_leitos_50  = np.min(np.where(H > capacidade_leitos*lotacao[1]))
	dia_colapso_leitos_80  = np.min(np.where(H > capacidade_leitos*lotacao[2]))
	dia_colapso_leitos_100 = np.min(np.where(H > capacidade_leitos*lotacao[3]))
	dia_colapso_leitos = (dia_colapso_leitos_30, dia_colapso_leitos_50,
		dia_colapso_leitos_80, dia_colapso_leitos_100)
	print(dia_colapso_leitos)

	dia_colapso_UTIs_30  = np.min(np.where(U > capacidade_UTIs*lotacao[0]))
	dia_colapso_UTIs_50  = np.min(np.where(U > capacidade_UTIs*lotacao[1]))
	dia_colapso_UTIs_80  = np.min(np.where(U > capacidade_UTIs*lotacao[2]))
	dia_colapso_UTIs_100 = np.min(np.where(U > capacidade_UTIs*lotacao[3]))
	dia_colapso_UTIs = (dia_colapso_UTIs_30, dia_colapso_UTIs_50,
		dia_colapso_UTIs_80,dia_colapso_UTIs_100)
	print(dia_colapso_UTIs)

	# TimeSeries
	datelist = [d.strftime('%d/%m/%Y')
            for d in pd.date_range(datetime.today(), periods = t_max)]

	print('Dia em que colapsa o sistema de saude (leitos comuns): 30, 50, 80, 100% capacidade')

	print(datelist[dia_colapso_leitos[0]])
	print(datelist[dia_colapso_leitos[1]])
	print(datelist[dia_colapso_leitos[2]])
	print(datelist[dia_colapso_leitos[3]])

	print('Dia em que colapsa o sistema de saude (UTI): 30, 50, 80, 100% capacidade')

	print(datelist[dia_colapso_UTIs[0]])
	print(datelist[dia_colapso_UTIs[1]])
	print(datelist[dia_colapso_UTIs[2]])
	print(datelist[dia_colapso_UTIs[3]])


def plots(filename, legenda, results, demograph_parameters, model_parameters):
	import matplotlib.pyplot as plt
	'''Makes two plots? 0) SEIR curve, 1) Hospital Demand'''

	capacidade_leitos = demograph_parameters.bed_ward
	capacidade_UTIs = demograph_parameters.bed_icu

	lotacao = model_parameters.lotation

	t_max = model_parameters.t_max

	#plot
	tamfig = (8,6)
	fsLabelTitle = 15 # Font Size: Label and Title
	fsPlotLegend = 12 # Font Size: Plot and Legend

	# SEIR
	plt.figure(0)
	plt.style.use('ggplot')
	(results
	# .div(1_000_000)
	[['Si', 'Sj', 'Ei', 'Ej', 'Ii', 'Ij', 'Ri', 'Rj']]
	.plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False))
	plt.title(f'Numero de Pessoas Atingidas com modelo:\n' + legenda, fontsize=fsLabelTitle)
	plt.legend(['Suscetiveis Idosas', 'Suscetiveis Jovens', 'Expostas Idosas', 'Expostas Jovens',
		'Infectadas Idosas', 'Infectadas Jovens', 'Removidas Idosas', 'Removidas Jovens'],
		fontsize=fsPlotLegend)
	plt.xlabel('Dias', fontsize=fsLabelTitle)
	plt.ylabel('Pessoas', fontsize=fsLabelTitle)
	plt.savefig("SEIR_" + filename + ".png")

	# Demanda Hospitalar
	plt.figure(1)
	plt.style.use('ggplot')
	(results
	# .div(1_000_000)
	[['Hi', 'Hj', 'Ui', 'Uj']]
	.plot(figsize = tamfig, fontsize = fsPlotLegend, logy=False))
	#plt.hlines(capacidade_leitos*lotacao[0],1,t_max) #30%
	#plt.hlines(capacidade_leitos*lotacao[1],1,t_max) #50%
	#plt.hlines(capacidade_leitos*lotacao[2],1,t_max) #80%
	plt.hlines(capacidade_leitos*lotacao[3],1, t_max, label='100% Leitos', colors='y', linestyles='dotted') #100%
	#plt.hlines(capacidade_UTIs*lotacao[0],1,t_max) #30%
	#plt.hlines(capacidade_UTIs*lotacao[1],1,t_max) #50%
	#plt.hlines(capacidade_UTIs*lotacao[2],1,t_max) #80%
	plt.hlines(capacidade_UTIs*lotacao[3],1, t_max, label='100% UTI', colors='g', linestyles='dashed') #100%
	plt.title(f'Demanda diaria de leitos:\n' + legenda, fontsize=fsLabelTitle)
	plt.legend(['Leito normal idosos', 'Leito normal jovens', 'UTI idosos', 'UTI jovens'
		,'100% Leitos','100% UTIs'], fontsize = fsPlotLegend)
	plt.xlabel('Dias', fontsize = fsLabelTitle)
	plt.ylabel('Leitos', fontsize = fsLabelTitle)
	plt.savefig("HU_" + filename + ".png")

	#tit1 = " Cenario 1 - Sem Isolamento" # Titulo Graficos
	#tit2 = " Cenario 2 - Isolamento Vertical" # Titulo Graficos

	# "Obitos -" + tit1
	# "Pico Idosos"
	# "Pico Jovens"

	# "Pico Leitos Comuns"
	# "Pico UTI"
	# "Colapso do"
	# "Sistema de Saúde"
	# "Capacidade de Leitos"


if __name__ == '__main__':


	demograph_parameters, covid_parameters, model_parameters = get_input_data()

	results = run_SEIR_ODE_model(demograph_parameters, covid_parameters, model_parameters)

	filename, legenda = auxiliar_names(covid_parameters, model_parameters)

	results.to_csv(filename + '.csv', index=False)


	plots(filename, legenda, results, demograph_parameters, model_parameters)