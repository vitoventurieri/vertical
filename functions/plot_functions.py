import matplotlib.pyplot as plt
from .utils import *


def auxiliar_names(covid_parameters, model_parameters):
	"""
	Provides filename and legend for plots from the sensitivity parameter analysis
	"""
	
	cp = covid_parameters
	mp = model_parameters
	
	alpha = cp.alpha			# incubation_rate
	beta = cp.beta				# infectiviy_rate
	gamma = cp.gamma			# contamination_rate
	
	# Variaveis de apoio
	# incubation_period = 1 / alpha
	# infectiviy_period = 1 / beta
	basic_reproduction_number = gamma / beta
	r0 = basic_reproduction_number
	
	omega_i = mp.contact_reduction_elderly
	omega_j = mp.contact_reduction_young
	
	Ei0 = mp.init_exposed_elderly				# initial exposed population old ones: 55+ years
	Ej0 = mp.init_exposed_young					# initial exposed population young ones: 0-54 years
	Ii0 = mp.init_infected_elderly				# initial infected population old ones: 55+ years
	Ij0 = mp.init_infected_young				# initial infected population young ones: 0-54 years
	Ri0 = mp.init_removed_elderly				# initial removed population old ones: 55+ years
	Rj0 = mp.init_removed_young					# initial removed population young ones: 0-54 years
	
	# AUTOMATIZACAO PARA ANALISE SENSIBILIDADE PARAMETROS
	#psel = 3                                        # 0 / 1 / 2 / 3
	#pvalue = (alpha, beta, gamma, omega_i)
	#pname = 'alpha', 'beta', 'gamma', 'wI'      # parametros
	#pInt = ("%.1f" % pvalue[psel])[0]           # parte inteira do parametro (1 caractere)
	#pDec = ("%.1f" % pvalue[psel])[2]           # parte decimal do parametro (1 caractere)
	
	filename = ('SEIR'
	+ '_r' + ("%.1f" % r0)[0] + '_' + ("%.1f" % r0)[2]
	+ '__g' + ("%.1f" % gamma)[0] + '_' + ("%.1f" % gamma)[2]
	+ '__wI' + ("%.1f" % omega_i)[0] + '_' + ("%.1f" % omega_i)[2]
	+ '__wJ' + ("%.1f" % omega_j)[0] + '_' + ("%.1f" % omega_j)[2]
	)
	
	#filename = pname[psel] + pInt + '_' + pDec
	#if psel == 3:
		#filename = filename + '__wJ' + ("%.1f" % omega_j)[0] + '_' + ("%.1f" % omega_j)[2]
	#leg = (
	#	f'SEIR($alpha$={"%0.1f" % alpha})',
	#	f'SEIR($\\beta$={"%0.1f" % beta})',
	#	f'SEIR($\\gamma$={"%0.1f" % gamma})',
	#	f'SEIR($\\omega_I$={"%0.1f" % omega_i}, $\\omega_J$={"%0.1f" % omega_j})'
	#)
	#legenda = leg[psel]
	legenda = (
		f'$\\alpha$={"%0.1f" % alpha}, $\gamma$={"%0.1f" % gamma}, $r_0$={"%0.1f" % r0}\n'
		f'Idosos: $\\omega_e$={"%0.1f" % omega_i}, $E_{{e0}}$={Ei0}, $I_{{e0}}$={Ii0}, $R_{{e0}}$={Ri0}\n'
		f'Jovens: $\\omega_y$={"%0.1f" % omega_j}, $E_{{y0}}$={Ej0}, $I_{{y0}}$={Ij0}, $R_{{y0}}$={Rj0}'
	)
	
	return filename, legenda


def plots(filename, legenda, results, demograph_parameters, model_parameters):

    """
    Makes two plots? 0) SEIR curve, 1) Hospital Demand
    """
	
    N = demograph_parameters.population
    capacidade_leitos = demograph_parameters.bed_ward
    capacidade_UTIs = demograph_parameters.bed_icu

    lotacao = model_parameters.lotation

    t_max = model_parameters.t_max

    # plot
    tamfig = (8,6)
    fsLabelTitle = 11   # Font Size: Label and Title
    fsPlotLegend = 10   # Font Size: Plot and Legend

    # SEIR
    plt.figure(0)
    plt.style.use('ggplot')
    (results
        # .div(1_000_000)
        [['Si', 'Sj', 'Ei', 'Ej', 'Ii', 'Ij', 'Ri', 'Rj']]
        .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False)
     )
    plt.title(f'Pessoas atingidas com modelo SEIR: $N$={N}, ' + legenda, fontsize=fsLabelTitle)
    plt.legend(['Suscetiveis Idosas', 'Suscetiveis Jovens', 'Expostas Idosas', 'Expostas Jovens',
                'Infectadas Idosas', 'Infectadas Jovens', 'Removidas Idosas', 'Removidas Jovens'],
               fontsize=fsPlotLegend)
    plt.xlabel('Dias', fontsize=fsLabelTitle)
    plt.ylabel('Pessoas', fontsize=fsLabelTitle)
    plt.savefig(os.path.join(get_output_dir(), "SEIR_" + filename + ".png"))

    # Demanda Hospitalar
    plt.figure(1)
    plt.style.use('ggplot')
    (results
        # .div(1_000_000)
        [['Hi', 'Hj', 'Ui', 'Uj']]
        .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False)
     )

    # plt.hlines(capacidade_leitos*lotacao[0],1,t_max) #30%
    # #plt.hlines(capacidade_leitos*lotacao[1],1,t_max) #50%
    # plt.hlines(capacidade_leitos*lotacao[2],1,t_max) #80%
    plt.hlines(capacidade_leitos*lotacao[3], 1, t_max, label='100% Leitos', colors='y', linestyles='dotted')  # 100%

    # plt.hlines(capacidade_UTIs*lotacao[0],1,t_max) #30%
    # plt.hlines(capacidade_UTIs*lotacao[1],1,t_max) #50%
    # plt.hlines(capacidade_UTIs*lotacao[2],1,t_max) #80%
    plt.hlines(capacidade_UTIs*lotacao[3], 1, t_max, label='100% UTI', colors='g', linestyles='dashed')  # 100%

    plt.title(f'Demanda diaria de leitos: $N$={N}, ' + legenda, fontsize=fsLabelTitle)
    plt.legend(['Leito normal idosos', 'Leito normal jovens', 'UTI idosos',
                'UTI jovens', '100% Leitos', '100% UTIs'], fontsize=fsPlotLegend)
    plt.xlabel('Dias', fontsize=fsLabelTitle)
    plt.ylabel('Leitos', fontsize=fsLabelTitle)
    plt.savefig(os.path.join(get_output_dir(), "HU_" + filename + ".png"))

    # tit1 = " Cenario 1 - Sem Isolamento" # Titulo Graficos
    # tit2 = " Cenario 2 - Isolamento Vertical" # Titulo Graficos

    # "Obitos -" + tit1
    # "Pico Idosos"
    # "Pico Jovens"

    # "Pico Leitos Comuns"
    # "Pico UTI"
    # "Colapso do"
    # "Sistema de Saúde"
    # "Capacidade de Leitos"
