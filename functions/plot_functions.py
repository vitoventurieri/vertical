import matplotlib.pyplot as plt
from .utils import *


def auxiliar_names(covid_parameters, model_parameters):
    """
    Provides filename and legend for plots from the sensitivity parameter analysis
    """

    alpha = covid_parameters.alpha
    beta = covid_parameters.beta
    gamma = covid_parameters.gamma

    omega_i = model_parameters.contact_reduction_elderly
    omega_j = model_parameters.contact_reduction_young

    # AUTOMATIZACAO PARA ANALISE SENSIBILIDADE PARAMETROS
    psel = 3                                        # 0 / 1 / 2 / 3
    pvalue = (alpha, beta, gamma, omega_i)
    pname = 'alpha', 'beta', 'gamma', 'omegaI'      # parametros
    pInt = ("%.1f" % pvalue[psel])[0]               # parte inteira do parametro (1 caractere)
    pDec = ("%.1f" % pvalue[psel])[2]               # parte decimal do parametro (1 caractere)
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


def plots(filename, legenda, results, demograph_parameters, model_parameters):

    """
    Makes two plots? 0) SEIR curve, 1) Hospital Demand
    """

    capacidade_leitos = demograph_parameters.bed_ward
    capacidade_UTIs = demograph_parameters.bed_icu

    lotacao = model_parameters.lotation

    t_max = model_parameters.t_max

    # plot
    tamfig = (8,6)
    fsLabelTitle = 15   # Font Size: Label and Title
    fsPlotLegend = 12   # Font Size: Plot and Legend

    # SEIR
    plt.figure(0)
    plt.style.use('ggplot')
    (results
        # .div(1_000_000)
        [['Si', 'Sj', 'Ei', 'Ej', 'Ii', 'Ij', 'Ri', 'Rj']]
        .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False)
     )
    plt.title(f'Numero de Pessoas Atingidas com modelo:\n' + legenda, fontsize=fsLabelTitle)
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

    plt.title(f'Demanda diaria de leitos:\n' + legenda, fontsize=fsLabelTitle)
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
