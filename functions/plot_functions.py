import matplotlib.pyplot as plt
from .utils import *
from datetime import datetime

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
    basic_reproduction_number = beta / gamma
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
    
    time = datetime.today()
    time = time.strftime('%Y%m%d%H%M')
    
    filename = (time
    + '_r' + ("%.1f" % r0)[0] + '_' + ("%.1f" % r0)[2]
    + '__g' + ("%.1f" % gamma)[0] + '_' + ("%.1f" % gamma)[2]
    )

    return filename


def plots(results, demograph_parameters, model_parameters, plot_dir):

    """
    Makes two plots: 0) SEIR curve, 1) Hospital Demand
    """
    
    N = demograph_parameters.population
    capacidade_leitos = demograph_parameters.bed_ward
    capacidade_UTIs = demograph_parameters.bed_icu

    lotacao = model_parameters.lotation
    omegas_i = model_parameters.contact_reduction_elderly
    omegas_j = model_parameters.contact_reduction_young
    t_max = model_parameters.t_max

    # plot
    tamfig = (8,6)
    fsLabelTitle = 11   # Font Size: Label and Title
    fsPlotLegend = 10   # Font Size: Plot and Legend

    def format_float(float_number, precision=2):
        return str(round(float_number,2)).replace(".", "")
    
    for omega_i in omegas_i:
        for omega_j in omegas_j:
            
            plt.figure(0)
            plt.style.use('ggplot')
        
            (results.query('omega_i == @omega_i & omega_j == @omega_j')
                # .div(1_000_000)
                [['Si', 'Sj', 'Ei', 'Ej', 'Ii', 'Ij', 'Ri', 'Rj']]
                .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False)
            )
            
            f_omega_i = format_float(omega_i, 1)
            f_omega_j = format_float(omega_j, 1)

            plt.title(f'Pessoas atingidas com modelo SEIR: $N$={N}, ' + f'$\omega_i={omega_i}$, $\omega_j={omega_j}$', fontsize=fsLabelTitle)
            plt.legend(['Suscetiveis Idosas', 'Suscetiveis Jovens', 'Expostas Idosas', 'Expostas Jovens',
                        'Infectadas Idosas', 'Infectadas Jovens', 'Removidas Idosas', 'Removidas Jovens'],
                    fontsize=fsPlotLegend)
            plt.xlabel('Dias', fontsize=fsLabelTitle)
            plt.ylabel('Pessoas', fontsize=fsLabelTitle)

            filename = f"wi_{f_omega_i}_wj_{f_omega_j}"
            plt.savefig(os.path.join(plot_dir, "SEIR_" + filename + ".png"))

            plt.figure(1)
            plt.style.use('ggplot')

            (results.query('omega_i == @omega_i & omega_j == @omega_j')
                # .div(1_000_000)
                [['Hi', 'Hj', 'Ui', 'Uj']]
                .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False)
            )
                    
            plt.hlines(capacidade_leitos,
                        1, 
                        t_max, 
                        label=f'100% Leitos', colors='y', linestyles='dotted')

            plt.hlines(capacidade_UTIs,
                        1, 
                        t_max, 
                        label=f'100% Leitos (UTI)', colors='y', linestyles='dashed')
                
            plt.title(f'Demanda diaria de leitos: $N$={N}, ' + f'$\omega_i={omega_i}$, $\omega_j={omega_j}$', fontsize=fsLabelTitle)
            plt.legend(['Leito normal idosos', 'Leito normal jovens', 'UTI idosos',
                       'UTI jovens', '100% Leitos', '100% UTIs'], fontsize=fsPlotLegend)
            plt.xlabel('Dias', fontsize=fsLabelTitle)
            plt.ylabel('Leitos', fontsize=fsLabelTitle)
            plt.savefig(os.path.join(plot_dir, "HU_" + filename + ".png"))    
