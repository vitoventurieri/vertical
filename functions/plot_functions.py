# TODO
# 1) Revisar fit
# 6) Incorporar plot para efeito de sobrecarga do sistema
# 7) Incorporar cenário alternativo com isolamento horizontal,
#     talvez travar o range de omega nos plots
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from .utils import *

#def city_cases_dataset(city):
#    """
#
#    :param city:
#    :return:
#    """
#    if city == 'fortaleza':
#        dfcity = pd.read_csv(r"data/casos_CE_wcota.csv", sep=';')
#    elif city == 'sao_paulo':
#        dfcity = pd.read_csv(r"data/casos_SP_wcota.csv", sep=';')
#    elif city == 'maceio':
#        dfcity = pd.read_csv(r"data/casos_AL_wcota.csv", sep=';')
#    elif city == 'sao_luiz':
#        dfcity = pd.read_csv(r"data/casos_MA_wcota.csv", sep=';')
#        # display(df_cidade)
#    else:
#        raise("Not fitted rt for city: ", city)
#
#    # dataset source : https://github.com/wcota/covid19br/blob/master/README.md
#    # W. Cota, “Monitoring the number of COVID-19 cases and deaths in brazil at municipal and federative units level”,
#    # SciELOPreprints:362 (2020), 10.1590/scielopreprints.362 - license (CC BY-SA 4.0) acess 30/07/2020
#
#    return dfcity


def number_formatter(number, pos=None):
    """
    Convert a number into a human readable format.
    :param pos:
    :param number:

    :return:
    """
    magnitude = 0
    while abs(number) >= 1000:
        magnitude += 1
        number /= 1000.0
    return '%.1f%s' % (number, ['', 'K', 'M', 'B', 'T', 'Q'][magnitude])


def format_float(float_number, precision=2):
    """

    :param float_number:
    :param precision:
    :return:
    """
    return str(round(float_number, precision)).replace(".", "")


def auxiliar_names(covid_parameters, model_parameters):
    """
    Provides filename with timestamp and IC_analysis type
    (2: Single Run, 1: Confidence Interval, 3: Sensitivity Analysis)
    as string for files

    :param covid_parameters:
    :param model_parameters:
    :return:
    """

    time = datetime.today()
    time = time.strftime('%Y%m%d%H%M')

    if model_parameters.IC_analysis == 2:  # SINGLE RUN

        beta = covid_parameters.beta  # infectiviy_rate
        gamma = covid_parameters.gamma  # contamination_rate

        basic_reproduction_number = beta / gamma
        r0 = basic_reproduction_number

        filename = (time
                    + '_single_run'
                    + '_r' + ("%.1f" % r0)[0] + '_' + ("%.1f" % r0)[2]
                    + '__g' + ("%.1f" % gamma)[0] + '_' + ("%.1f" % gamma)[2]
                    )
    elif model_parameters.IC_analysis == 1:  # CONFIDENCE INTERVAL
        filename = (time + '_confidence_interval')
    elif model_parameters.IC_analysis == 3:  # SENSITIVITY_ANALYSIS
        filename = (time + '_sensitivity_analysis')
    else:  # Rt analysis
        filename = (time + '_Rt')
    return filename


def pos_format(title_fig, main_label_y, main_label_x):
    """
    põe labels, legenda e título, pós-formata eixo y

    :param title_fig:
    :param main_label_y:
    :param main_label_x:
    :return:
    """

    plt.title(title_fig)
    plt.legend()
    plt.xlabel(main_label_x)
    plt.ylabel(main_label_y)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(number_formatter))


# TIPOS DE PLOT
# PLOT CONFIDENCE INTERVAL
def plot_ci(Y, cor, t_space):
    """
    plota intervalo dos percentis 5 e 95%

    :param Y:
    :param cor:
    :param t_space:
    :return:
    """
    plt.fill_between(t_space,
                     np.quantile(Y, 0.05, axis=0),
                     np.quantile(Y, 0.95, axis=0).clip(Y[0, 0]),
                     color=cor, alpha=0.2)


# PLOT CONFIDENCE INTERVAL
def plot_median(Y, cor, ls, line_label, t_space):
    """
    plota mediana, i.e. percentil 50%

    :param Y:
    :param cor:
    :param ls:
    :param line_label:
    :param t_space:
    :return:
    """
    plt.plot(t_space,
             np.quantile(Y, 0.5, axis=0),  # MEDIANA
             ls,
             color=cor,
             label=line_label)


# PLOT TOTAL (i.e sem ser por idade)
def plot_total(Yi, Yj, name_variable,
               title_fig,
               fig_number,
               main_label_y,
               main_label_x,
               IC_analysis, t_space,
               ls, cor, isolation_name, i,
               plot_dir, filetype):
    """
    plota curvas totais (i.e sem ser por idade)
    com intervalo de confiança (5% percentil, mediana, 95% percentil)
    ou um único valor (SINGLE RUN)
    """
    plt.figure(fig_number)

    if IC_analysis == 2:  # SINGLE RUN
        plt.plot(t_space,
                 (Yi + Yj),
                 ls[i % 2],  # 0: dashed linestyle, 1: solid linestyle
                 color=cor[i],
                 label='Total' + isolation_name[i])
    else:  # CONFIDENCE INTERVAL
        plot_median(Yi + Yj, cor[i], ls[i % 2], 'Total' + isolation_name[i], t_space)
        plot_ci(Yi + Yj, cor[i], t_space)

    pos_format(title_fig, main_label_y, main_label_x)
    plt.savefig(os.path.join(plot_dir, name_variable + "_diff_isol" + filetype))


# PLOT POR IDADE - IDOSOS E JOVENS
def plot_byage(Yi, Yj, name_variable,
               title_fig,
               fig_number,
               main_label_y,
               main_label_x,
               IC_analysis, t_space,
               ls, cor, isolation_name, i,
               plot_dir, filetype):
    """
    plota curvas por idade
    com intervalo de confiança (5% percentil, 50% mediana, 95% percentil)
    ou um único valor (SINGLE RUN)
    """
    plt.figure(fig_number)
    if IC_analysis == 2:  # SINGLE RUN
        plt.plot(t_space,
                 Yi,
                 ls[i % 2],  # 0: dashed linestyle, 1: solid linestyle
                 color=cor[2 * i],
                 label=('Elderly' + isolation_name[i]))
        plt.plot(t_space,
                 Yj,
                 ls[(i + 1) % 2],  # 0: dashed linestyle, 1: solid linestyle
                 color=cor[1 + 2 * i],
                 label=('Young' + isolation_name[i]))
        complemento = isolation_name[i]
    else:  # CONFIDENCE INTERVAL
        plot_median(Yi, cor[2 * i], ls[i % 2], 'Elderly' + isolation_name[i], t_space)
        plot_ci(Yi, cor[2 * i], t_space)
        
        plot_median(Yj, cor[1 + 2 * i], ls[(i + 1) % 2], 'Young' + isolation_name[i], t_space)
        plot_ci(Yj, cor[1 + 2 * i], t_space)
        
        plot_median(Yj + Yi, cor[3 + 2 * i], ls[(i + 1) % 2], 'Young + elderly' + isolation_name[i], t_space)
        plot_ci(Yj + Yi, cor[3 + 2 * i], t_space)

        complemento = '_diff_isol'

        #dfcity_query = parameter_for_rt_fit_analisys(city_code)

        #exibition_date = dfcity_query.loc[1, 'date']

        #print('1º dia da simulação: ' + str(exibition_date))
        #plt.plot(dfcity_query.loc[:, 'deaths'].values)

    pos_format(title_fig, main_label_y, main_label_x)
    plt.savefig(os.path.join(plot_dir,
                             name_variable + "ij" + complemento + filetype))


def plots(results, covid_parameters, model_parameters, plot_dir):
    """
    Makes plots:
        CONFIDENCE INTERVAL AND SINGLE RUN
    
    Figures numbers:

    1,4,7) Infected by age group for each degree of isolation
    
    2,5,8) Bed demand by age group for each degree of isolation
    
    3,6,9) Deceased by age group for each degree of isolation
    
    10) Infected for different degrees of isolation
    
    100) INFECTADOS - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS

    110) HOSPITALIZADOS TOTAL - DIFERENTES ISOLAMENTOS

    101) HOSPITALIZADOS UTI - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
    
    23) Fit Infected
    
    24) Fit Deceased
    
        CONFIDENCE INTERVAL
    5% QUARTIL, MEDIANA, 95% QUARTIL
    
        SENSITIVITY ANALYSIS
    1) Infected people (r0)
    
    
    Degrees of isolation (i)
    no isolation, vertical, horizontal
        
    IC_Analysis
    1: Confidence Interval; 2: Single Run; 3: Sensitivity Analysis
    
    Age groups
    Elderly (60+); Young (0-59)
    
    Hospital Bed
    Ward; ICU
    
    """

    # capacidade_leitos = model_parameters.bed_ward
    # capacidade_UTIs = model_parameters.bed_icu
    IC_analysis = model_parameters.IC_analysis

    t_max = model_parameters.t_max
    t_space = np.arange(0, t_max)

    isolation_name = model_parameters.isolation_level  # ["without_isolation", "elderly_isolation"]

    # VARIÁVEIS PADRÃO DOS PLOTS
    # plot
    # fsLabelTitle = 11  # Font Size: Label and Title
    # fsPlotLegend = 10  # Font Size: Plot and Legend

    # leg_loc = 'upper left'  # 'best' # 'upper right' #

    # fig_style = "ggplot"
    # tamfig = (8, 6)  # Figure Size
    # mpl.rcParams['figure.figsize'] = (8, 6)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 11

    # Font Sizes
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rc('figure', figsize=(8, 6))  # Figure Size

    plt.style.use("ggplot")  # "ggplot" # "classic" #

    plt.rc("legend", loc='upper left')  # 'best' # 'upper right' #

    main_label_x = 'Days'
    main_label_y = 'Infected people'

    cor = ['b', 'r', 'k', 'g', 'y', 'm']  # Line Colors
    ls = ['-.', '-']  # Line Style

    filetype = '.pdf'  # '.pdf' # '.png' # '.svg' #

    # isolation_name = [' (unmitigated)', ' (vertical isolation)', ' (horizontal isolation)']

    print('Plotando resultados')

    # 1: without; 2: vertical; 3: horizontal isolation 
    for i in range(len(isolation_name)):  # 2: paper

        isolation_name_i = isolation_name[i]
        # omega_i = mp.contact_reduction_elderly[i]
        # omega_j = mp.contact_reduction_young[i]

        # f_omega_i = format_float(omega_i, 1)
        # f_omega_j = format_float(omega_j, 1)

        # filename = f"we_{f_omega_i}_wy_{f_omega_j}"
        filename = isolation_name_i

        # main_title = f'SEIR: $\omega_e={omega_i}$, $\omega_y={omega_j}$'
        main_title = 'SEIR:' + isolation_name_i

        if IC_analysis == 2:  # SINGLE RUN
            # Ii = results.query('omega_i == @omega_i & omega_j == @omega_j')['Ii']
            # Ij = results.query('omega_i == @omega_i & omega_j == @omega_j')['Ij']
            # Hi = results.query('omega_i == @omega_i & omega_j == @omega_j')['Hi']
            # Hj = results.query('omega_i == @omega_i & omega_j == @omega_j')['Hj']
            # Ui = results.query('omega_i == @omega_i & omega_j == @omega_j')['Ui']
            # Uj = results.query('omega_i == @omega_i & omega_j == @omega_j')['Uj']
            # Mi = results.query('omega_i == @omega_i & omega_j == @omega_j')['Mi']
            # Mj = results.query('omega_i == @omega_i & omega_j == @omega_j')['Mj']

            query_condition = 'isolamento == @isolation_name_i'
            Ii = results.query(query_condition)['Ii']
            Ij = results.query(query_condition)['Ij']
            Hi = results.query(query_condition)['Hi']
            Hj = results.query(query_condition)['Hj']
            Ui = results.query(query_condition)['Ui']
            Uj = results.query(query_condition)['Uj']
            Mi = results.query(query_condition)['Mi']
            Mj = results.query(query_condition)['Mj']

        else:  # SENSITIVITY ANALYSIS OR CONFIDENCE INTERVAL

            Si = np.zeros((len(results), t_max))
            Sj = np.zeros((len(results), t_max))
            Ei = np.zeros((len(results), t_max))
            Ej = np.zeros((len(results), t_max))
            Ii = np.zeros((len(results), t_max))
            Ij = np.zeros((len(results), t_max))
            Ri = np.zeros((len(results), t_max))
            Rj = np.zeros((len(results), t_max))
            Hi = np.zeros((len(results), t_max))
            Hj = np.zeros((len(results), t_max))
            Ui = np.zeros((len(results), t_max))
            Uj = np.zeros((len(results), t_max))
            Mi = np.zeros((len(results), t_max))
            Mj = np.zeros((len(results), t_max))

            for ii in range(len(results)):
                query_condition = 'isolamento == @isolation_name_i'

                Si[ii, ] = results[ii].query(query_condition)['Si']
                Sj[ii, ] = results[ii].query(query_condition)['Sj']
                Ei[ii, ] = results[ii].query(query_condition)['Ei']
                Ej[ii, ] = results[ii].query(query_condition)['Ej']
                Ii[ii, ] = results[ii].query(query_condition)['Ii']
                Ij[ii, ] = results[ii].query(query_condition)['Ij']
                Ri[ii, ] = results[ii].query(query_condition)['Ri']
                Rj[ii, ] = results[ii].query(query_condition)['Rj']
                Hi[ii, ] = results[ii].query(query_condition)['Hi']
                Hj[ii, ] = results[ii].query(query_condition)['Hj']
                Ui[ii, ] = results[ii].query(query_condition)['Ui']
                Uj[ii, ] = results[ii].query(query_condition)['Uj']
                Mi[ii, ] = results[ii].query(query_condition)['Mi']
                Mj[ii, ] = results[ii].query(query_condition)['Mj']

        if IC_analysis == 3:

            r0 = covid_parameters.beta / covid_parameters.gamma
            r0min = r0[0]
            r0max = r0[len(results) - 1]

            for ii in range(len(results)):
                a = (r0[ii] - r0min) / (r0max - r0min)
                plt.plot(t_space, Ii[ii, ] + Ij[ii, ],
                         color=[a, 0, 1 - a, 1],
                         label=isolation_name_i,
                         linewidth=0.5)

            plt.title(main_title)
            plt.xlabel(main_label_x)
            plt.ylabel(main_label_y)
            mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors', ['blue', 'red'])
            sm = plt.cm.ScalarMappable(cmap=mymap,
                                       norm=plt.Normalize(vmin=r0min, vmax=r0max))
            cbar = plt.colorbar(sm)
            cbar.set_label('Basic Reproduction Number', rotation=90)

            plt.savefig(os.path.join(plot_dir,
                                     "I_" + filename + 'VariosR0' + filetype))

        else:  # SINGLE RUN OR CONFIDENCE INTERVAL
            # INFECTADOS TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Ii, Ij, 'I',  # Yi,Yj,variable_name_Y
                       'Infected People by different isolation degrees',  # figure title
                       10,  # figure number
                       main_label_y,  # label_y
                       main_label_x,
                       IC_analysis, t_space,
                       ls, cor, isolation_name, i,
                       plot_dir, filetype)

            # INFECTADOS - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Ii, Ij, 'I',  # Yi,Yj,variable_name_Y
                       'Infected by age group for different isolation degrees',  # figure title
                       100,  # figure number
                       main_label_y,  # label_y
                       main_label_x,
                       IC_analysis, t_space,
                       ls, cor, isolation_name, i,
                       plot_dir, filetype)

            # HOSPITALIZADOS TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Hi, Hj, 'H',  # Yi,Yj,variable_name_Y
                       'Hospitalized People by different isolation degrees',  # figure title
                       110,  # figure number
                       'Bed Demand',  # label_y
                       main_label_x,
                       IC_analysis, t_space,
                       ls, cor, isolation_name, i,
                       plot_dir, filetype)

            # HOSPITALIZADOS UTI - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Ui, Uj, 'U',  # Yi,Yj,variable_name_Y
                       'ICU Bed Demand by age group for different isolation degrees',  # figure title
                       101,  # figure number
                       'ICU Bed Demand',  # label_y
                       main_label_x,
                       IC_analysis, t_space,
                       ls, cor, isolation_name, i,
                       plot_dir, filetype)

            # INFECTADOS - IDOSOS E JOVENS
            plot_byage(Ii, Ij, 'I',  # Yi,Yj,variable_name_Y
                       'Infected by age group' + isolation_name[i],  # figure title
                       i,  # figure number
                       main_label_y,  # label_y
                       main_label_x,
                       IC_analysis, t_space,
                       ls, cor, isolation_name, i,
                       plot_dir, filetype)

            # LEITOS DEMANDADOS - IDOSOS E JOVENS
            plt.figure(3 + i)

            if IC_analysis == 2:  # Single run
                plt.plot(t_space, Hi, ls[0], color=cor[0], label='Ward for Elderly')
                plt.plot(t_space, Hj, ls[1], color=cor[1], label='Ward for Young')
                plt.plot(t_space, Ui, ls[1], color=cor[2], label='ICU for Elderly')
                plt.plot(t_space, Uj, ls[0], color=cor[3], label='ICU for Young')

            else:  # Confidence interval
                plot_median(Hi, cor[0], ls[0], 'Ward for Elderly', t_space)
                plot_median(Hj, cor[1], ls[1], 'Ward for Young', t_space)
                plot_median(Ui, cor[2], ls[1], 'ICU for Elderly', t_space)
                plot_median(Uj, cor[3], ls[0], 'ICU for Young', t_space)

                plot_ci(Hi, cor[0], t_space)
                plot_ci(Hj, cor[1], t_space)
                plot_ci(Ui, cor[2], t_space)
                plot_ci(Uj, cor[3], t_space)
            pos_format(main_title, 'Bed Demand', main_label_x)

            plt.savefig(os.path.join(plot_dir, "HUey_" + filename + filetype))

            # OBITOS - IDOSOS E JOVENS
            plt.figure(6 + i)

            # OBITOS - IDOSOS E JOVENS
            plot_byage(Mi, Mj, 'M',  # Yi,Yj,variable_name_Y
                       'Deceased by age group' + isolation_name[i],  # figure title
                       6 + i,  # figure number
                       'Deceased people',  # label_y
                       main_label_x,
                       IC_analysis, t_space,
                       ls, cor, isolation_name, i,
                       plot_dir, filetype)

            if model_parameters.fit_analysis:
                #city_code = model_parameters.city
                #dfcity_query = parameter_for_rt_fit_analisys(city_code)
                dfcity_query = model_parameters.df_cidade
                exibition_date = dfcity_query.loc[1, 'date']
                print('1º dia da simulação: ' + str(exibition_date))
                plt.plot(dfcity_query.loc[:, 'deaths'].values)
            


            # ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            plt.savefig(os.path.join(plot_dir, "Mey_" + filename + filetype))


            

            #        plt.figure(3+i)
            #        plt.style.use('ggplot')

            #        (results.query('omega_i == @omega_i & omega_j == @omega_j')
            # .div(1_000_000)
            #            [['Hi', 'Hj', 'Ui', 'Uj']]
            #            .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False)
            #        )

            #        plt.hlines(capacidade_leitos,
            #                    1,
            #                    t_max,
            #                    label=f'100% Leitos', colors='y', linestyles='dotted')

            #        plt.hlines(capacidade_UTIs,
            #                    1,
            #                    t_max,
            #                    label=f'100% Leitos (UTI)', colors='y', linestyles='dashed')

            #        plt.title(f'Demanda diaria de leitos: $N$={N}, ' + f'$\omega_i={omega_i}$,
            #        $\omega_j={omega_j}$', fontsize=fsLabelTitle)
            #        plt.legend(['Leito normal idosos', 'Leito normal jovens', 'UTI idosos',
            #                'UTI jovens', '100% Leitos', '100% UTIs'], fontsize=fsPlotLegend)
            #        plt.xlabel('Dias', fontsize=fsLabelTitle)
            #        plt.ylabel('Leitos', fontsize=fsLabelTitle)
            #        plt.savefig(os.path.join(plot_dir, "HU_" + filename + ".png"))

            # FIT
    #         startdate = model_parameters.startdate
    #         if (IC_analysis == 4) and (not startdate == []) and (i == 0):

    #             for ifig in range(11):
    #                 plt.close(ifig)

    #             dfMS = model_parameters.dfMS
    #             state_name = model_parameters.state_name
    #             data_sim = pd.to_datetime(t_space, unit='D',
    #                                       origin=pd.Timestamp(startdate))
    #             # data_sim = pd.date_range(start=startdate, periods=t_max, freq='D')

    #             r0 = model_parameters.r0_fit
    #             sub_report = model_parameters.sub_report

    #             x_ticks = 14  # de quantos em quantos dias aparece tick no plot

    #             E0 = model_parameters.init_exposed_elderly + model_parameters.init_exposed_young
    #             I0 = model_parameters.init_infected_elderly + model_parameters.init_infected_young
    #             R0 = model_parameters.init_removed_elderly + model_parameters.init_removed_young
    #             S0 = model_parameters.population - E0 - I0 - R0

    #             # OBITOS - IDOSOS E JOVENS
    #             plt.figure(23)

    #             plot_median(Mi + Mj, cor[0], ls[0], 'Model', data_sim)


    #             plt.plot(dfMS['data'], dfMS['obitosAcumulado'],
    #                      ls[1], color=cor[1], label='Reported Data')

    #             plot_ci(Mi + Mj, cor[0], data_sim)

    #             ax = plt.gca()
    #             ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_ticks))
    #             ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #             # ax.set_xlim(pd.Timestamp('15/03/2020'), pd.Timestamp('15/04/2020'))

    #             plt.gcf().autofmt_xdate()  # Rotation

    #             # ax.set_ylim(0, 1_000)

    #             title_fit = ('r0 = (' + ("%.1f" % r0[0])
    #                          + ', ' + ("%.1f" % r0[1]) + ') @'
    #                          + pd.to_datetime(startdate, format='%Y-%m-%d')
    #                          .strftime('%d/%m/%Y')
    #                          + ' - subreport = ' + ("%d" % sub_report)
    #                          + ' - ' + state_name + '\n'
    #                          + 'SEIR(0) = '
    #                          + f"({S0:,.0f}; {E0:,.0f}; {I0:,.0f}; {R0:,.0f})")

    #             pos_format(title_fit, 'Deceased people', 'Date')

    #             plt.savefig(os.path.join(plot_dir,
    #                                      "Fit_" + state_name + "_M" + filetype))

    #             # INFECTADOS - IDOSOS E JOVENS
    #             plt.figure(24)

    #             plot_median(Ii + Ij, cor[0], ls[0], 'Model', data_sim)

    #             plt.plot(dfMS['data'], dfMS['casosAcumulado'] * sub_report,
    #                      ls[1], color=cor[1], label='Reported Data*sub_report')

    #             plot_ci(Ii + Ij, cor[0], data_sim)

    #             ax = plt.gca()
    #             ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_ticks))
    #             ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #             # ax.set_xlim(pd.Timestamp('15/03/2020'),
    #             #            pd.Timestamp('15/05/2020')) # 15/04/2020

    #             plt.gcf().autofmt_xdate()  # Rotation

    #             # ax.set_ylim(0, 100_000) # 10_000
    #             pos_format(title_fit, 'Infected people', 'Date')

    #             plt.savefig(os.path.join(plot_dir,
    #                                      "Fit_" + state_name + "_I" + filetype))

    # if (IC_analysis == 4) and (not startdate == []):
    #     for ifig in range(11):
    #         plt.close(ifig)
