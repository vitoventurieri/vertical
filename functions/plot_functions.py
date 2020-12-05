import os
import numpy as np
import pandas as pd
#from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

#from .utils import *

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


def pos_format(title_fig, main_label_y, main_label_x):
    """
    põe labels, legenda e título, pós-formata eixo y

    :param title_fig:
    :param main_label_y:
    :param main_label_x:
    :return:
    """

    plt.suptitle(title_fig)
    plt.title(main_title, fontsize=8)
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

# PLOT bed capacity line
def bed_capacity_line(fig_number,capacity,name_variable,plot_type="total"):
                if plot_type == "byage":
                    name_variable += "ey"
                
                plt.figure(fig_number)
                plt.hlines(capacity,
                                1,
                                max(t_space),
                                label=f'100% bed capacity'
                                , colors='k'
                                , linestyles='dotted')
                plt.legend(loc='right')
                plt.savefig(os.path.join(plot_dir, f"{name_variable}_diff_isol" + filetype))


def format_box_plot():
        lst_labels = ["Young (without isolation)",
                    "Young (elderly isolation)",
                    "Elderly (without isolation)",
                    "Elderly (elderly isolation)",
                    "Total (without isolation)",
                    "Total (elderly isolation)"]
        
        # adjust labels
        plt.subplots_adjust(left=0.25)
        
        # plt.xlabel("Removed people")
        #plt.ylabel(main_label_y)
        
        # format axis
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_yticklabels(lst_labels[::-1])
        # ax.set_xscale('log')
        # ax.set_xlim((0,10e6))
        ax.xaxis.set_major_formatter(FuncFormatter(number_formatter))
        ax.grid(False)


# PLOT TOTAL (i.e sem ser por idade)
def plot_total(Yi, Yj, name_variable,
               title_fig,
               fig_number,
               main_label_y,
               isolation_degree_idx):
    """
    plota curvas totais (i.e sem ser por idade)
    com intervalo de confiança (5% percentil, mediana, 95% percentil)
    ou um único valor (SINGLE RUN)
    """
    plt.figure(fig_number)

    i = isolation_degree_idx

    if ic_analysis == 2:  # SINGLE RUN
        plt.plot(t_space,
                 (Yi + Yj),
                 ls[i % 2],  # 0: dashed linestyle, 1: solid linestyle
                 color=cor[i],
                 label='Total' + isolation_name[i])
    else:  # CONFIDENCE INTERVAL
        plot_median(Yi + Yj, cor[i], ls[i % 2], 'Total' + isolation_name[i], t_space)
        plot_ci(Yi + Yj, cor[i], t_space)

    pos_format(title_fig, main_label_y, main_label_x)
    if fig_number > 10:  # plots juntos
        if name_variable == 'M':
            plt.legend(loc='upper left')
        if name_variable == 'R':
            plt.legend(loc='right')
    plt.savefig(os.path.join(plot_dir, name_variable + "_diff_isol" + filetype))


# PLOT POR IDADE - IDOSOS E JOVENS
def plot_byage(Yi, Yj, name_variable,
               title_fig,
               fig_number,
               main_label_y,
               isolation_degree_idx):
    """
    plota curvas por idade
    com intervalo de confiança (5% percentil, 50% mediana, 95% percentil)
    ou um único valor (SINGLE RUN)
    """

    i = isolation_degree_idx

    plt.figure(fig_number)

    if ic_analysis == 2:  #  mp.analysis == 'Single Run'
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

        complemento = isolation_name[i]
        if fig_number < 10:  # plots separados
            complemento = ''

        plot_median(Yi, cor[2 * i], ls[i % 2], 'Elderly' + complemento, t_space)
        plot_ci(Yi, cor[2 * i], t_space)
        
        plot_median(Yj, cor[1 + 2 * i], ls[(i + 1) % 2], 'Young' + complemento, t_space)
        plot_ci(Yj, cor[1 + 2 * i], t_space)
        
        plot_median(Yj + Yi, cor[3 + 2 * i], ls[(i + 1) % 2], 'Total' + complemento, t_space)
        plot_ci(Yj + Yi, cor[3 + 2 * i], t_space)

        complemento = '_diff_isol'

        #dfcity_query = parameter_for_rt_fit_analisys(city_code)

        #exibition_date = dfcity_query.loc[1, 'date']

        #print('1º dia da simulação: ' + str(exibition_date))
        #plt.plot(dfcity_query.loc[:, 'deaths'].values)

    pos_format(title_fig, main_label_y, main_label_x)
    if fig_number > 10:  # plots juntos
        if name_variable == 'M':
            plt.legend(loc='upper left')
        if name_variable == 'R':
            plt.legend(loc='right')

        plt.savefig(os.path.join(plot_dir,
                             name_variable + "ey" + complemento + filetype))


def plots(results, covid_parameters, model_parameters, plot_dir_main):
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
    
    
    Degrees of isolation (isolation_degree_idx)
    no isolation, vertical, horizontal
        
    ic_analysis
    1: Confidence Interval; 2: Single Run; 3: Sensitivity Analysis
    
    Age groups
    Elderly (60+); Young (0-59)
    
    Hospital Bed
    Ward; ICU
    
    """

    capacidade_leitos = model_parameters.bed_ward
    capacidade_UTIs = model_parameters.bed_icu
    city_name = model_parameters.city_name
    runs = model_parameters.runs
      

    # VARIÁVEIS PADRÃO DOS PLOTS
    global plot_dir
    global ic_analysis, t_space, isolation_name
    global main_title, main_label_x, main_label_y, cor, ls, filetype

    plot_dir = plot_dir_main

    ic_analysis = model_parameters.IC_analysis
    t_max = model_parameters.t_max
    t_space = np.arange(0, t_max)
    #  ["without_isolation", "elderly_isolation"]
    isolation_name = model_parameters.isolation_level

    main_title = f'{city_name}, {runs} runs'

    main_label_x = 'Days'
    main_label_y = 'Infected people'

    cor = ['b', 'r', 'k', 'g', 'y', 'm']  # Line Colors
    ls = ['-.', '-']  # Line Style

    filetype = '.pdf'  # '.pdf' # '.png' # '.svg' #

    # small_size = 10
    # medium_size = 10
    # suptitle_size = 11
    # title_size = 8

    # # Font Sizes
    # plt.rc('font', size=small_size)  # controls default text sizes
    # plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    # plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=small_size)  # legend fontsize
    # plt.rc('figure', titlesize=title_size)  # fontsize of the figure title
    
    # plt.rc('figure', figsize=(8, 6))  # Figure Size

    #print(plt.style.available)
    lst_style = ['ggplot','classic','seaborn-paper','bmh','fast']
    plt.style.use(lst_style[2])
    
    plt.rc("legend", loc='upper right')  # 'best' # 'upper right' #

    print('Plotando resultados')

    # dataframe from last day for boxplot
    df_last_day = pd.DataFrame()

    # 0: without; 1: vertical isolation 
    for isolation_degree_idx in range(len(isolation_name)):

        isolation_name_i = isolation_name[isolation_degree_idx]
        filename = isolation_name_i 

        if ic_analysis == 2:  # 'Single Run'

            query_condition = 'isolamento == @isolation_name_i '
            Ii = results.query(query_condition)['Ii']
            Ij = results.query(query_condition)['Ij']
            Hi = results.query(query_condition)['Hi']
            Hj = results.query(query_condition)['Hj']
            Ui = results.query(query_condition)['Ui']
            Uj = results.query(query_condition)['Uj']
            Mi = results.query(query_condition)['Mi']
            Mj = results.query(query_condition)['Mj']

        else:  # SENSITIVITY ANALYSIS OR CONFIDENCE INTERVAL
            Si,Sj,Ei,Ej,Ii,Ij,Ri,Rj,Hi,Hj,Ui,Uj,Mi,Mj = (
                np.zeros((len(results), t_max)) for _ in range(14))

            for ii in range(len(results)):
                query_condition = 'isolamento == @isolation_name_i '

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

        if ic_analysis == 3:  #   mp.analysis == 'Sensitivity'

            r0 = covid_parameters.beta / covid_parameters.gamma
            r0min = r0[0]
            r0max = r0[len(results) - 1]

            for ii in range(len(results)):
                a = (r0[ii] - r0min) / (r0max - r0min)
                plt.plot(t_space, Ii[ii, ] + Ij[ii, ],
                         color=[a, 0, 1-a, 1],
                         label=isolation_name_i ,
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

            # DIFERENTES ISOLAMENTOS (mesmo gráfico, fig_number fixo)
            # par: total
            # ímpar: byage
            #
            # figures:
            # 100 total (I)
            # 101 byage (I)
            # 102 total (H)
            # 103 byage (H)
            # 104 total (U)
            # 105 byage (U)
            # 106 total (R)
            # 107 byage (R)
            # 108 total (M)
            # 109 byage (M)

            # INFECTADOS TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Yi=Ii, Yj=Ij, name_variable='I',
               title_fig='Infected people by different isolation degrees',
               fig_number=100,
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)

            # INFECTADOS - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Yi=Ii, Yj=Ij, name_variable='I',
               title_fig='Infected by age group for different isolation degrees',
               fig_number=101,
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)

            # HOSPITALIZADOS WARD - TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Yi=Hi, Yj=Hj, name_variable='H',
               title_fig='Ward hospitalized people by different isolation degrees',
               fig_number=102,
               main_label_y='Bed demand',
               isolation_degree_idx=isolation_degree_idx)
            
            # HOSPITALIZADOS WARD - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Yi=Hi, Yj=Hj, name_variable='H',
               title_fig='Ward bed demand by age group for different isolation degrees',
               fig_number=103,
               main_label_y='Ward bed demand',
               isolation_degree_idx=isolation_degree_idx)

            # HOSPITALIZADOS UTI - TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Yi=Ui, Yj=Uj, name_variable='U',
               title_fig='ICU hospitalized people by different isolation degrees',
               fig_number=104,
               main_label_y='Bed demand',
               isolation_degree_idx=isolation_degree_idx)

            # HOSPITALIZADOS UTI - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Yi=Ui, Yj=Uj, name_variable='U',
               title_fig='ICU bed demand by age group for different isolation degrees',
               fig_number=105,
               main_label_y='ICU bed demand',
               isolation_degree_idx=isolation_degree_idx)
            
            # REMOVIDOS TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Yi=Ri, Yj=Rj, name_variable='R',
               title_fig='Removed people by different isolation degrees',
               fig_number=106,
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)

            # REMOVIDOS - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Yi=Ri, Yj=Rj, name_variable='R',
               title_fig='Removed by age group for different isolation degrees',
               fig_number=107,
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)
            
            # OBITOS TOTAL - DIFERENTES ISOLAMENTOS
            plot_total(Yi=Mi, Yj=Mj, name_variable='M',
               title_fig='Deceased people by different isolation degrees',
               fig_number=108,
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)

            # OBITOS - IDOSOS E JOVENS - DIFERENTES ISOLAMENTOS
            plot_byage(Yi=Mi, Yj=Mj, name_variable='M',
               title_fig='Deceased by age group for different isolation degrees',
               fig_number=109,
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)
           
            # horizontal line for 100% capacity (total hospitalized plots)
            if isolation_degree_idx == 1:  # vertical (last iteration)
                bed_capacity_line(fig_number=102,  # total ward
                            capacity=capacidade_leitos,
                            name_variable="H")
                bed_capacity_line(fig_number=103,  # byage ward
                            capacity=capacidade_leitos,
                            name_variable="H",
                            plot_type="byage")
                bed_capacity_line(fig_number=104,  # total icu
                            capacity=capacidade_UTIs,
                            name_variable="U")
                bed_capacity_line(fig_number=105,  # byage icu
                            capacity=capacidade_UTIs,
                            name_variable="U",
                            plot_type="byage")

            # DIFERENTES ISOLAMENTOS (gráficos separados, fig_number varia)
            # par: no isolation
            # ímpar: vertical isolation
            # 
            # figures:
            # 0 byage (I) no isolation
            # 1 byage (I) vertical isolation
            # 2 byage (H,U) no isolation
            # 3 byage (H,U) vertical isolation
            # 4 byage (M) no isolation
            # 5 byage (M) vertical isolation

            # INFECTADOS - IDOSOS E JOVENS           
            plot_byage(Yi=Ii, Yj=Ij, name_variable='I',
               title_fig='Infected by age group' + isolation_name_i,
               fig_number=isolation_degree_idx,  # 0, 1
               main_label_y=main_label_y,
               isolation_degree_idx=isolation_degree_idx)

            # LEITOS DEMANDADOS - IDOSOS E JOVENS
            plt.figure(2 + isolation_degree_idx)  # 2,3

            if ic_analysis == 2:  # Single run
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
            
            ylabel = 'Bed demand'
            pos_format(title_fig=ylabel+isolation_name_i,
                        main_label_y=ylabel,
                        main_label_x=main_label_x)
            plt.savefig(os.path.join(plot_dir, "HUey" + filename + filetype))

            # OBITOS - IDOSOS E JOVENS
            plt.figure(4 + isolation_degree_idx)  # 4,5

            # OBITOS - IDOSOS E JOVENS           
            plot_byage(Yi=Mi, Yj=Mj, name_variable='M',
               title_fig='Deceased by age group' + isolation_name_i,
               fig_number=4 + isolation_degree_idx,  # 4,5
               main_label_y='Deceased people',
               isolation_degree_idx=isolation_degree_idx)

            if model_parameters.fit_analysis:
                dfcity_query = model_parameters.df_cidade
                exibition_date = dfcity_query.loc[1, 'data']
                print('1º dia da simulação: ' + str(exibition_date))
                plt.plot(dfcity_query.loc[:, 'obitosAcumulado'].values, color='goldenrod', label='Observed')
                plt.legend(loc='upper left')
                        
            plt.savefig(os.path.join(plot_dir, "Mey" + filename + filetype))           

            # BOXPLOT
            last_day_idx = -1
            # Removed
            df_last_day["Re_"+isolation_name_i] = Ri[:,last_day_idx]
            df_last_day["Ry_"+isolation_name_i] = Rj[:,last_day_idx]
            df_last_day["R_"+isolation_name_i] = Ri[:,last_day_idx]+Rj[:,last_day_idx]
            # Deceased
            df_last_day["Me_"+isolation_name_i] = Mi[:,last_day_idx]
            df_last_day["My_"+isolation_name_i] = Mj[:,last_day_idx]
            df_last_day["M_"+isolation_name_i] = Mi[:,last_day_idx]+Mj[:,last_day_idx]
    
    # sort columns alphabetically
    df_last_day = df_last_day.reindex(sorted(df_last_day.columns), axis=1)
      
    # BOXPLOT DIFERENTES ISOLAMENTOS (gráficos juntos, fig_number fixo)
            
    # figures:
    # 1000 byage, total (R) removed
    # 1001 byage, total (M) deceased

    plt.figure(1000)
    df_last_day.loc[:, df_last_day.columns.str.startswith('R')].boxplot(vert=False, showfliers=False)
    plt.suptitle(f'Removed people at day {t_max}')
    plt.title(main_title, fontsize=8)
    format_box_plot()
    plt.savefig(os.path.join(plot_dir, "BoxPlot_R" + filetype))
    
    plt.figure(1001)
    df_last_day.loc[:, df_last_day.columns.str.startswith('M')].boxplot(vert=False, showfliers=False)
    plt.suptitle(f'Deacesed people at day {t_max}')
    plt.title(main_title, fontsize=8)
    format_box_plot()
    plt.savefig(os.path.join(plot_dir, "BoxPlot_M" + filetype))

    plt.close('all')


    #             for ifig in range(11):
    #                 plt.close(ifig)

    #             data_sim = pd.to_datetime(t_space, unit='D',
    #                                       origin=pd.Timestamp(startdate))
    #             # data_sim = pd.date_range(start=startdate, periods=t_max, freq='D')

    #             x_ticks = 14  # de quantos em quantos dias aparece tick no plot

    #             ax = plt.gca()
    #             ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_ticks))
    #             ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #             # ax.set_xlim(pd.Timestamp('15/03/2020'), pd.Timestamp('15/04/2020'))

    #             plt.gcf().autofmt_xdate()  # Rotation

