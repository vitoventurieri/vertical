# cd C:\Users\Fuck\Documents\python\quarentena_vertical
# python SEIR_IJ_v7.py

import numpy as np
import pandas as pd


def get_input_data():
    '''Provides the inputs for the simulation'''
    from collections import namedtuple

    Demograph_Parameters = namedtuple('Demograph_Parameters',
                                      ['population',  # N
                                       'population_rate_old',  # percentual_pop_idosa
                                       'bed_regular',  # capacidade_leitos
                                       'bed_uti'  # capacidade_UTIs
                                       ]
                                      )
    demograph_parameters = Demograph_Parameters(
        # Brazilian Population
        population=210000000,  # 210 millions, 2020 forecast, Source: IBGE's app
        # Brazilian old people proportion (age: 55+)
        population_rate_old=0.2,  # 20%, 2020 forecast, Source: IBGE's app
        # Brazilian places
        bed_regular=295083,  # regular bed, Source: CNES, 13/04/2020
        bed_uti=32329,  # bed UTIs, Source: CNES, 13/04/2020
    )

    Covid_Parameters = namedtuple('Covid_Parameters',
                                  ['basic_reproduction_number',  # ErreZero
                                   'infectivity_period',  # tempo_de_infecciosidade
                                   'incubation_period',  # tempo_de_incubacao
                                   'mortality_rate_old',  # taxa_mortalidade_i
                                   'mortality_rate_young',  # taxa_mortalidade_j
                                   'los_regular',  # los_leito
                                   'los_uti',  # los_uti
                                   'internation_rate_regular_old',  # tax_int_i
                                   'internation_rate_uti_old',  # tax_uti_i
                                   'internation_rate_regular_young',  # tax_int_j
                                   'internation_rate_uti_young'  # tax_uti_j
                                   ]
                                  )
    covid_parameters = Covid_Parameters(
        # Basic Reproduction Number
        basic_reproduction_number=2.3,  # 0.8#1.3#1.8#2.3#2.8#
        # Infectivity Period (in days)
        infectivity_period=10,  # 5#7.5#10#12.5#15
        # Incubation Period (in days)
        incubation_period=5,  # 1#2.5#5#7.5#10#12.5#15
        # Mortality Rates, Source: min CDC
        mortality_rate_old=0.034,  # old ones: 55+ years
        mortality_rate_young=0.002,  # young ones: 0-54 years
        # Length of Stay (in days)
        los_regular=8.9 + 2,  # regular, Source: Wuhan
        los_uti=8 + 3,  # UTI, Source: Wuhan
        # Internation Rate by type and age, Source: min CDC
        internation_rate_regular_old=0.263,  # regular for old ones: 55+ years
        internation_rate_uti_old=0.071,  # UTI for old ones: 55+ years
        internation_rate_regular_young=0.154,  # regular for young ones: 0-54 years
        internation_rate_uti_young=0.03  # UTI for young ones: 0-54 years
    )

    Model_Parameters = namedtuple('Model_Parameters',
                                  ['contact_reduction_old',  # omega_i
                                   'contact_reduction_young',  # omega_j
                                   'lotation',  # lotacao
                                   'init_exposed_old',  # Ei0
                                   'init_exposed_young',  # Ej0
                                   'init_infected_old',  # Ii0
                                   'init_infected_young',  # Ij0
                                   'init_removed_old',  # Ri0
                                   'init_removed_young'  # Rj0
                                   ]
                                  )
    model_parameters = Model_Parameters(
        # Social contact reduction factor
        contact_reduction_old=1.0,  # 0.2#0.4#0.6#0.8#1.0# # old ones: 55+ years
        contact_reduction_young=1.0,  # 0.2#0.4#0.6#0.8#1.0# # young ones: 0-54 years
        # Scenaries for health system colapse
        lotation=(0.3, 0.5, 0.8, 1),  # 30, 50, 80, 100% capacity
        init_exposed_old=20000,  # initial exposed population old ones: 55+ years
        init_exposed_young=20000,  # initial exposed population young ones: 0-54 years
        init_infected_old=5520,  # initial infected population old ones: 55+ years
        init_infected_young=10000,  # initial infected population young ones: 0-54 years
        init_removed_old=3000,  # initial removed population old ones: 55+ years
        init_removed_young=6240  # initial removed population young ones: 0-54 years
    )

    # Tempo de analise (dias)
    t_max = 2 * 365
    # t_max: 'number of days to run'

    return (demograph_parameters, covid_parameters, model_parameters, t_max)


def run_SEIR_ODE_model(demograph_parameters, covid_parameters, model_parameters, t_max) -> pd.DataFrame:
    '''Runs the simulation'''
    from scipy.integrate import odeint

    (N, percentual_pop_idosa, capacidade_leitos, capacidade_UTIs) = demograph_parameters
    (ErreZero, tempo_de_infecciosidade, tempo_de_incubacao,
     taxa_mortalidade_i, taxa_mortalidade_j, los_leito, los_uti,
     tax_int_i, tax_uti_i, tax_int_j, tax_uti_j) = covid_parameters

    (omega_i, omega_j, lotacao, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0) = model_parameters

    # Variaveis apresentadas em base diaria
    # A grid of time points (in days)
    t = range(t_max)
    # dt = .1
    # t = np.linspace(0, t_max, int(t_max/dt) + 1)

    # CONDICOES INICIAIS
    # Suscetiveis
    Si0 = N * percentual_pop_idosa - Ii0 - Ri0 - Ej0 - Ei0  # Suscetiveis idosos
    Sj0 = N * (1 - percentual_pop_idosa) - Ij0 - Rj0 - Ej0 - Ei0  # Suscetiveis jovens
    # Leitos normais demandados
    Hi0 = Ii0 * tax_int_i
    Hj0 = Ij0 * tax_int_j
    # Leitos UTIs demandados
    Ui0 = Ii0 * tax_uti_i
    Uj0 = Ij0 * tax_uti_j
    # Obitos
    Mi0 = Ri0 * taxa_mortalidade_i
    Mj0 = Rj0 * taxa_mortalidade_j

    # Variaveis de apoio
    alpha = 1 / tempo_de_incubacao
    los_leito_inv = 1 / los_leito
    los_uti_inv = 1 / los_uti
    gamma = 1 / tempo_de_infecciosidade
    beta = ErreZero * gamma

    # The SEIR model differential equations.
    def deriv(y, t, N, beta, gamma, omega_i, omega_j, alpha, los_leito_inv, los_uti_inv,
              tax_int_i, tax_uti_i, tax_int_j, tax_uti_j, taxa_mortalidade_i, taxa_mortalidade_j):
        '''Computes the Derivatives'''

        #   S, E, I, R = y
        #   dSdt = -beta * S * I / N
        #   dEdt = -dSdt - alpha*E
        #   dIdt = alpha*E - gamma*I
        #   dRdt = gamma * I

        # Vetor variaveis incognitas
        Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = y

        # Leis de Evolucao modelo SEIR com S e E subdivididos por idade
        # SISTEMA DE EDOs acoplado
        dSidt = - beta * omega_i * Si * (Ii + Ij) / N
        dSjdt = - beta * omega_j * Sj * (Ii + Ij) / N
        dEidt = - dSidt - alpha * Ei
        dEjdt = - dSjdt - alpha * Ej
        dIidt = alpha * Ei - gamma * Ii
        dIjdt = alpha * Ej - gamma * Ij
        dRidt = gamma * Ii
        dRjdt = gamma * Ij

        # TALVEZ PUDESSE SER PosProcessado, nao eh acoplado

        # CONFIRMAR CASOS NEGATIVOS

        # Leitos Normais demandados
        dHidt = tax_int_i * alpha * Ei - los_leito_inv * Hi
        dHjdt = tax_int_j * alpha * Ej - los_leito_inv * Hj
        # Leitos UTIs demandados
        dUidt = tax_uti_i * alpha * Ei - los_uti_inv * Ui
        dUjdt = tax_uti_j * alpha * Ej - los_uti_inv * Uj
        # Obitos
        dMidt = taxa_mortalidade_i * dRidt
        dMjdt = taxa_mortalidade_j * dRjdt

        return (dSidt, dSjdt, dEidt, dEjdt, dIidt, dIjdt, dRidt, dRjdt,
                dHidt, dHjdt, dUidt, dUjdt, dMidt, dMjdt)

    # The SEIR model differential equations.
    def Euler(y0, t, params):
        '''Computes the Derivatives by Semi Implicit Euler Method'''
        # Vetor variaveis incognitas
        Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, Ui0, Uj0, Mi0, Mj0 = y0
        Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = [Si0], [Sj0], [Ei0], [Ej0], [Ii0], [Ij0], [Ri0], [
            Rj0], [Hi0], [Hj0], [Ui0], [Uj0], [Mi0], [Mj0]
        (N, beta, gamma, omega_i, omega_j, alpha, los_leito_inv, los_uti_inv,
         tax_int_i, tax_uti_i, tax_int_j, tax_uti_j, taxa_mortalidade_i, taxa_mortalidade_j) = params
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
            # Leitos Normais demandados
            dHidt = tax_int_i * alpha * Ei[-1] - los_leito_inv * Hi[-1]
            dHjdt = tax_int_j * alpha * Ej[-1] - los_leito_inv * Hj[-1]
            # Leitos UTIs demandados
            dUidt = tax_uti_i * alpha * Ei[-1] - los_uti_inv * Ui[-1]
            dUjdt = tax_uti_j * alpha * Ej[-1] - los_uti_inv * Uj[-1]
            # Obitos
            dMidt = taxa_mortalidade_i * dRidt
            dMjdt = taxa_mortalidade_j * dRjdt
            # dydt = (dSidt, dSjdt, dEidt, dEjdt, dIidt, dIjdt, dRidt, dRjdt, dHidt, dHjdt, dUidt, dUjdt, dMidt, dMjdt)
            # dy = dydt * dt
            next_Si = Si[-1] + dSidt * dt
            next_Sj = Sj[-1] + dSjdt * dt
            next_Ei = Ei[-1] + dEidt * dt
            next_Ej = Ej[-1] + dEjdt * dt
            next_Ii = Ii[-1] + dIidt * dt
            next_Ij = Ij[-1] + dIjdt * dt
            next_Ri = Ri[-1] + dRidt * dt
            next_Rj = Rj[-1] + dRjdt * dt
            next_Hi = Hi[-1] + dHidt * dt
            next_Hj = Hj[-1] + dHjdt * dt
            next_Ui = Ui[-1] + dUidt * dt
            next_Uj = Uj[-1] + dUjdt * dt
            next_Mi = dMidt * dt
            next_Mj = dMjdt * dt
            Si.append(next_Si)
            Sj.append(next_Sj)
            Ei.append(next_Ei)
            Ej.append(next_Ej)
            Ii.append(next_Ii)
            Ij.append(next_Ij)
            Ri.append(next_Ri)
            Rj.append(next_Rj)
            Hi.append(next_Hi)
            Hj.append(next_Hj)
            Ui.append(next_Ui)
            Uj.append(next_Uj)
            Mi.append(next_Mi)
            Mj.append(next_Mj)

        return np.stack([Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj]).T

    # Initial conditions vector
    y0 = Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, Ui0, Uj0, Mi0, Mj0

    # PARAMETROS PARA CALCULAR DERIVADAS
    args = (N, beta, gamma, omega_i, omega_j, alpha, los_leito_inv, los_uti_inv,
            tax_int_i, tax_uti_i, tax_int_j, tax_uti_j, taxa_mortalidade_i, taxa_mortalidade_j)

    # Integrate the SIR equations over the time grid, t
    # ret = odeint(deriv, y0, t, args)
    # Integrate the SIR equations over the time grid, t
    ret = Euler(y0, t, args)
    # Update the variables
    Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, Ui, Uj, Mi, Mj = ret.T

    return pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej, 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
                         'Hi': Hi, 'Hj': Hj, 'Ui': Ui, 'Uj': Uj, 'Mi': Mi, 'Mj': Mj}, index=t)


def auxiliar_names(demograph_parameters, covid_parameters, model_parameters):
    '''Provides filename and legend for plots from the sensitivity parameter analysis'''
    (N, percentual_pop_idosa, capacidade_leitos, capacidade_UTIs) = demograph_parameters
    (ErreZero, tempo_de_infecciosidade, tempo_de_incubacao,
     taxa_mortalidade_i, taxa_mortalidade_j, los_leito, los_uti,
     tax_int_i, tax_uti_i, tax_int_j, tax_uti_j) = covid_parameters
    (omega_i, omega_j, lotacao, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0) = model_parameters

    # AUTOMATIZACAO PARA ANALISE SENSIBILIDADE PARAMETROS
    psel = 3  # 0#1#2#3#
    pvalue = (ErreZero, tempo_de_incubacao, tempo_de_infecciosidade, omega_i)
    pname = 'r0', 'tincub', 'tinfec', 'omegaI'  # parametros
    pInt = ("%.1f" % pvalue[psel])[0]  # parte inteira do parametro (1 caractere)
    pDec = ("%.1f" % pvalue[psel])[2]  # parte decimal do parametro (1 caractere)
    filename = pname[psel] + pInt + '_' + pDec
    if psel == 3:
        filename = filename + '__omegaJ' + ("%.1f" % omega_j)[0] + '_' + ("%.1f" % omega_j)[2]
    leg = (
        f'SEIR($r_0$={"%0.1f" % ErreZero})',
        f'SEIR($t_{{incubation}}$={"%0.1f" % tempo_de_incubacao})',
        f'SEIR($t_{{infectivity}}$={"%0.1f" % tempo_de_infecciosidade})',
        f'SEIR($\\omega_I$={"%0.1f" % omega_i}, $\\omega_J$={"%0.1f" % omega_j})'
    )
    legenda = leg[psel]
    return filename, legenda


def post_processing(demograph_parameters, covid_parameters, model_parameters, t_max, results):
    from datetime import datetime
    '''Performs a post_processing analysis, forecast the overload of the health system'''
    (N, percentual_pop_idosa, capacidade_leitos, capacidade_UTIs) = demograph_parameters
    (ErreZero, tempo_de_infecciosidade, tempo_de_incubacao,
     taxa_mortalidade_i, taxa_mortalidade_j, los_leito, los_uti,
     tax_int_i, tax_uti_i, tax_int_j, tax_uti_j) = covid_parameters
    (omega_i, omega_j, lotacao, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0) = model_parameters

    # Expostos UTIs
    Ei = results[['Ei']].Ei
    Ej = results[['Ej']].Ej

    # Infectados
    I = results[['Ii']].Ii + results[['Ij']].Ij
    # Removidos: Recuperados ou Falecidos
    Ri = results[['Ri']].Ri
    Rj = results[['Rj']].Rj
    R = Ri + Rj
    # Demanda Total Leitos
    # Leitos comuns
    Hi = results[['Hi']].Hi
    Hj = results[['Hj']].Hj
    H = Hi + Hj
    # Leitos UTIs
    Ui = results[['Ui']].Ui
    Uj = results[['Uj']].Uj
    U = Ui + Uj

    # Incremento Removidos
    dRi = np.zeros(t_max)
    for i in range(0, t_max - 1):
        dRi[i] = abs(Ri[i + 1] - Ri[i])
    dRi[t_max - 1] = dRi[t_max - 2]  # ultimo dia estimado como igual ao penultimo dia do periodo
    # print(*dR)
    dRj = np.zeros(t_max)
    for i in range(0, t_max - 1):
        dRj[i] = abs(Rj[i + 1] - Rj[i])
    dRj[t_max - 1] = dRj[t_max - 2]  # ultimo dia estimado como igual ao penultimo dia do periodo
    # print(*dR)

    # Leitos Normais demandados
    # dHi = tax_int_i * alpha * Ei - los_leito_inv * Hi
    # dHj = tax_int_j * alpha * Ej - los_leito_inv * Hj
    # Leitos UTIs demandados
    # dUi = tax_uti_i * alpha * Ei - los_uti_inv * Ui
    # dUj = tax_uti_j * alpha * Ej - los_uti_inv * Uj
    # Obitos
    dMi = taxa_mortalidade_i * dRi
    dMj = taxa_mortalidade_j * dRj

    # IDENTIFICADOR DE DIAS DE COLAPSOS
    # Dia em que colapsa o sistema de saude: 30, 50, 80, 100% capacidade
    dia_colapso_leitos_30 = np.min(np.where(H > capacidade_leitos * lotacao[0])[0])
    dia_colapso_leitos_50 = np.min(np.where(H > capacidade_leitos * lotacao[1])[0])
    dia_colapso_leitos_80 = np.min(np.where(H > capacidade_leitos * lotacao[2])[0])
    dia_colapso_leitos_100 = np.min(np.where(H > capacidade_leitos * lotacao[3])[0])
    dia_colapso_leitos = (dia_colapso_leitos_30, dia_colapso_leitos_50,
                          dia_colapso_leitos_80, dia_colapso_leitos_100)
    print(dia_colapso_leitos)

    dia_colapso_UTIs_30 = np.min(np.where(U > capacidade_UTIs * lotacao[0])[0])
    dia_colapso_UTIs_50 = np.min(np.where(U > capacidade_UTIs * lotacao[1])[0])
    dia_colapso_UTIs_80 = np.min(np.where(U > capacidade_UTIs * lotacao[2])[0])
    dia_colapso_UTIs_100 = np.min(np.where(U > capacidade_UTIs * lotacao[3])[0])
    dia_colapso_UTIs = (dia_colapso_UTIs_30, dia_colapso_UTIs_50,
                        dia_colapso_UTIs_80, dia_colapso_UTIs_100)
    print(dia_colapso_UTIs)

    # Casos
    index_max = np.argmax(I)  # Dia de pico de casos
    Itot = sum(I)  # Numero total de casos

    # Pacientes Hospitalizados
    Leitos = sum(H) / los_leito  # 'Leitos Demandados'
    UTIs = sum(U) / los_uti  # 'UTIs Demandadas'

    # Mitot = sum(Mi)
    # Mjtot = sum(Mj)
    # Mtot = Mitot + Mjtot

    # TimeSeries
    # datelist = pd.date_range(datetime.today(), periods=t_max).tolist()
    datelist = [d.strftime('%d/%m/%Y')
                for d in pd.date_range(datetime.today(), periods=t_max)]

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


def plots(filename, legenda, results, demograph_parameters, model_parameters, t_max):
    import matplotlib.pyplot as plt
    '''Makes two plots? 0) SEIR curve, 1) Hospital Demand'''
    (N, percentual_pop_idosa, capacidade_leitos, capacidade_UTIs) = demograph_parameters
    (omega_i, omega_j, lotacao, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0) = model_parameters
    # plot
    tamfig = (8, 6)
    fsLabelTitle = 15  # Font Size: Label and Title
    fsPlotLegend = 12  # Font Size: Plot and Legend

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
     .plot(figsize=tamfig, fontsize=fsPlotLegend, logy=False))
    # plt.hlines(capacidade_leitos*lotacao[0],1,t_max) #30%
    # plt.hlines(capacidade_leitos*lotacao[1],1,t_max) #50%
    # plt.hlines(capacidade_leitos*lotacao[2],1,t_max) #80%
    plt.hlines(capacidade_leitos * lotacao[3], 1, t_max, label='100% Leitos', colors='y', linestyles='dotted')  # 100%
    # plt.hlines(capacidade_UTIs*lotacao[0],1,t_max) #30%
    # plt.hlines(capacidade_UTIs*lotacao[1],1,t_max) #50%
    # plt.hlines(capacidade_UTIs*lotacao[2],1,t_max) #80%
    plt.hlines(capacidade_UTIs * lotacao[3], 1, t_max, label='100% UTI', colors='g', linestyles='dashed')  # 100%
    plt.title(f'Demanda diaria de leitos:\n' + legenda, fontsize=fsLabelTitle)
    plt.legend(['Leito normal idosos', 'Leito normal jovens', 'UTI idosos', 'UTI jovens'
                   , '100% Leitos', '100% UTIs'], fontsize=fsPlotLegend)
    plt.xlabel('Dias', fontsize=fsLabelTitle)
    plt.ylabel('Leitos', fontsize=fsLabelTitle)
    plt.savefig("HU_" + filename + ".png")


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


if __name__ == '__main__':
    demograph_parameters, covid_parameters, model_parameters, t_max = get_input_data()

    results = run_SEIR_ODE_model(demograph_parameters, covid_parameters, model_parameters, t_max)

    filename, legenda = auxiliar_names(demograph_parameters,
                                       covid_parameters, model_parameters)

    results.to_csv(filename + '.csv', index=False)

    post_processing(demograph_parameters, covid_parameters, model_parameters, t_max, results)

    plots(filename, legenda, results, demograph_parameters, model_parameters, t_max)
