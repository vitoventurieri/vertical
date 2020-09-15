import numpy as np
import pandas as pd
from scipy.integrate import odeint


def get_rt_by_city(city):
    """
    Return a dataframe with r (basic reproduction number) over time t for a city

    :param city: string
    :return: dataframe with the selected city
    """
    cities = {'fortaleza': 'Fortaleza',
              'sao_paulo': 'SaoPaulo',
              'maceio': 'Maceio',
              'sao_luiz': 'SaoLuis'}

    return pd.read_csv(f"data/Re_{cities[city]}.csv")


def append_df(df, ret, t, nivel_isolamento):
    """
    Append the dataframe

    :param df: dataframe to be appended
    :param ret: solution of the SEIR
    :param t: time to append
    :param nivel_isolamento: string "without isolation" and "elderly isolation"
    :return: df appended
    """
    Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, dHi, dHj, Ui, Uj, dUi, dUj, Mi, Mj, pHi, pHj, pUi, pUj, pMi, pMj = ret.T
    df = df.append(pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej,
                                 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
                                 'Hi': Hi, 'Hj': Hj, 'dHi': dHi, 'dHj': dHj, 'Ui': Ui, 'Uj': Uj,
                                 'dUi': dUi, 'dUj': dUj, 'Mi': Mi, 'Mj': Mj,
                                 'pHi': pHi, 'pHj': pHj, 'pUi': pUi, 'pUj': pUj, 'pMi': pMi,
                                 'pMj': pMj}, index=t)
                   .assign(isolamento=nivel_isolamento))
    return df


def run_SEIR_ODE_model(covid_parameters, model_parameters) -> pd.DataFrame:
    """
    Runs the simulation

    :param covid_parameters:
    :param model_parameters:

    :return: DF_list
    pd.DataFrame with results for SINGLE RUN
    list of pd.DataFrame for SENSITIVITY ANALYSIS AND CONFIDENCE INTERVAL
    """

    cp = covid_parameters
    mp = model_parameters

    # Variaveis apresentadas em base diaria
    # A grid of time points (in days)
    t = range(mp.t_max)

    # CONDICOES INICIAIS
    # Initial conditions vector
    SEIRHUM_0_0 = initial_conditions(mp)

    niveis_isolamento = mp.isolation_level  # ["without_isolation", "elderly_isolation"]

    if mp.IC_analysis == 4:

        fonte_rt = pd.read_csv(r"data/Re_SaoPaulo.csv", sep=',')
        # fonte_rt = SaoLuis(mp.city)

        runs = len(cp.alpha)
        print('Rodando ' + str(runs) + ' casos')
        print('Para ' + str(mp.t_max) + ' dias')
        print('Para cada um dos ' + str(len(niveis_isolamento)) + ' niveis de isolamento de entrada')
        print('')

        aNumber = 180
        tNumber = mp.t_max // aNumber
        tNumberEnd = mp.t_max % aNumber
        if tNumberEnd != 0:
            aNumber += 1
        else:
            tNumberEnd = tNumber


        DF_list = list()  # list of data frames
        for ii in range(runs):  # sweeps the data frames list
            df = pd.DataFrame()
            for i in range(len(niveis_isolamento)):
                # 1: without; 2: vertical

                # Integrate the SEIR equations over the time grid, t
                # PARAMETROS PARA CALCULAR DERIVADAS
                args = args_assignment(cp, mp, i, ii)
                argslist = list(args)
                SEIRHUM_0 = SEIRHUM_0_0
                t = range(tNumber)
                ret = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
                Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, dHi, dHj, Ui, Uj, dUi, dUj, Mi, Mj, \
                    pHi, pHj, pUi, pUj, pMi, pMj = ret.T
                contador = 0

                for a in range(aNumber):
                    if a == aNumber - 1:
                        t = range(tNumberEnd + 1)
                    else:
                        t = range(tNumber + 1)
                    casa_negativa = -1
                    SEIRHUM_0 = Si[casa_negativa], Sj[casa_negativa], Ei[casa_negativa], Ej[casa_negativa], Ii[
                        casa_negativa], Ij[casa_negativa], Ri[casa_negativa], Rj[casa_negativa], Hi[casa_negativa], Hj[
                                    casa_negativa], dHi[casa_negativa], dHj[casa_negativa], Ui[casa_negativa], Uj[
                                    casa_negativa], dUi[casa_negativa], dUj[casa_negativa], Mi[casa_negativa], Mj[
                                    casa_negativa], pHi[casa_negativa], pHj[casa_negativa], pUi[casa_negativa], pUj[
                                    casa_negativa], pMi[casa_negativa], pMj[casa_negativa]
                    retTemp = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
                    ret = retTemp[1:]
                    Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, dHi, dHj, Ui, Uj, dUi, dUj, Mi, Mj, \
                        pHi, pHj, pUi, pUj, pMi, pMj = ret.T
                    t = t[1:]

                    contador = contador + 1
                    if a < 50:
                        effectiver =  fonte_rt.iloc[(contador + 43), -3]  # np.random.random()/2 + 1
                        print(effectiver)
                        argslist[2] = (cp.gamma[ii] * effectiver * mp.population) / (Si[-1] + Sj[-1])
                        args = tuple(argslist)

                    elif a == 50:
                        argslist[2] = (cp.gamma[ii] * 1.17 * mp.population) / (Si[-1] + Sj[-1])
                    #     args = tuple(argslist)
                    #
                    else:
                    #     print(argslist[2])
                         pass

                    df = append_df(df, ret, t, niveis_isolamento[i])

            DF_list.append(df)
    elif mp.IC_analysis == 2:
        ii = 1
        df = pd.DataFrame()

        # 1: without; 2: vertical
        for i in range(len(niveis_isolamento)):
            # PARAMETROS PARA CALCULAR DERIVADAS
            args = args_assignment(cp, mp, i, ii)
            # Integrate the SEIR equations over the time grid, t
            ret = odeint(derivSEIRHUM, SEIRHUM_0_0, t, args)
            # Append the solutions
            df = append_df(df, ret, t, niveis_isolamento[i])


        DF_list = df

    else:
        SEIRHUM_0 = SEIRHUM_0_0
        DF_list = list()  # list of data frames

        runs = len(cp.alpha)
        print('Rodando ' + str(runs) + ' casos')
        print('Para ' + str(mp.t_max) + ' dias')
        print('Para cada um dos ' + str(len(niveis_isolamento)) + ' niveis de isolamento de entrada')
        print('')

        for ii in range(runs):  # sweeps the data frames list
            df = pd.DataFrame()
            # 1: without; 2: vertical
            for i in range(len(niveis_isolamento)):
                # PARAMETROS PARA CALCULAR DERIVADAS
                args = args_assignment(cp, mp, i, ii)
                # Integrate the SEIR equations over the time grid, t
                ret = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
                # Append the solutions
                df = append_df(df, ret, t, niveis_isolamento[i])
            DF_list.append(df)

    return DF_list


def initial_conditions(mp):
    """
    Assembly of the initial conditions

    :param mp: model_parameters (named tuple)
    :return: tuple SEIRHUM_0 with the variables:
    Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, Ui0, Uj0, Mi0, Mj0
    S: Suscetible, Exposed, Infected, Removed, Ward Bed demand, ICU bed demand, Death
    i: elderly (idoso, 60+); j: young (jovem, 0-59 years)
    : (check derivSEIRHUM for variables definitions)
    """
    Ei0 = mp.init_exposed_elderly  # Ee0
    Ej0 = mp.init_exposed_young  # Ey0
    Ii0 = mp.init_infected_elderly  # Ie0
    Ij0 = mp.init_infected_young  # Iy0
    Ri0 = mp.init_removed_elderly  # Re0
    Rj0 = mp.init_removed_young  # Ry0
    Hi0 = mp.init_hospitalized_ward_elderly  # He0
    Hj0 = mp.init_hospitalized_ward_young  # Hy0
    dHi0 = mp.init_hospitalized_ward_elderly_excess
    dHj0 = mp.init_hospitalized_ward_young_excess
    Ui0 = mp.init_hospitalized_icu_elderly  # Ue0
    Uj0 = mp.init_hospitalized_icu_young  # Uy0
    dUi0 = mp.init_hospitalized_icu_elderly_excess
    dUj0 = mp.init_hospitalized_icu_young_excess
    Mi0 = mp.init_deceased_elderly  # Me0
    Mj0 = mp.init_deceased_young  # My0
    pHi0 = 0
    pHj0 = 0
    pUi0 = 0
    pUj0 = 0
    pMi0 = 0
    pMj0 = 0

    # Suscetiveis
    Si0 = mp.population * mp.population_rate_elderly - Ii0 - Ri0 - Ei0  # Suscetiveis idosos
    Sj0 = mp.population * (1 - mp.population_rate_elderly) - Ij0 - Rj0 - Ej0  # Suscetiveis jovens

    SEIRHUM_0 = Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0, \
                dHi0, dHj0, Ui0, Uj0, dUi0, dUj0, Mi0, Mj0, \
                pHi0, pHj0, pUi0, pUj0, pMi0, pMj0
    return SEIRHUM_0

def args_assignment(cp, mp, i, ii):
    """
    Assembly of the derivative parameters

    :param cp: covid_parameters
    :param mp: model_parameters
    :param i: sweeps niveis_isolamento = ["without isolation", "elderly isolation"]
    :param ii: sweeps runs for CONFIDENCE INTERVAL and SENSITIVITY ANALYSIS
    :return: tuple args with the variables: (check derivSEIRHUM for variables definitions)

    N, alpha, beta, gamma, delta,
    los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
    taxa_mortalidade_i, taxa_mortalidade_j, contact_matrix, pI,
    infection_to_hospitalization, infection_to_icu, capacidade_UTIs, capacidade_Ward, Normalization_constant,
    pH, pU

    """
    N0 = mp.population
    pI = mp.population_rate_elderly
    Normalization_constant = mp.Normalization_constant[0]
    # Because if the constant be scaled after changing the contact matrix again,
    # it should lose the effect of reducing infection rate
    if mp.IC_analysis == 2:  # SINGLE RUN
        alpha = cp.alpha
        beta = cp.beta
        gamma = cp.gamma
        delta = cp.delta
        taxa_mortalidade_i = cp.mortality_rate_elderly
        taxa_mortalidade_j = cp.mortality_rate_young

        tax_int_i = cp.internation_rate_ward_elderly
        tax_int_j = cp.internation_rate_ward_young

        tax_uti_i = cp.internation_rate_icu_elderly
        tax_uti_j = cp.internation_rate_icu_young
    else:  # CONFIDENCE INTERVAL OR SENSITIVITY ANALYSIS
        alpha = cp.alpha[ii]
        taxa_mortalidade_i = cp.mortality_rate_elderly[ii]
        taxa_mortalidade_j = cp.mortality_rate_young[ii]
        beta = cp.beta[ii]
        gamma = cp.gamma[ii]
        delta = cp.delta[ii]

        tax_int_i = cp.internation_rate_ward_elderly[ii]
        tax_int_j = cp.internation_rate_ward_young[ii]

        tax_uti_i = cp.internation_rate_icu_elderly[ii]
        tax_uti_j = cp.internation_rate_icu_young[ii]

    contact_matrix = mp.contact_matrix[i]
    # taxa_mortalidade_i = cp.mortality_rate_elderly
    # taxa_mortalidade_j = cp.mortality_rate_young
    pH = cp.pH
    pU = cp.pU

    los_leito = cp.los_ward
    los_uti = cp.los_icu

    infection_to_hospitalization = cp.infection_to_hospitalization
    infection_to_icu = cp.infection_to_icu

    proportion_of_ward_mortality_over_total_mortality_elderly = 0.407565
    proportion_of_ward_mortality_over_total_mortality_young = 0.357161

    proportion_of_icu_mortality_over_total_mortality_elderly = 0.592435
    proportion_of_icu_mortality_over_total_mortality_young = 0.642839

    # tax_int_i = cp.internation_rate_ward_elderly
    # tax_int_j = cp.internation_rate_ward_young
    #
    # tax_uti_i = cp.internation_rate_icu_elderly
    # tax_uti_j = cp.internation_rate_icu_young

    capacidade_UTIs = mp.bed_icu
    capacidade_Ward = mp.bed_ward

    args = (N0, alpha, beta, gamma, delta,
            los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
            taxa_mortalidade_i, taxa_mortalidade_j, contact_matrix, pI,
            infection_to_hospitalization, infection_to_icu, capacidade_UTIs, capacidade_Ward, Normalization_constant,
            pH, pU, proportion_of_ward_mortality_over_total_mortality_elderly, proportion_of_ward_mortality_over_total_mortality_young, proportion_of_icu_mortality_over_total_mortality_elderly, proportion_of_icu_mortality_over_total_mortality_young)
    return args

def derivSEIRHUM(SEIRHUM, t, N0, alpha, beta, gamma, delta,
                 los_leito, los_uti, tax_int_i, tax_int_j, tax_uti_i, tax_uti_j,
                 taxa_mortalidade_i, taxa_mortalidade_j, contact_matrix, pI,
                 infection_to_hospitalization, infection_to_icu, capacidade_UTIs, capacidade_Ward,
                 Normalization_constant, pH, pU, proportion_of_ward_mortality_over_total_mortality_elderly, proportion_of_ward_mortality_over_total_mortality_young, proportion_of_icu_mortality_over_total_mortality_elderly, proportion_of_icu_mortality_over_total_mortality_young):
    """
    Compute the derivatives of all the compartments

    :param SEIRHUM: array with the following variables
    S: Suscetible, E: Exposed, I: Infected, R: Recovered,
    H: Hospitalized, U: ICU, M: Deacesed
    suffixes i: elderly (idoso, 60+); j: young (jovem, 0-59 years)
    :param t: time to compute the derivative
    :param N0: population
    :param alpha: incubation rate
    :param beta: contamination rate
    :param gamma: infectivity rate
    :param delta:
    :param los_leito: average Length Of Stay for wards
    :param los_uti: average Length Of Stay in ICU beds
    :param tax_int_i: hospitalization rate for elderly in ward beds
    :param tax_int_j: hospitalization rate for young in ward beds
    :param tax_uti_i: hospitalization rate for elderly in ICU beds
    :param tax_uti_j: hospitalization rate for young in ICU beds
    :param taxa_mortalidade_i: mortality rate for elderly
    :param taxa_mortalidade_j: mortality rate for young
    :param contact_matrix:
    :param pI: elderly population proportion
    :param infection_to_hospitalization: time [days] from infection to hospitalization
    :param infection_to_icu: time [days] from infection to ICU hospitalization
    :param capacidade_UTIs: available ICU beds
    :param capacidade_Ward: available ward beds
    :param Normalization_constant:
    :param pH: pre-Hospitalization compartment
    :param pU: pre-ICU compartment
    :return: derivatives
    """
    # Vetor variaveis incognitas
    Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj, dHi, dHj, Ui, Uj, dUi, dUj, Mi, Mj, pHi, pHj, pUi, pUj, pMi, pMj = SEIRHUM

    Iij = np.array([[Ij / ((1 - pI) * N0)], [Ii / (pI * N0)]])
    Sij = np.array([[Sj], [Si]])
    dSijdt = -(beta / Normalization_constant) * np.dot(contact_matrix, Iij) * Sij
    dSjdt = dSijdt[0]
    dSidt = dSijdt[1]
    dEidt = - dSidt - alpha * Ei
    dEjdt = - dSjdt - alpha * Ej
    dIidt = alpha * Ei - gamma * Ii
    dIjdt = alpha * Ej - gamma * Ij
    dRidt = gamma * Ii
    dRjdt = gamma * Ij

    dpHi = -tax_int_i * dSidt - pHi / infection_to_hospitalization
    dpHj = -tax_int_j * dSjdt - pHj / infection_to_hospitalization
    dpUi = -tax_uti_i * dSidt - pUi / infection_to_icu
    dpUj = -tax_uti_j * dSjdt - pUj / infection_to_icu
    dpMi = 1 #-taxa_mortalidade_i * dSidt - pMi * delta
    dpMj = 1 #-taxa_mortalidade_j * dSjdt - pMj * delta

    coisa = 1 / 500
    coisa2 = -coisa * (Hi + Hj - capacidade_Ward)
    coisa = 1 / 50
    coisa3 = -coisa * (Ui + Uj - capacidade_UTIs)

    # Leitos demandados
    dHidt = (pHi / infection_to_hospitalization) * (1 - 1 / (1 + np.exp(coisa2))) - Hi / los_leito
    dHjdt = (pHj / infection_to_hospitalization) * (1 - 1 / (1 + np.exp(coisa2))) - Hj / los_leito

    dUidt = (pUi / infection_to_icu) * (1 - 1 / (1 + np.exp(coisa3))) - Ui / los_uti
    dUjdt = (pUj / infection_to_icu) * (1 - 1 / (1 + np.exp(coisa3))) - Uj / los_uti

    # Leitos demandados em excesso
    ddHidt = (pHi / infection_to_hospitalization) * (1 / (1 + np.exp(coisa2)))
    ddHjdt = (pHj / infection_to_hospitalization) * (1 / (1 + np.exp(coisa2)))

    ddUidt = (pUi / infection_to_icu) * (1 / (1 + np.exp(coisa3)))
    ddUjdt = (pUj / infection_to_icu) * (1 / (1 + np.exp(coisa3)))

    # Obitos

    dMidt = (Ui / los_uti) * (taxa_mortalidade_i*proportion_of_icu_mortality_over_total_mortality_elderly/tax_uti_i) + (Hi / los_uti) * (taxa_mortalidade_i*proportion_of_ward_mortality_over_total_mortality_elderly/tax_int_i) + ddHidt * pH + ddUidt * pU
    dMjdt = (Uj / los_uti) * (taxa_mortalidade_j*proportion_of_icu_mortality_over_total_mortality_young/tax_uti_j) + (Hj / los_uti) * (taxa_mortalidade_j*proportion_of_ward_mortality_over_total_mortality_young/tax_int_j) + ddHjdt * pH + ddUjdt * pU

    #dMidt = (Ui / los_uti) * (taxa_mortalidade_i/tax_uti_i) + ddHidt * pH + ddUidt * pU
    #dMjdt = (Uj / los_uti) * (taxa_mortalidade_j/tax_uti_j) + ddHjdt * pH + ddUjdt * pU

    # dMidt = pMi * delta + ddHidt * pH + ddUidt * pU
    # dMjdt = pMj * delta + ddHjdt * pH + ddUjdt * pU

    return (dSidt, dSjdt, dEidt, dEjdt, dIidt, dIjdt, dRidt, dRjdt,
            dHidt, dHjdt, ddHidt, ddHjdt, dUidt, dUjdt, ddUidt, ddUjdt, dMidt, dMjdt,
            dpHi, dpHj, dpUi, dpUj, dpMi, dpMj)
