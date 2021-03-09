import numpy as np
import pandas as pd
from scipy.integrate import odeint


def append_df(df, ret, t, nivel_isolamento):
    """
    Append the dataframe

    :param df: dataframe to be appended
    :param ret: solution of the SEIR
    :param t: time to append
    :param nivel_isolamento: string "without isolation" and "elderly isolation"
    :return: df appended
    """

    (Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj,
     WARD_excess_i, WARD_excess_j, Ui, Uj, ICU_excess_i, ICU_excess_j, Mi, Mj,
     pHi, pHj, pUi, pUj, pMi, pMj,
     WARD_survive_i, WARD_survive_j,
     WARD_death_i, WARD_death_j,
     ICU_survive_i, ICU_survive_j,
     ICU_death_i, ICU_death_j,
     WARD_discharged_ICU_survive_i,
     WARD_discharged_ICU_survive_j)  = ret.T
    df = df.append(pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej,
                                 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
                                 'Hi': Hi, 'Hj': Hj,
                                 'WARD_excess_i': WARD_excess_i, 'WARD_excess_j': WARD_excess_j, 'Ui': Ui, 'Uj': Uj,
                                 'ICU_excess_i': ICU_excess_i, 'ICU_excess_j': ICU_excess_j, 'Mi': Mi, 'Mj': Mj,
                                 'pHi': pHi, 'pHj': pHj, 'pUi': pUi, 'pUj': pUj,
                                 'pMi': pMi, 'pMj': pMj,
                                 'WARD_survive_i': WARD_survive_i, 'WARD_survive_j': WARD_survive_j,
                                 'WARD_death_i': WARD_death_i,'WARD_death_j': WARD_death_j,
                                 'ICU_survive_i':ICU_survive_i,'ICU_survive_j': ICU_survive_j,
                                 'ICU_death_i' : ICU_death_i,'ICU_death_j': ICU_death_j,
                                 'WARD_discharged_ICU_survive_i': WARD_discharged_ICU_survive_i,
                                 'WARD_discharged_ICU_survive_j':WARD_discharged_ICU_survive_j },
                                 index=t)
                   .assign(isolamento=nivel_isolamento))
    return df


def run_SEIR_ODE_model_no_constraints(covid_parameters, model_parameters) -> pd.DataFrame:
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

    if mp.IC_analysis == 4:  #  mp.analysis == 'Rt'

        df_rt_city = mp.df_rt_city

        runs = len(cp.alpha)
        print('Rodando ' + str(runs) + ' casos')
        print('Para ' + str(mp.t_max) + ' dias')
        print('Para cada um dos ' + str(len(niveis_isolamento)) 
                + ' niveis de isolamento de entrada')
        print('')

        aNumber = 180  # TODO: check 180 or comment
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

                (Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj,
                 WARD_excess_i, WARD_excess_j, Ui, Uj, ICU_excess_i, ICU_excess_j, Mi, Mj,
                 pHi, pHj, pUi, pUj, pMi, pMj,
                 WARD_survive_i, WARD_survive_j,
                 WARD_death_i, WARD_death_j,
                 ICU_survive_i, ICU_survive_j,
                 ICU_death_i, ICU_death_j,
                 WARD_discharged_ICU_survive_i,
                 WARD_discharged_ICU_survive_j)  = ret.T
                contador = 0

                for a in range(aNumber):
                    if a == aNumber - 1:
                        t = range(tNumberEnd + 1)
                    else:
                        t = range(tNumber + 1)

                    SEIRHUM_0 = tuple([x[-1] for x in [Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj,
                                                       WARD_excess_i, WARD_excess_j, Ui, Uj, ICU_excess_i, ICU_excess_j, Mi, Mj,
                                                       pHi, pHj, pUi, pUj, pMi, pMj,
                                                       WARD_survive_i, WARD_survive_j,
                                                       WARD_death_i, WARD_death_j,
                                                       ICU_survive_i, ICU_survive_j,
                                                       ICU_death_i, ICU_death_j,
                                                       WARD_discharged_ICU_survive_i,
                                                       WARD_discharged_ICU_survive_j] ])

                    retTemp = odeint(derivSEIRHUM, SEIRHUM_0, t, args)
                    ret = retTemp[1:]
                    (Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj,
                     WARD_excess_i, WARD_excess_j, Ui, Uj, ICU_excess_i, ICU_excess_j, Mi, Mj,
                     pHi, pHj, pUi, pUj, pMi, pMj,
                     WARD_survive_i, WARD_survive_j,
                     WARD_death_i, WARD_death_j,
                     ICU_survive_i, ICU_survive_j,
                     ICU_death_i, ICU_death_j,
                     WARD_discharged_ICU_survive_i,
                     WARD_discharged_ICU_survive_j)  = ret.T
                    t = t[1:]

                    contador += 1
                    if a < mp.initial_deaths_to_fit:
                        # TODO: comentar por que 43 e -3
                        effectiver = df_rt_city.iloc[(contador + 43), -3]  # np.random.random()/2 + 1
                        print(effectiver)
                        argslist[2] = (cp.gamma[ii] * effectiver * mp.population) / (Si[-1] + Sj[-1])
                        args = tuple(argslist)

                    elif a == mp.initial_deaths_to_fit:
                        # TODO: comentar por que 1.17
                        argslist[2] = (cp.gamma[ii] * 1.17 * mp.population) / (Si[-1] + Sj[-1])
                    else:
                    #     print(argslist[2])
                         pass

                    df = append_df(df, ret, t, niveis_isolamento[i])

            DF_list.append(df)
    elif mp.IC_analysis == 2:  # mp.analysis == 'Single Run'
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
        print('Para cada um dos ' + str(len(niveis_isolamento)) 
                + ' niveis de isolamento de entrada')
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
    WARD_excess_i0 = mp.init_hospitalized_ward_elderly_excess
    WARD_excess_j0 = mp.init_hospitalized_ward_young_excess
    Ui0 = mp.init_hospitalized_icu_elderly  # Ue0
    Uj0 = mp.init_hospitalized_icu_young  # Uy0
    ICU_excess_i0 = mp.init_hospitalized_icu_elderly_excess
    ICU_excess_j0 = mp.init_hospitalized_icu_young_excess
    Mi0 = mp.init_deceased_elderly  # Me0
    Mj0 = mp.init_deceased_young  # My0

    # Suscetiveis
    Si0 = mp.population * mp.population_rate_elderly - Ii0 - Ri0 - Ei0  # Suscetiveis idosos
    Sj0 = mp.population * (1 - mp.population_rate_elderly) - Ij0 - Rj0 - Ej0  # Suscetiveis jovens

    (pHi0, pHj0, pUi0, pUj0, pMi0, pMj0,
     WARD_survive_i0, WARD_survive_j0, WARD_death_i0, WARD_death_j0,
     ICU_survive_i0, ICU_survive_j0, ICU_death_i0, ICU_death_j0,
     WARD_discharged_ICU_survive_i0, WARD_discharged_ICU_survive_j0) = (
        0 for _ in range(16))

    SEIRHUM_0 = (Si0, Sj0, Ei0, Ej0, Ii0, Ij0, Ri0, Rj0, Hi0, Hj0,
                 WARD_excess_i0, WARD_excess_j0, Ui0, Uj0, ICU_excess_i0, ICU_excess_j0, Mi0, Mj0,
                 pHi0, pHj0, pUi0, pUj0, pMi0, pMj0,
                 WARD_survive_i0, WARD_survive_j0,
                 WARD_death_i0, WARD_death_j0,
                 ICU_survive_i0, ICU_survive_j0,
                 ICU_death_i0, ICU_death_j0,
                 WARD_discharged_ICU_survive_i0,
                 WARD_discharged_ICU_survive_j0)
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
    los_WARD, los_ICU, tax_int_i, tax_int_j, tax_ICU_i, tax_ICU_j,
    taxa_mortalidade_i, taxa_mortalidade_j, contact_matrix, pI,
    infection_to_hospitalization, infection_to_icu, capacidade_UTIs, capacidade_Ward, Normalization_constant,
    pH, pU

    """
    N0 = mp.population
    pI = mp.population_rate_elderly
    Normalization_constant = mp.Normalization_constant[0]
    # Because if the constant be scaled after changing the contact matrix again,
    # it should lose the effect of reducing infection rate
    if mp.IC_analysis == 2:  # mp.analysis == 'Single Run'
        alpha = cp.alpha
        beta = cp.beta
        gamma = cp.gamma
        delta = cp.delta
        taxa_mortalidade_i = cp.mortality_rate_elderly
        taxa_mortalidade_j = cp.mortality_rate_young

        #TODO: LINHAS 258 a 262
        tax_int_i = cp.internation_rate_ward_elderly
        tax_int_j = cp.internation_rate_ward_young

        tax_ICU_i = cp.internation_rate_icu_elderly
        tax_ICU_j = cp.internation_rate_icu_young
    else:  # CONFIDENCE INTERVAL OR SENSITIVITY ANALYSIS
        alpha = cp.alpha[ii]
        taxa_mortalidade_i = cp.mortality_rate_elderly[ii]
        taxa_mortalidade_j = cp.mortality_rate_young[ii]
        beta = cp.beta[ii]
        gamma = cp.gamma[ii]
        delta = cp.delta[ii]

        tax_int_i = cp.internation_rate_ward_elderly[ii]
        tax_int_j = cp.internation_rate_ward_young[ii]

        tax_ICU_i = cp.internation_rate_icu_elderly[ii]
        tax_ICU_j = cp.internation_rate_icu_young[ii]

    contact_matrix = mp.contact_matrix[i]
    # taxa_mortalidade_i = cp.mortality_rate_elderly
    # taxa_mortalidade_j = cp.mortality_rate_young
    pH = cp.pH
    pU = cp.pU

    los_WARD = cp.los_ward
    los_ICU = cp.los_icu

    infection_to_hospitalization = cp.infection_to_hospitalization
    infection_to_icu = cp.infection_to_icu

    proportion_of_ward_mortality_over_total_mortality_elderly = cp.ward_mortality_proportion_elderly
    proportion_of_ward_mortality_over_total_mortality_young = cp.ward_mortality_proportion_young

    proportion_of_icu_mortality_over_total_mortality_elderly = cp.icu_mortality_proportion_elderly
    proportion_of_icu_mortality_over_total_mortality_young = cp.icu_mortality_proportion_young

    # tax_int_i = cp.internation_rate_ward_elderly
    # tax_int_j = cp.internation_rate_ward_young
    #
    # tax_ICU_i = cp.internation_rate_icu_elderly
    # tax_ICU_j = cp.internation_rate_icu_young

    capacidade_UTIs = mp.bed_icu
    capacidade_Ward = mp.bed_ward

    WARD_survive_proportion_i = cp.WARD_survive_proportion_i
    WARD_survive_proportion_j = cp.WARD_survive_proportion_j
    ICU_survive_proportion_i = cp.ICU_survive_proportion_i
    ICU_survive_proportion_j = cp.ICU_survive_proportion_j
    los_WARD_survive_i = cp.los_WARD_survive_i
    los_WARD_survive_j = cp.los_WARD_survive_j
    los_WARD_death_i = cp.los_WARD_death_i
    los_WARD_death_j = cp.los_WARD_death_j
    los_ICU_survive_i = cp.los_ICU_survive_i
    los_ICU_survive_j = cp.los_ICU_survive_j
    los_ICU_death_i = cp.los_ICU_death_i
    los_ICU_death_j = cp.los_ICU_death_j
    los_discharged_ICU_survive_i = cp.los_discharged_ICU_survive_i
    los_discharged_ICU_survive_j = cp.los_discharged_ICU_survive_j

    args = (N0, alpha, beta, gamma, delta,
            los_WARD, los_ICU, tax_int_i, tax_int_j, tax_ICU_i, tax_ICU_j,
            taxa_mortalidade_i, taxa_mortalidade_j, contact_matrix, pI,
            infection_to_hospitalization, infection_to_icu,
            capacidade_UTIs, capacidade_Ward, Normalization_constant, pH, pU,
            proportion_of_ward_mortality_over_total_mortality_elderly,
            proportion_of_ward_mortality_over_total_mortality_young,
            proportion_of_icu_mortality_over_total_mortality_elderly,
            proportion_of_icu_mortality_over_total_mortality_young,
            WARD_survive_proportion_i, WARD_survive_proportion_j,
            ICU_survive_proportion_i, ICU_survive_proportion_j,
            los_WARD_survive_i, los_WARD_survive_j,
            los_WARD_death_i, los_WARD_death_j,
            los_ICU_survive_i, los_ICU_survive_j,
            los_ICU_death_i, los_ICU_death_j,
            los_discharged_ICU_survive_i, los_discharged_ICU_survive_j)
    return args

def derivSEIRHUM(SEIRHUM, t, N0, alpha, beta, gamma, delta,
                 los_WARD, los_ICU, tax_int_i, tax_int_j, tax_ICU_i, tax_ICU_j,
                 taxa_mortalidade_i, taxa_mortalidade_j, contact_matrix, pI,
                 infection_to_hospitalization, infection_to_icu,
                 capacidade_UTIs, capacidade_Ward, Normalization_constant, pH, pU,
                 proportion_of_ward_mortality_over_total_mortality_elderly,
                 proportion_of_ward_mortality_over_total_mortality_young,
                 proportion_of_icu_mortality_over_total_mortality_elderly,
                 proportion_of_icu_mortality_over_total_mortality_young,
                 WARD_survive_proportion_i, WARD_survive_proportion_j,
                 ICU_survive_proportion_i, ICU_survive_proportion_j,
                 los_WARD_survive_i, los_WARD_survive_j,
                 los_WARD_death_i, los_WARD_death_j,
                 los_ICU_survive_i, los_ICU_survive_j,
                 los_ICU_death_i, los_ICU_death_j,
                 los_discharged_ICU_survive_i, los_discharged_ICU_survive_j):
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
    :param los_WARD: average Length Of Stay for wards
    :param los_ICU: average Length Of Stay in ICU beds
    :param tax_int_i: hospitalization rate for elderly in ward beds
    :param tax_int_j: hospitalization rate for young in ward beds
    :param tax_ICU_i: hospitalization rate for elderly in ICU beds
    :param tax_ICU_j: hospitalization rate for young in ICU beds
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

    (Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj, Hi, Hj,
     WARD_excess_i, WARD_excess_j, Ui, Uj, ICU_excess_i, ICU_excess_j, Mi, Mj,
     pHi, pHj, pUi, pUj, pMi, pMj,
     WARD_survive_i, WARD_survive_j,
     WARD_death_i, WARD_death_j,
     ICU_survive_i, ICU_survive_j,
     ICU_death_i, ICU_death_j,
     WARD_discharged_ICU_survive_i,
     WARD_discharged_ICU_survive_j)  = SEIRHUM

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
    dpUi = -tax_ICU_i * dSidt - pUi / infection_to_icu
    dpUj = -tax_ICU_j * dSjdt - pUj / infection_to_icu
    dpMi = 0
    dpMj = 0

    # ## Bed demand - WITH constraints over Ward and ICUs

    # const_dot_balanceWard = (-0.01) * (WARD_survive_i + WARD_survive_j
    #                    + WARD_death_i + WARD_death_j
    #                    + WARD_discharged_ICU_survive_i
    #                    + WARD_discharged_ICU_survive_j
    #                    - capacidade_Ward)
    #
    # const_dot_balanceICU = (-0.06) * (ICU_survive_i + ICU_survive_j
    #                    + ICU_death_i + ICU_death_j
    #                    - capacidade_UTIs)

    # dWARD_survive_idt =  (pHi / infection_to_hospitalization) * WARD_survive_proportion_i \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceWard))) - WARD_survive_i / los_WARD_survive_i
    # dWARD_survive_jdt =  (pHj / infection_to_hospitalization) * WARD_survive_proportion_j \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceWard))) - WARD_survive_j / los_WARD_survive_j
    #
    # dWARD_death_idt =  (pHi / infection_to_hospitalization) * (1 - WARD_survive_proportion_i) \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceWard))) - WARD_death_i / los_WARD_death_i
    # dWARD_death_jdt = (pHj / infection_to_hospitalization) * (1 - WARD_survive_proportion_j) \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceWard))) - WARD_death_j / los_WARD_death_j
    #
    # dICU_survive_idt = (pUi / infection_to_icu) * ICU_survive_proportion_i \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceICU))) - ICU_survive_i / los_ICU_survive_i
    # dICU_survive_jdt =(pUj / infection_to_icu) * ICU_survive_proportion_j \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceICU))) - ICU_survive_j / los_ICU_survive_j
    #
    # dICU_death_idt = (pUi / infection_to_icu) * (1 - ICU_survive_proportion_i) \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceICU))) - ICU_death_i / los_ICU_death_i
    # dICU_death_jdt = (pUj / infection_to_icu) * (1 - ICU_survive_proportion_j) \
    #     * (1 - 1 / (1 + np.exp(const_dot_balanceICU))) - ICU_death_j / los_ICU_death_j
    #
    # dWARD_discharged_ICU_survive_idt = (ICU_survive_i / los_ICU_survive_i) - WARD_discharged_ICU_survive_i / los_discharged_ICU_survive_i
    # dWARD_discharged_ICU_survive_jdt = (ICU_survive_j / los_ICU_survive_j) - WARD_discharged_ICU_survive_j / los_discharged_ICU_survive_j
    #
    # ## Excess beds
    # dWARD_excess_idt = (pHi / infection_to_hospitalization) * (1 / (1 + np.exp(const_dot_balanceWard)))
    # dWARD_excess_jdt = (pHj / infection_to_hospitalization) * (1 / (1 + np.exp(const_dot_balanceWard)))
    #
    # dICU_excess_idt = (pUi / infection_to_icu) * (1 / (1 + np.exp(const_dot_balanceICU)))
    # dICU_excess_jdt = (pUj / infection_to_icu) * (1 / (1 + np.exp(const_dot_balanceICU)))
    #
    # ## DEATHS - WITH constraints over Ward and ICUs
    #
    # dMidt = (WARD_death_i / los_WARD_death_i) + (ICU_death_i / los_ICU_death_i) \
    #     + dWARD_excess_idt * pH + dICU_excess_idt * pU
    # dMjdt = (WARD_death_j / los_WARD_death_j) + (ICU_death_j / los_ICU_death_j) \
    #     + dWARD_excess_jdt * pH + dICU_excess_jdt * pU

    ## Bed demand - NO constraints over Ward and ICUs

    dWARD_survive_idt = (pHi / infection_to_hospitalization) * WARD_survive_proportion_i  - WARD_survive_i / los_WARD_survive_i
    dWARD_survive_jdt = (pHj / infection_to_hospitalization) * WARD_survive_proportion_j  - WARD_survive_j / los_WARD_survive_j

    dWARD_death_idt = (pHi / infection_to_hospitalization) * (1 - WARD_survive_proportion_i)  - WARD_death_i / los_WARD_death_i
    dWARD_death_jdt = (pHj / infection_to_hospitalization) * (1 - WARD_survive_proportion_j)  - WARD_death_j / los_WARD_death_j

    dICU_survive_idt = (pUi / infection_to_icu) * ICU_survive_proportion_i - ICU_survive_i / los_ICU_survive_i
    dICU_survive_jdt = (pUj / infection_to_icu) * ICU_survive_proportion_j - ICU_survive_j / los_ICU_survive_j

    dICU_death_idt = (pUi / infection_to_icu) * (1 - ICU_survive_proportion_i) - ICU_death_i / los_ICU_death_i
    dICU_death_jdt = (pUj / infection_to_icu) * (1 - ICU_survive_proportion_j) - ICU_death_j / los_ICU_death_j

    dWARD_discharged_ICU_survive_idt = (ICU_survive_i / los_ICU_survive_i) - WARD_discharged_ICU_survive_i / los_discharged_ICU_survive_i
    dWARD_discharged_ICU_survive_jdt = (ICU_survive_j / los_ICU_survive_j) - WARD_discharged_ICU_survive_j / los_discharged_ICU_survive_j

    dWARD_excess_idt = 0
    dWARD_excess_jdt = 0

    dICU_excess_idt = 0
    dICU_excess_jdt = 0

    # DEATHS - NO constraints over Ward and ICUs

    dMidt = (WARD_death_i / los_WARD_death_i) + (ICU_death_i / los_ICU_death_i) \
        + dWARD_excess_idt * pH + dICU_excess_idt * pU
    dMjdt = (WARD_death_j / los_WARD_death_j) + (ICU_death_j / los_ICU_death_j) \
        + dWARD_excess_jdt * pH + dICU_excess_jdt * pU

    ## Dummy Count:

    dHidt = dWARD_survive_idt + dWARD_death_idt + dWARD_discharged_ICU_survive_idt
    dHjdt = dWARD_survive_jdt + dWARD_death_jdt + dWARD_discharged_ICU_survive_jdt

    dUidt = dICU_survive_idt + dICU_death_idt
    dUjdt = dICU_survive_jdt + dICU_death_jdt


    return (dSidt, dSjdt, dEidt, dEjdt, dIidt, dIjdt, dRidt, dRjdt,
            dHidt, dHjdt, dWARD_excess_idt, dWARD_excess_jdt, dUidt, dUjdt, dICU_excess_idt, dICU_excess_jdt, dMidt, dMjdt,
            dpHi, dpHj, dpUi, dpUj, dpMi, dpMj,
            dWARD_survive_idt, dWARD_survive_jdt,
            dWARD_death_idt, dWARD_death_jdt,
            dICU_survive_idt, dICU_survive_jdt,
            dICU_death_idt, dICU_death_jdt,
            dWARD_discharged_ICU_survive_idt,
            dWARD_discharged_ICU_survive_jdt)
