import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import odeint

# The SEIR model differential equations.
def Euler(SEIR_0, t, params):
    """
    Computes the Derivatives by Semi Implicit Euler Method
    :return:
    """

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


def HUM_analysis(SEIR, t, covid_parameters):
    """
    Provides H (ward) U (ICU) M (deaths) variables, in a post-processment
    :return:
    """
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

    # print(Ei[:10])

    Hi, Hj, Ui, Uj, Mi, Mj = [Hi0], [Hj0], [Ui0], [Uj0], [Mi0], [Mj0]
    # Ei, Ej, Ii, Ij = [Ei], [Ej], [Ii], [Ij]
    dt = t[1] - t[0]
    for i in t[1:]:
        # Leitos Normais demandados
        dHidt = tax_int_i * alpha * Ei[i-1] - Hi[i-1] / (los_leito)
        dHjdt = tax_int_j * alpha * Ej[i-1] - Hj[i-1] / (los_leito)
        # Leitos UTIs demandados
        dUidt = tax_uti_i * alpha * Ei[i-1] - Ui[i-1] / (los_uti)
        dUjdt = tax_uti_j * alpha * Ej[i-1] - Uj[i-1] / (los_uti)
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


def run_SEIR_ODE_model(demograph_parameters, covid_parameters, model_parameters) -> pd.DataFrame:
    """
    Runs the simulation
    """
    # from scipy.integrate import odeint

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
    # dt = .1
    # t = np.linspace(0, t_max, int(t_max/dt) + 1)

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

    # Integrate the SEIR equations over the time grid, t
    # ret = odeint(derivSEIR, SEIR_0, t, args)
    # Calculate the variables by Euler Semi Implicit Method
    ret = Euler(SEIR_0, t, args)
    # Update the variables
    Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj = ret.T
    SEIR = Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj

    # POST PROCESS to obtain the hospital demand (ward and ICUs) and deaths
    HUM = HUM_analysis(SEIR, t, cp)

    Hi, Hj, Ui, Uj, Mi, Mj = HUM

    #print(Hi[:10])
    print('Maximo de obitos idosos em um unico dia: %d' % max(Mi))
    print('Maximo de obitos jovens em um unico dia: %d' % max(Mj))
    print('Total de obitos: %d' % sum(Mi + Mj))

    health_system_colapse_identifier(Hi, Hj, Ui, Uj, dp, mp)

    df = pd.DataFrame({'Si': Si, 'Sj': Sj, 'Ei': Ei, 'Ej': Ej, 'Ii': Ii, 'Ij': Ij, 'Ri': Ri, 'Rj': Rj,
                       'Hi': Hi, 'Hj': Hj, 'Ui': Ui, 'Uj': Uj, 'Mi': Mi, 'Mj': Mj}, index=t)

    return df


def health_system_colapse_identifier(Hi, Hj, Ui, Uj, dp, mp):
    """
    Performs a post_processing analysis,
    forecast the date to a load of the health system for 30,50,80,100 %
    considers the inital date as today.
    """
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
    print('Dias para atingir 30, 50, 80, 100% da capacidade de leitos comuns')
    print(dia_colapso_leitos)

    dia_colapso_UTIs_30  = np.min(np.where(U > capacidade_UTIs*lotacao[0]))
    dia_colapso_UTIs_50  = np.min(np.where(U > capacidade_UTIs*lotacao[1]))
    dia_colapso_UTIs_80  = np.min(np.where(U > capacidade_UTIs*lotacao[2]))
    dia_colapso_UTIs_100 = np.min(np.where(U > capacidade_UTIs*lotacao[3]))
    dia_colapso_UTIs = (dia_colapso_UTIs_30, dia_colapso_UTIs_50,
                        dia_colapso_UTIs_80,dia_colapso_UTIs_100)
    print('Dias para atingir 30, 50, 80, 100% da capacidade de UTIs')
    print(dia_colapso_UTIs)

    # TimeSeries
    datelist = [d.strftime('%d/%m/%Y')
            for d in pd.date_range(datetime.today(), periods = t_max)]
        #for d in pd.date_range(start = '26/2/2020', periods = t_max)]

    print('Dia em que atinge 30, 50, 80, 100% capacidade de leitos comuns')

    print(datelist[dia_colapso_leitos[0]])
    print(datelist[dia_colapso_leitos[1]])
    print(datelist[dia_colapso_leitos[2]])
    print(datelist[dia_colapso_leitos[3]])

    print('Dia em que atinge 30, 50, 80, 100% capacidade de UTIs')

    print(datelist[dia_colapso_UTIs[0]])
    print(datelist[dia_colapso_UTIs[1]])
    print(datelist[dia_colapso_UTIs[2]])
    print(datelist[dia_colapso_UTIs[3]])

def derivSEIR(SEIR, t, N, alpha, beta, gamma, omega_i, omega_j):
    """
    Calculate the derivatives for the odeint function
    """
    # Vetor variaveis incognitas
    Si, Sj, Ei, Ej, Ii, Ij, Ri, Rj = SEIR
    
    dSidt = - beta * omega_i * Si * (Ii + Ij) / N
    dSjdt = - beta * omega_j * Sj * (Ii + Ij) / N
    dEidt = - dSidt - alpha * Ei
    dEjdt = - dSjdt - alpha * Ej
    dIidt = alpha * Ei - gamma * Ii
    dIjdt = alpha * Ej - gamma * Ij
    dRidt = gamma * Ii
    dRjdt = gamma * Ij
    
    return dSidt, dSjdt, dEidt, dEjdt, dIidt, dIjdt, dRidt, dRjdt
	
