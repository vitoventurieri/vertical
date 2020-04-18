import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

def run_SEIR_ODE_model(
        N: 'population size',
        Ev0: 'init. exposed population',
        Ej0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        beta: 'infection probability',
        gamma: 'removal probability', 
        omega: 'atenuating factor for beta. Example omega = 0.4 equals a 60% reduction in contacts for the group of study', 
        alpha_inv: 'incubation period', 
        losleito_inv: 'tempo de permanecia no leito normal', 
        losuti_inv: 'tempo de permanecia na UTI', 
        tax_intV: 'taxa de internação de pacientes Idosos em leito normal', 
        tax_utiV: 'taxa de internação de pacientes Idosos em leito de UTI', 
        tax_intJ: 'taxa de internação de pacientes Jovens em leito normal', 
        tax_utiJ: 'taxa de internação de pacientes Jovens em leito de UTI', 
        t_max: 'numer of days to run'
    ) -> pd.DataFrame:
	
	#constantes
    percentual_pop_idosa = 0.2
	
    V0 = (N - I0 - R0 - Ej0 -Ev0)*percentual_pop_idosa
    J0 = (N - I0 - R0 - Ej0 -Ev0)*(1-percentual_pop_idosa)
    HJ0 = I0*tax_intJ*(1-percentual_pop_idosa)
    UJ0 = I0*tax_utiV*(1-percentual_pop_idosa)
    HV0 = I0*tax_intV*percentual_pop_idosa
    UV0 = I0*tax_utiV*percentual_pop_idosa
    alpha = 1/alpha_inv
    losleito_inv = 1/losleito
    losuti_inv = 1/losuti


    # A grid of time points (in days)
    t = range(t_max)

    # The SEIR model differential equations.
    def deriv(y, t, N, beta, gamma, omega, alpha, losleito_inv, losuti_inv, tax_intV, tax_utiV, tax_intJ, tax_utiJ):
     #   S, E, I, R = y
     #   dSdt = -beta * S * I / N
     #   dEdt = -dSdt - alpha*E
     #   dIdt = alpha*E - gamma*I
     #   dRdt = gamma * I
     
        J, V, Ev, Ej, I, R, HV, UV, HJ, UJ = y
        dVdt = -beta * omega * V * I / N
        dJdt = -beta * J * I / N
        dEvdt = - dVdt - alpha*Ev
        dEjdt = - dJdt - alpha*Ej
        dIdt = alpha*Ej + alpha*Ev - gamma*I
        dRdt = gamma * I
        dHVdt = tax_intV*alpha*Ej -losleito_inv*HV
        dUVdt = tax_utiV*alpha*Ej -losuti_inv*UV
        dHJdt = tax_intJ*alpha*Ev -losleito_inv*HJ
        dUJdt = tax_utiJ*alpha*Ev -losuti_inv*UJ

        return dJdt, dVdt, dEvdt, dEjdt, dIdt, dRdt, dHVdt, dUVdt, dHJdt, dUJdt

    # Initial conditions vector
    y0 = J0, V0, Ej0, Ev0, I0, R0, HV0, UV0, HJ0, UJ0
    print (*y0)

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, omega, alpha, losleito_inv, losuti_inv, tax_intV, tax_utiV, tax_intJ, tax_utiJ))
    J, V, Ev, Ej, I, R, HV, UV, HJ, UJ = ret.T
    

    return pd.DataFrame({'J': J,'V': V, 'Ev': Ev, 'Ej': Ej, 'I': I, 'R': R, 'HV': HV, 'UV': UV, 'HJ': HJ, 'UJ': UJ,}, index=t)
    DataFrame


if __name__ == '__main__':
    N = 200000000
	#Parametros iniciais e constantes
    Ev0, Ej0, I0, R0 = 1, 1, 1, 0
	#Digite o R0 desejado
    ErreZero = 2.3
	#Digite o periodo que doença infecta os outros
    tempo_de_infeciosidade = 10
    gamma = 1/tempo_de_infeciosidade
    beta = ErreZero*gamma
    alpha_inv = 5
    omega = 0.6
    losleito = 8.9
    losuti = 8
	# taxas usando limite inferior do CDC, ajustadas por idade da piramide etaria brasileira  Fonte: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm?s_cid=mm6912e2_w#T1_down
    tax_intV = 0.252
    tax_utiV = 0.068
    tax_intJ = 0.102
    tax_utiJ = 0.017
    t_max = 2*365
    results = run_SEIR_ODE_model(N, Ev0, Ej0, I0, R0, beta, gamma, omega, alpha_inv, losleito, losuti, tax_intV, tax_utiV, tax_intJ, tax_utiJ, t_max)
    results.to_csv('Modelo_idade_omega0ponto6.csv', index=False)

    # plot
    plt.style.use('ggplot')
    (results
     # .div(1_000_000)
     [['Ev', 'Ej', 'I', 'R', 'V', 'J', 'HV', 'UV', 'HJ', 'UJ' ]]
     .plot(figsize=(8,6), fontsize=20, logy=False))
    params_title = (
        f'SEIR($\gamma$={gamma}, $\\beta$={beta}, $\\alpha$={1/alpha_inv}, $N$={N}, $\\omega$={omega} '
        f'$E_0$={Ej0}, $I_0$={I0}, $R_0$={R0})'
    )
    plt.title(f'Numero de Pessoas Atingidas com modelo:\n' + params_title,
              fontsize=20)
    plt.legend(['Expostas Idosas', 'Expostas Jovens', 'Infectadas', 'Recuperadas', 'Idosos', 'Jovens', 'Leito normal idosos', 'Uti idosos', 'Leito normal jovens', 'Uti jovens'], fontsize=20)
    plt.xlabel('Dias', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    plt.show()