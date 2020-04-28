import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
import dask.bag as db

'''
Adds confidence interval for the parameters
divided into 8 compartments (S,E,I,R for elderly (60+) and young ones (0-60))
omega as attenuating contact factor (as social isolation)
calculate daily demand for hospitalization (ward=H, icu = U)
estimate cumulative daily death (M)
plots as 5% quantile, median and 95% quantile
based on bayes_seir code from original COVID-19 repo
'''

DEFAULT_PARAMS = {
    'fator_subr': 1, #40.0,
	
	
	
## SE DER MELHORAR DADOS ICU
# ECDC: https://www.ecdc.europa.eu/en/covid-19/questions-answers at 23 Apr 2020
# CDC USA: http://dx.doi.org/10.15585/mmwr.mm6912e2 
# LI: http://dx.doi.org/10.1056/NEJMoa2001316
# VERITY: http://dx.doi.org/10.1016/s1473-3099(20)30243-7
# ZHOU: http://dx.doi.org/10.1016/s0140-6736(20)30566-3 (Table 2)

## CONFERIR SE SAO PRA LOGNORMAL
    # these are 95% confidence intervals
    # for a lognormal 
    'gamma': (7.0, 12.0),# days, infectivity period, ECDC 
    'alpha': (4.1, 7.0), # days, incubation period, LI
    'R0_': (1.4, 3.9), # [], basic reproduction number, LI
    'ward_internation_rate_e': (0.0610, 0.2089), # percentage, AJUSTE VERITY IBGE 
    'icu_internation_rate_e': (0.0235, 0.0804), # percentage, AJUSTE VERITY IBGE x FATOR CDC USA U/(H+U)=1/3.6
    'ward_internation_rate_y': (0.0124, 0.0426), # percentage, AJUSTE VERITY IBGE
    'icu_internation_rate_y': (0.0031, 0.0107), # percentage, AJUSTE VERITY IBGE x FATOR CDC USA U/(H+U)=1/5
    'ward_LOS': (4.0, 12.0), # ZHOU
    'icu_LOS': (7.0, 14.0), # ZHOU
    'mortality_rate_e': (0.019060, 0.066335), # percentage, AJUSTE VERITY IBGE
    'mortality_rate_y': (0.000681, 0.002851) # percentage, AJUSTE VERITY IBGE
}



def make_lognormal_params_95_ci(lb, ub):
	mean = (ub*lb)**(1/2)
	std = (ub/lb)**(1/4)
	
	
	# http://broadleaf.com.au/resource-material/lognormal-distribution-summary/
	# v1 = ub
	# v2 = lb
	# z1 = 1.96
	# z2 = 1.96
	# std = log( v1 / v2 ) / (z1 - z2)
	# mu = ( z2 * log(v1) - z1 * log(v2)) / (z2 - z1)
	
	return mean, std



def initial_conditions(
		N: 'population size',
		pE: 'elderly proportion',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
    	M0: 'init. deacesed population',
        fator_subr: 'subreporting factor, multiples I0 and E0'
	):


	## TALVEZ TROCAR CONDICAO INICIAL EQUIVALENTE AO NUMERO DE OBITOS ACUMULADOS DO MS DO DIA SIMULADO
	## APOS SIMULAR CASO COM UM INFECTADO E r0 EXPOSTOS
	
	## POR QUE SUB_REPORT NAO MULTIPLICA R0 ?????????

	## DEFINIR INTERVALO DE CONFIANCA PARA AS CONDICOES INICIAIS ?????????
    Ee0 = fator_subr * E0 * pE
    Ie0 = fator_subr * I0 * pE
    Re0 = fator_subr * R0 * pE
    Se0 = N * pE - (Ie0 + Re0 + Ee0)
    
    Ey0 = fator_subr * E0 * (1 - pE)
    Iy0 = fator_subr * I0 * (1 - pE)
    Ry0 = fator_subr * R0 * (1 - pE)
    Sy0 = N * (1 - pE) - (Iy0 + Ry0 + Ey0)


   ## 4) USAR COMO CONDICAO INICIAL PARA M, DADOS DO MIN SAUDE, 
   ## DISTRIBUIDOS PELA PROPORCAO IDOSA OU COM OUTRO AJUSTE ??????????????????????	
    Me0 = pE * M0
    My0 = (1-pE) * M0
	
    return Se0, Ee0, Ie0, Re0, Sy0, Ey0, Iy0, Ry0, Me0, My0

def run_SEIR_BAYES_model(
		N: 'population size',
		Se0: 'init. exposed population',
		Ee0: 'init. exposed population',
		Ie0: 'init. infected population',
		Re0: 'init. removed population',
		Sy0: 'init. exposed population',
		Ey0: 'init. exposed population',
		Iy0: 'init. infected population',
		Ry0: 'init. removed population',
		Me0: 'init. deacesed population',
		My0: 'init. deacesed population',
		omega_e: 'attenuating contact factor',
		omega_y: 'attenuating contact factor',
		ward_internation_rate_e: 'ward_internation_rate_e',
		icu_internation_rate_e: 'icu_internation_rate_e',
		ward_internation_rate_y: 'ward_internation_rate_y',
		icu_internation_rate_y: 'icu_internation_rate_y',
		ward_LOS: 'ward_LOS',
		icu_LOS: 'icu_LOS',
		mortality_rate_e: 'mortality_rate_e',
		mortality_rate_y: 'mortality_rate_y',
		R0__params: 'repr. rate mean and std',
		gamma_inv_params: 'removal rate mean and std',
		alpha_inv_params: 'incubation rate mean and std',
		t_max: 'number of days to run',
		runs: 'number of runs'
    ):


	t_space = np.arange(0, t_max)
	
	size = (t_max, runs)
	
	Se = np.zeros(size)
	Ee = np.zeros(size)
	Ie = np.zeros(size)
	Re = np.zeros(size)
	Sy = np.zeros(size)
	Ey = np.zeros(size)
	Iy = np.zeros(size)
	Ry = np.zeros(size)
	
	Se[0, ], Ee[0, ], Ie[0, ], Re[0, ] = Se0, Ee0, Ie0, Re0
	Sy[0, ], Ey[0, ], Iy[0, ], Ry[0, ] = Sy0, Ey0, Iy0, Ry0
	
	R0_ = npr.lognormal(*map(np.log, R0__params), runs)
	gamma = 1/npr.lognormal(*map(np.log, gamma_inv_params), runs)
	alpha = 1/npr.lognormal(*map(np.log, alpha_inv_params), runs)
	beta = R0_*gamma
	
	ward_internation_rate_e = npr.lognormal(*map(np.log, ward_internation_rate_e), runs)
	icu_internation_rate_e = npr.lognormal(*map(np.log, icu_internation_rate_e), runs)
	ward_internation_rate_y = npr.lognormal(*map(np.log, ward_internation_rate_y), runs)
	icu_internation_rate_y = npr.lognormal(*map(np.log, icu_internation_rate_y), runs)
	ward_LOS = npr.lognormal(*map(np.log, ward_LOS), runs)
	icu_LOS = npr.lognormal(*map(np.log, icu_LOS), runs)
	mortality_rate_e = npr.lognormal(*map(np.log, mortality_rate_e), runs)
	mortality_rate_y = npr.lognormal(*map(np.log, mortality_rate_y), runs)
	
	## 1) ONDE DEFINIR AS CONDICOES INICIAIS DE H, U COM A MEDIA OU COM A DISTRIBUICAO DOS PARAMS ??????????????????????
	## 2) CONTABILIZAR ALTAS COMO LOS NAS CONDICOES INICIAIS DE H, U ??????????????????????
	He0 = Ie0 * ward_internation_rate_e
	Hy0 = Iy0 * ward_internation_rate_y
	Ue0 = Ie0 * icu_internation_rate_e
	Uy0 = Iy0 * icu_internation_rate_y
	
	He = np.zeros(size)
	Ue = np.zeros(size)
	Me = np.zeros(size)
	Hy = np.zeros(size)
	Uy = np.zeros(size)
	My = np.zeros(size)
		
	He[0, ], Ue[0, ], Me[0, ] = He0, Ue0, Me0
	Hy[0, ], Uy[0, ], My[0, ] = Hy0, Uy0, My0
	
	for t in t_space[1:]:



		# beta * omega * (Ie+Iy)/N * Se
		SEe = npr.binomial(Se[t-1, ].astype('int'), expon(scale=1/(beta * omega_e * (Ie[t-1, ]+Iy[t-1, ])/N)).cdf(1))
		# E * alpha
		EIe = npr.binomial(Ee[t-1, ].astype('int'), expon(scale=1/alpha).cdf(1))
		# I * gamma
		IRe = npr.binomial(Ie[t-1, ].astype('int'), expon(scale=1/gamma).cdf(1))
		
		SEy = npr.binomial(Sy[t-1, ].astype('int'), expon(scale=1/(beta * omega_y * (Ie[t-1, ]+Iy[t-1, ])/N)).cdf(1))
		EIy = npr.binomial(Ey[t-1, ].astype('int'), expon(scale=1/alpha).cdf(1))
		IRy = npr.binomial(Iy[t-1, ].astype('int'), expon(scale=1/gamma).cdf(1))
		
		
		# TIRAR E e POR EI
		# E * alpha * internation
		EHe = npr.binomial(Ee[t-1, ].astype('int'), expon(scale=1/(alpha*ward_internation_rate_e)).cdf(1))
		EUe = npr.binomial(Ee[t-1, ].astype('int'), expon(scale=1/(alpha*icu_internation_rate_e)).cdf(1))
		
		## 3) PRECISA INVERTER LOS OU PODE USAR COMO NUMERADOR NO SCALE ??????????????????????
		# H / LOS
		HXe = npr.binomial(He[t-1, ].astype('int'), expon(scale=ward_LOS).cdf(1))
		UXe = npr.binomial(Ue[t-1, ].astype('int'), expon(scale=icu_LOS).cdf(1))
		
		
		EHy = npr.binomial(Ey[t-1, ].astype('int'), expon(scale=1/(alpha*ward_internation_rate_y)).cdf(1))
		EUy = npr.binomial(Ey[t-1, ].astype('int'), expon(scale=1/(alpha*icu_internation_rate_y)).cdf(1))
		
		HXy = npr.binomial(Hy[t-1, ].astype('int'), expon(scale=ward_LOS).cdf(1))
		UXy = npr.binomial(Uy[t-1, ].astype('int'), expon(scale=icu_LOS).cdf(1))
		
		# I * gamma * mortality
		IMe = npr.binomial(Ie[t-1, ].astype('int'), expon(scale=1/(gamma*mortality_rate_e)).cdf(1))
		IMy = npr.binomial(Iy[t-1, ].astype('int'), expon(scale=1/(gamma*mortality_rate_y)).cdf(1))
		
		dSe =  0 - SEe
		dEe = SEe - EIe
		dIe = EIe - IRe
		dRe = IRe - 0
		
		dSy =  0 - SEy
		dEy = SEy - EIy
		dIy = EIy - IRy
		dRy = IRy - 0
		
		dHe = EHe - HXe
		dUe = EUe - UXe
		dMe = IMe
		
		dHy = EHy - HXy
		dUy = EUy - UXy
		dMy = IMy
		
		Se[t, ] = Se[t-1, ] + dSe
		Ee[t, ] = Ee[t-1, ] + dEe
		Ie[t, ] = Ie[t-1, ] + dIe
		Re[t, ] = Re[t-1, ] + dRe
		
		Sy[t, ] = Sy[t-1, ] + dSy
		Ey[t, ] = Ey[t-1, ] + dEy
		Iy[t, ] = Iy[t-1, ] + dIy
		Ry[t, ] = Ry[t-1, ] + dRy
		
		
		He[t, ] = He[t-1, ] + dHe
		Ue[t, ] = Ue[t-1, ] + dUe
		Me[t, ] = Me[t-1, ] + dMe
		
		Hy[t, ] = Hy[t-1, ] + dHy
		Uy[t, ] = Uy[t-1, ] + dUy
		My[t, ] = My[t-1, ] + dMy

    
	return Se, Ee, Ie, Re, Sy, Ey, Iy, Ry, He, Ue, Me, Hy, Uy, My, t_space


# def seir_bayes_plot(N, pE, omega_e, omega_y,
                    # ward_internation_rate_e_params, ward_internation_rate_y_params,
					# icu_internation_rate_e_params, icu_internation_rate_y_params,
					# ward_LOS_params, icu_LOS_params,
					# mortality_rate_e_params, mortality_rate_y_params,
					# R0__params,
                    # gamma_inv_params,
                    # alpha_inv_params,
                    # t_max, runs, Se, Ee, Ie, Re, Sy, Ey, Iy, Ry, He, Ue, Me, Hy, Uy, My, t_space):
    
    # S0 = Se[0, 0] + Sy[0, 0]
    # E0 = Ee[0, 0] + Ey[0, 0]
    # I0 = Ie[0, 0] + Iy[0, 0]
    # R0 = Re[0, 0] + Ry[0, 0]
    
    # S = Se + Sy
    # E = Ee + Ey
    # I = Ie + Iy
    # R = Re + Ry
    
    
    # H0 = He[0, 0] + Hy[0, 0]
    # U0 = Ue[0, 0] + Uy[0, 0]
    # M0 = Me[0, 0] + My[0, 0]
        
    # H = He + Hy
    # U = Ue + Uy
    # M = Me + My


    # # plot
    # algorithm_text = (
        # f"for {runs} runs, do:\n"
		# f"\t$S_0={S0}$\n\t$E_0={E0}$\n\t$I_0={I0}$\n\t$R_0={R0}$\n"
        # #f"\t$S_{{e0}}={Se0}$\n\t$E_{{e0}}={Ee0}$\n\t$I_{{e0}}={Ie0}$\n\t$R_{{e0}}={Re0}$\n"
        # #f"\t$S_{{y0}}={Sy0}$\n\t$E_{{y0}}={Ey0}$\n\t$I_{{y0}}={Iy0}$\n\t$R_{{y0}}={Ry0}$\n"
         # "\t$\\gamma \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$\\alpha \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$R0 \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$ward_e \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$ward_y  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$icu_e \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$icu_y  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$LOS_w \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$LOS_i  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$d_e \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
         # "\t$d_y  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"		 
        # f"\t$\\beta = \\gamma R0$\n"
        # f"\tSolve SEIR$(\\alpha, \\gamma, \\beta)$"
    # ).format(*gamma_inv_params, *alpha_inv_params, *R0__params,
	# *ward_internation_rate_e_params, *ward_internation_rate_y_params,
	# *icu_internation_rate_e_params, *icu_internation_rate_y_params,
	# *ward_LOS_params, *icu_LOS_params,
	# *mortality_rate_e_params, *mortality_rate_y_params)

    # title = '(RESULTADO PRELIMINAR) Pessoas afetadas pelo COVID-19, segundo o modelo SEIR-Bayes'
    # plt.style.use('ggplot')
    # fig, ax = plt.subplots(figsize=(16,9))
    # plt.plot(t_space, E.mean(axis=1), '--', t_space, I.mean(axis=1), '--', marker='o')
    # plt.title(title, fontsize=20)
    # plt.legend(['Expostas ($\mu \pm \sigma$)',
                # 'Infectadas ($\mu \pm \sigma$)'],
               # fontsize=20, loc='lower right')
    # plt.xlabel('t (Dias a partir de 20/Abril/2020)', fontsize=20)
    # plt.ylabel('Pessoas', fontsize=20)
    # plt.fill_between(t_space,
                     # I.mean(axis=1) + I.std(axis=1), 
                     # (I.mean(axis=1) - I.std(axis=1)).clip(I0),
                     # color='b', alpha=0.2)
    # plt.fill_between(t_space, 
                     # E.mean(axis=1) + E.std(axis=1), 
                     # (E.mean(axis=1) - E.std(axis=1)).clip(I0),
                     # color='r', alpha=0.2)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.05, 0.95, algorithm_text,
            # transform=ax.transAxes, fontsize=14,
            # verticalalignment='top', bbox=props)
    # plt.yscale('log')
    # return fig



def seir_bayes_plot(N, pE, omega_e, omega_y,
					ward_internation_rate_e_params, ward_internation_rate_y_params,
					icu_internation_rate_e_params, icu_internation_rate_y_params,
					ward_LOS_params, icu_LOS_params,
					mortality_rate_e_params, mortality_rate_y_params,
					R0__params,
					gamma_inv_params,
					alpha_inv_params,
					t_max, runs, Se, Ee, Ie, Re, Sy, Ey, Iy, Ry, He, Ue, Me, Hy, Uy, My, t_space):
	
	S0 = Se[0, 0] + Sy[0, 0]
	E0 = Ee[0, 0] + Ey[0, 0]
	I0 = Ie[0, 0] + Iy[0, 0]
	R0 = Re[0, 0] + Ry[0, 0]
	
	S = Se + Sy
	E = Ee + Ey
	I = Ie + Iy
	R = Re + Ry
	
	
	H0 = He[0, 0] + Hy[0, 0]
	U0 = Ue[0, 0] + Uy[0, 0]
	M0 = Me[0, 0] + My[0, 0]
		
	H = He + Hy
	U = Ue + Uy
	M = Me + My
	
	
	# plot
	algorithm_text = (
		f"for {runs} runs, do:\n"
		f"\t$S_0={S0}$\n\t$E_0={E0}$\n\t$I_0={I0}$\n\t$R_0={R0}$\n"
		#f"\t$S_{{e0}}={Se0}$\n\t$E_{{e0}}={Ee0}$\n\t$I_{{e0}}={Ie0}$\n\t$R_{{e0}}={Re0}$\n"
		#f"\t$S_{{y0}}={Sy0}$\n\t$E_{{y0}}={Ey0}$\n\t$I_{{y0}}={Iy0}$\n\t$R_{{y0}}={Ry0}$\n"
		"\t$\\gamma \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$\\alpha \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$R0 \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$ward_e \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$ward_y  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$icu_e \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$icu_y  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$LOS_w \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$LOS_i  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$d_e \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"
		"\t$d_y  \\sim LogNormal(\mu={:.04}, \\sigma={:.04})$\n"		 
		f"\t$\\beta = \\gamma R0$\n"
		f"\tSolve SEIR$(\\alpha, \\gamma, \\beta)$"
	).format(*gamma_inv_params, *alpha_inv_params, *R0__params,
	*ward_internation_rate_e_params, *ward_internation_rate_y_params,
	*icu_internation_rate_e_params, *icu_internation_rate_y_params,
	*ward_LOS_params, *icu_LOS_params,
	*mortality_rate_e_params, *mortality_rate_y_params)
	
	title = '(PRELIMINAR) Hospitalizations and deaths by COVID-19'
	plt.style.use('ggplot')
	fig, ax = plt.subplots(figsize=(16,9))
	plt.plot(t_space, np.quantile(H,0.5,axis=1), '-.', color = 'b')#, marker='^')
	plt.plot(t_space, np.quantile(U,0.5,axis=1), '-', color = 'r')#, marker='o')
	plt.plot(t_space, np.quantile(M,0.5,axis=1), '--', color = 'g')#, marker='s')
		#plt.plot(t_space, M.mean(axis=1), '--', color = 'b', marker='s')
	plt.title(title, fontsize=20)
	#plt.legend(['Death ($\mu \pm \sigma$)'],
	plt.legend(['Daily ward demand', 'Daily ICU demand', 'Cumulative death'],
			title = '5% quantile, median, 95% quantile',
			fontsize=14, loc='lower right')
	plt.xlabel('t (Days from 20/Apr/2020)', fontsize=20)
	plt.ylabel('Amount', fontsize=20)
	plt.fill_between(t_space,
			np.quantile(H,0.05,axis=1), 
			np.quantile(H,0.95,axis=1).clip(H0),
			color='b', alpha=0.2)
	plt.fill_between(t_space,
			np.quantile(U,0.05,axis=1), 
			np.quantile(U,0.95,axis=1).clip(U0),
			color='r', alpha=0.2)
	plt.fill_between(t_space,
					np.quantile(M,0.05,axis=1), 
					np.quantile(M,0.95,axis=1).clip(M0),
					color='g', alpha=0.2)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.05, 0.95, algorithm_text,
			transform=ax.transAxes, fontsize=14,
			verticalalignment='top', bbox=props)
	plt.yscale('log')
	return fig


if __name__ == '__main__':
	
	# PROJECAO IBGE 2020 at https://www.ibge.gov.br/apps/populacao/projecao/ at 11th Apr 2020	
	N = 211_755_692  # IBGE
	pE = 0.1425 # 60+ IBGE
	E0, I0, R0, M0 = 260_000, 304_000, 472_000, 3_000
	R0__params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['R0_'])
	gamma_inv_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['gamma'])
	alpha_inv_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['alpha'])
	
	ward_internation_rate_e_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['ward_internation_rate_e'])
	ward_internation_rate_y_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['ward_internation_rate_y'])
	icu_internation_rate_e_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['icu_internation_rate_e'])
	icu_internation_rate_y_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['icu_internation_rate_y'])
	ward_LOS_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['ward_LOS'])
	icu_LOS_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['icu_LOS'])
	mortality_rate_e_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['mortality_rate_e'])
	mortality_rate_y_params = make_lognormal_params_95_ci(*DEFAULT_PARAMS['mortality_rate_y'])
	
	
	fator_subr = DEFAULT_PARAMS['fator_subr']
	omega_e = 1
	omega_y = 1
	t_max = 30*6
	runs = 1_000
	
	Se0, Ee0, Ie0, Re0, Sy0, Ey0, Iy0, Ry0, Me0, My0 = initial_conditions(
													N, pE, E0, I0, R0, M0, fator_subr)
	
	
	Se, Ee, Ie, Re, Sy, Ey, Iy, Ry, He, Ue, Me, Hy, Uy, My, t_space = run_SEIR_BAYES_model(
									N, Se0, Ee0, Ie0, Re0, Sy0, Ey0, Iy0, Ry0, Me0, My0,
									omega_e, omega_y,
									ward_internation_rate_e_params, ward_internation_rate_y_params,
									icu_internation_rate_e_params, icu_internation_rate_y_params,
									ward_LOS_params, icu_LOS_params,
									mortality_rate_e_params, mortality_rate_y_params,
									R0__params,
									gamma_inv_params,
									alpha_inv_params,
									t_max, runs)
	
	
	## NAO SEI PQ TEM ESSE COMPORTAMENTO DE ZERAR NO t = 50
	## NAO MANJO DE MULTIPLOS PLOTS, EXPORTARIA CADA FIGURA UMA A UMA, POR ORA TEM 3
	## 1) EXPOSTOS E INFECTADOS; 2) DEMANDA LEITOS COMUNS E UTIS; 3) OBITOS
	fig = seir_bayes_plot(N, pE, omega_e, omega_y,
					ward_internation_rate_e_params, ward_internation_rate_y_params,
					icu_internation_rate_e_params, icu_internation_rate_y_params,
					ward_LOS_params, icu_LOS_params,
					mortality_rate_e_params, mortality_rate_y_params,
					R0__params,
					gamma_inv_params,
					alpha_inv_params,
					t_max, runs, Se, Ee, Ie, Re, Sy, Ey, Iy, Ry, He, Ue, Me, Hy, Uy, My, t_space)
	plt.show()
