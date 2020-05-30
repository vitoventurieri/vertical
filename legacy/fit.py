# -*- coding: utf-8 -*-
"""

@author: Fuck
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import datetime


# INPUT

state_name = 'São Paulo'

metodo = "subreport" # "fator_verity" #



# IMPORT DATA
df = pd.read_excel (r'C:\Users\Fuck\Downloads\HIST_PAINEL_COVIDBR_21mai2020.xlsx')
# data	semanaEpi	populacaoTCU2019	casosAcumulado	obitosAcumulado	Recuperadosnovos	emAcompanhamentoNovos
states = { 'coduf': [76, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53],
		'state_name': ['Brasil','Rondônia','Acre','Amazonas','Roraima','Pará','Amapá','Tocantins','Maranhão','Piauí','Ceará','Rio Grande do Norte','Paraíba','Pernambuco','Alagoas','Sergipe','Bahia','Minas Gerais','Espiríto Santo','Rio de Janeiro','São Paulo','Paraná','Santa Catarina','Rio Grande do Sul','Mato Grosso do Sul','Mato Grosso','Goiás','Distrito Federal'],
		'populationTCU2019': [210_147_125, 1_777_225, 881_935, 4_144_597, 605_761, 8_602_865, 845_731, 1_572_866, 7_075_181, 3_273_227, 9_132_078, 3_506_853, 4_018_127, 45_919_049, 3_337_357, 2_298_696, 14_873_064, 21_168_791, 4_018_650, 17_264_943, 7_164_788, 9_557_071, 11_433_957, 11_377_239, 2_778_986, 3_484_466, 7_018_354, 3_015_268]}
states = pd.DataFrame(states, columns = ['coduf', 'state_name', 'populationTCU2019'])

# INITIAL DATE
if state_name == 'Pernambuco':
	startdate = '2020-05-02'
	r0 = (1.1, 1.3)
	coduf = 26 
	# population = 9_557_071
elif state_name == 'Santa Catarina':
	startdate = '2020-05-10'
	r0 = (1.1, 1.2)
	coduf = 42
	# population = 7_164_788
elif state_name == 'São Paulo':
	startdate = '2020-04-29'
	r0 = (1.15, 1.32)
	coduf = 35
	# population = 45_919_049
elif state_name == 'Brasil':
	startdate = '2020-05-19'
	coduf = 76
	# population = 210_147_125

states_set = states[states['coduf'] == coduf ]
population = states_set['populationTCU2019'].values[0]
				
dfMS = df[df['coduf'] == coduf ]
dfMS = dfMS[dfMS['codmun'].isnull()] # only rows without city

dfMS['data'] = pd.to_datetime(dfMS['data'])
dfMS['obitosAcumulado'] = pd.to_numeric(dfMS['obitosAcumulado'])

M0_MS = dfMS['obitosAcumulado'].max() # most recent cumulative deaths
R0_MS = dfMS['Recuperadosnovos'].max() + M0_MS  # most recent removed
I0_MS = dfMS['emAcompanhamentoNovos'].max() # most recent active reported cases

# IDENTIFY 13 DAYS AGO
# Hypothesis: one takes 13 days to recover
backdate = pd.to_datetime(startdate) - pd.DateOffset(days=13)
# DECEASED
M0 = dfMS['obitosAcumulado'][dfMS['data'] == startdate].values[0]
# CUMULATIVE INFECTED FROM THE PREVIOUS 13 DAYS
Infect = dfMS['casosAcumulado'][dfMS['data'].
								between(backdate,startdate,inclusive = True)]

# ESTIMATED INITIAL CONDITIONS
if metodo == "subreport":
	sub_report = 15
	Infect = Infect * sub_report
	# INFECTED
	I0 = max(Infect) - min(Infect)
	# RECOVERED
	R0 = min(Infect) # max(Infect) - I0
elif metodo == "fator_verity":
	I0 = M0 * 165 # estimated from Verity to Brazil: country, state, city
	R0 = I0 * 0.6 # Hypothesis Removed correspond to 60% of the Infected
# EXPOSED
E0 = 0.8 * I0  # Hypothesis Exposed correspond to 80% of the Infected

# TimeSeries
datelist = [d.strftime('%d/%m/%Y')
	for d in pd.date_range(start = startdate, periods = 365)]


# PLOT
fig_style = "ggplot" # "ggplot" # "classic" #
# plot
tamfig = (8,6)     # Figure Size
fsLabelTitle = 11   # Font Size: Label and Title
fsPlotLegend = 10   # Font Size: Plot and Legend


cor = ['b','r','k','g','y','m']     # Line Colors
ls = ['-.', '-']                    # Line Style
leg_loc = 'upper left' # 'upper right' # 'upper left' #
filetype = '.pdf'      # '.pdf' # '.png' # '.svg' #

plt.figure(11, figsize = tamfig)
plt.style.use(fig_style)

plt.plot(dfMS['data'], dfMS['obitosAcumulado'], ls[0], color = cor[0])

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gcf().autofmt_xdate() # Rotation

plt.title(state_name, fontsize=fsLabelTitle)
plt.legend(['MinSaude'], loc = leg_loc, fontsize=fsPlotLegend)
plt.xlabel('Date', fontsize=fsLabelTitle)
plt.ylabel('Deceased people', fontsize=fsLabelTitle)
plt.show()




# IMPORT DATA - Florianopolis
data = pd.read_csv(r"C:\Users\Fuck\Downloads\covid_anonimizado.csv") 

data['Data da notificação'] = pd.to_datetime(data['Data da notificação'])
data['Data do início dos sintomas'] = pd.to_datetime(data['Data do início dos sintomas'], format='%Y-%m-%d')
delay = data['Data da notificação'] - data['Data do início dos sintomas']




