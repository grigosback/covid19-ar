import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt

from datetime import date
from bs4 import BeautifulSoup
from scipy.optimize import curve_fit


def exponenial_func(t, a, b):
    return a * np.exp(t * b)

matplotlib.use("Agg")

# Conseguir la fecha actual, de mañana y de pasado mañana
today = date.today()
today = today.strftime("%#m/%#d/%y")
print(today)

# # Conseguir Datos de John Hopkins University
# url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
# data = pd.read_csv(url)

# Ingreso manual de datos
data = pd.Series(
	[
		'3/5/20',
        '3/6/20',
        '3/8/20',
        '3/9/20',
        '3/10/20',
        '3/11/20',
        '3/12/20',
        '3/13/20',
        '3/14/20',
        '3/15/20',
        '3/16/20',
        '3/17/20',
        '3/18/20',
        '3/19/20',
        '3/20/20',
        '3/21/20',
        '3/22/20',
        '3/23/20',
        '3/24/20',
        '3/25/20',
        '3/26/20',
        '3/27/20',
        '3/28/20',
        '3/29/20',
        '3/30/20',
        '3/31/20',
	]
)

# days = np.array(
#     [
#         5,
#         6,
#         8,
#         9,
#         10,
#         11,
#         12,
#         13,
#         14,
#         15,
#         16,
#         17,
#         18,
#         19,
#         20,
#         21,
#         22,
#         23,
#         24,
#         25,
#         26,
#         27,
#         28,
#         29,
#         30,
#         31,
#     ]
# )
today_index = data.tolist().index(today)
# today_index = data.str.index(today) 


cases = np.array(
    [
        1,
        2,
        12,
        17,
        19,
        21,
        31,
        34,
        45,
        56,
        65,
        79,
        97,
        128,
        158,
        225,
        266,
        301,
        385,
        502,
        589,
    ]
)
end = len(cases)
print(data[:])
xfechas = pd.to_datetime(data[:])
print(xfechas)
# dias_dt = np.array(xfechas - np.datetime64('2020-01-01'))/ np.timedelta64(1,'D')
# print(dias_dt)

popt, pcov = curve_fit(exponenial_func, data.index[2:end], cases[2:end])
a = popt[0]
b = popt[1]
fit = a * np.exp(data.index * b)
fit_sup = (a + pcov[0, 0] ** 0.5) * np.exp(data.index * (b + pcov[1, 1] ** 0.5))
fit_inf = (a - pcov[0, 0] ** 0.5) * np.exp(data.index * (b - pcov[1, 1] ** 0.5))

xs = data.index[0:end] + 5
ys = cases

# Plot de Casos Confirmados
fig = plt.figure(figsize=(10, 5))
params = {"ytick.color" : '#78c9ff',
          "xtick.color" : '#78c9ff',
          "axes.labelcolor" : '#78c9ff',
          "axes.edgecolor" : '#78c9ff'}
plt.rcParams.update(params)
ax = fig.add_subplot(111)
plt.grid()
ax.plot(data[:end], ys, "o")
tmp = plt.xticks(rotation='vertical')
for x,y in zip(data[:end],ys):
    label = y
    ax.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',
                 color='#78c9ff') # horizontal alignment can be left, right or center
plt.xlabel("Fecha")
plt.ylabel("Casos Confirmados")
fig.patch.set_facecolor('#78c9ff')
fig.patch.set_alpha(0.1)
ax.patch.set_facecolor('#78c9ff')
ax.patch.set_alpha(0.01)
plt.savefig("graphs/casosconfirmados.png", dpi=200, bbox_inches="tight")


# Plot de Estimacion con cotas
fig = plt.figure(figsize=(10, 5))
plt.grid()
params = {"ytick.color" : '#78c9ff',
          "xtick.color" : '#78c9ff',
          "axes.labelcolor" : '#78c9ff',
          "axes.edgecolor" : '#78c9ff'}
plt.rcParams.update(params)
ax = fig.add_subplot(111)
# plt.yscale("Log")
ax.plot(data[:end], cases, "o")
ax.plot(data, fit)
ax.plot(data, fit_sup, linestyle="--")
ax.plot(data, fit_inf, linestyle="--")
plt.fill_between(data, fit_sup, fit_inf, color="gray")
plt.xlabel("Fecha")
plt.ylabel("N° de contagios")
plt.legend(["Datos", "Estimación", "Cota superior", "Cota inferior"])
# plt.xlim(5,20)
# plt.ylim(0,175)
fig.patch.set_facecolor('#78c9ff')
fig.patch.set_alpha(0.1)
ax.patch.set_facecolor('#78c9ff')
ax.patch.set_alpha(0.01)
tmp = plt.xticks(rotation='vertical')
plt.savefig("graphs/covid.png", dpi=200, bbox_inches="tight")
# plt.show()





# argdata = data.loc[lambda data: data['Country/Region'] == 'Argentina'].transpose()
# # argdata = data.loc[lambda data: data['Province/State'] == 'Hubei'] # (en caso de que se trate de una provincia)



# y = (argdata.to_numpy()[4:])[:,0].transpose()


# fig = plt.figure()
# params = {"ytick.color" : '#78c9ff',
#           "xtick.color" : '#78c9ff',
#           "axes.labelcolor" : '#78c9ff',
#           "axes.edgecolor" : '#78c9ff'}
# plt.rcParams.update(params)
# ax = fig.add_subplot(111)
# ax.plot(xfechas,y, '.')
# tmp = plt.xlim(np.datetime64('2020-03-01'), xfechas[-1])
# tmp = plt.xticks(rotation='vertical')
# fig.patch.set_facecolor('#78c9ff')
# fig.patch.set_alpha(0.1)
# ax.patch.set_facecolor('#78c9ff')
# ax.patch.set_alpha(0.01)
# plt.savefig('Exponential.png')
# # plt.show()



# popt_exponential, pcov_exponential = curve_fit(exponenial_func, np.array(x,dtype='float64'), np.array(y,dtype='float64'), p0=[0.1,0.1])

# a = popt_exponential[0]
# k = popt_exponential[1]
# xcont = np.linspace(20,90,1000)
# ycont = a*np.exp(k*xcont)
# fig = plt.figure()
# params = {"ytick.color" : '#78c9ff',
#           "xtick.color" : '#78c9ff',
#           "axes.labelcolor" : '#78c9ff',
#           "axes.edgecolor" : '#78c9ff'}
# plt.rcParams.update(params)
# ax = fig.add_subplot(111)
# ax.plot(np.datetime64('2020-01-01')+xcont*np.timedelta64(1,'D'), ycont)
# ax.plot(xfechas,y, '.')
# tmp = plt.xlim(np.datetime64('2020-03-01'), np.datetime64('2020-04-01'))
# tmp = plt.ylim(1, np.max(ycont))
# tmp = plt.xticks(rotation='vertical')
# plt.ylabel("Casos reportados")
# fig.patch.set_facecolor('#78c9ff')
# fig.patch.set_alpha(0.1)
# ax.patch.set_facecolor('#78c9ff')
# ax.patch.set_alpha(0.01)
# plt.savefig('AjusteLineal.png')



# fig = plt.figure()
# params = {"ytick.color" : '#78c9ff',
#           "xtick.color" : '#78c9ff',
#           "axes.labelcolor" : '#78c9ff',
#           "axes.edgecolor" : '#78c9ff'}
# plt.rcParams.update(params)
# ax = fig.add_subplot(111)
# ax.semilogy(np.datetime64('2020-01-01')+xcont*np.timedelta64(1,'D'), ycont)
# ax.semilogy(xfechas,y, '.')
# tmp = plt.xlim(np.datetime64('2020-03-01'), np.datetime64('2020-04-01'))
# tmp = plt.ylim(1, np.max(ycont))
# tmp = plt.xticks(rotation='vertical')
# plt.ylabel("Casos reportados")
# fig.patch.set_facecolor('#78c9ff')
# fig.patch.set_alpha(0.1)
# ax.patch.set_facecolor('#78c9ff')
# ax.patch.set_alpha(0.01)
# plt.savefig('ExpFit.png')
# # plt.show()

# Fechas = np.array([np.datetime64(today), np.datetime64(tommorow), np.datetime64(the_day_after_tommorow)])
# x1 = ((Fechas - np.datetime64('2020-01-01'))/ np.timedelta64(1,'D'))
# Casos = a*np.exp(k*x1)


# Actualizo el texto en la pagina web
with open('Dashboard.html') as html_file:
	soup = BeautifulSoup(html_file.read(), features='html.parser')

	new_tag = soup.new_tag("p")
	new_tag['id'] = "Stats"
	texto = "El último dato que tenemos es " + str(round(cases[today_index-1]))+ " del día de ayer"
	new_tag.string = texto
	for tag in soup.find_all(id="Stats"):
		tag.replace_with(new_tag)

	new_tag = soup.new_tag("p")
	new_tag['id'] = "Pred0"
	texto = "Para hoy se estiman "+str(round(fit[today_index]))+" casos, con una cota superior de "+str(round(fit_sup[today_index]))+" y una cota inferior de "+str(round(fit_inf[today_index]))
	new_tag.string = texto
	for tag in soup.find_all(id="Pred0"):
		tag.replace_with(new_tag)

	new_tag = soup.new_tag("p")
	new_tag['id'] = "Pred1"
	texto = "Para mañana se estiman "+str(round(fit[today_index+1]))+" casos, con una cota superior de "+str(round(fit_sup[today_index+1]))+" y una cota inferior de "+str(round(fit_inf[today_index+1]))
	new_tag.string = texto
	for tag in soup.find_all(id="Pred1"):
		tag.replace_with(new_tag)

	new_tag = soup.new_tag("p")
	new_tag['id'] = "Pred2"
	texto = "Para pasado mañana se estiman "+str(round(fit[today_index+2]))+" casos, con una cota superior de "+str(round(fit_sup[today_index+2]))+" y una cota inferior de "+str(round(fit_inf[today_index+2]))
	new_tag.string = texto
	for tag in soup.find_all(id="Pred2"):
		tag.replace_with(new_tag)

	# new_tag = soup.new_tag("p")
	# new_tag['id'] = "X2"
	# texto = "Periodo de duplicacion: {:.1f} dias.\n".format(np.log(2)/k)
	# new_tag.string = texto
	# for tag in soup.find_all(id="X2"):
	# 	tag.replace_with(new_tag)

	# new_tag = soup.new_tag("p")
	# new_tag['id'] = "X10"
	# texto = "Periodo de 10x: {:.1f} dias.\n".format(np.log(10)/k)
	# new_tag.string = texto
	# for tag in soup.find_all(id="X10"):
	# 	tag.replace_with(new_tag)

	new_text = soup.prettify()
	with open('Dashboard.html', mode='w') as new_html_file:
		new_html_file.write(new_text)
