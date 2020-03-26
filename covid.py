#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")
get_ipython().run_line_magic("matplotlib", "qt")


def exponenial_func(t, a, b):
    return a * np.exp(t * b)


# %%
days = np.array(
    [
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
    ]
)


days = days - days[0]
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
    ]
)
end = len(cases)

# %%
popt, pcov = curve_fit(exponenial_func, days[2:end], cases[2:end])
a = popt[0]
b = popt[1]
fit = a * np.exp(days * b)
fit_sup = (a + pcov[0, 0] ** 0.5) * np.exp(days * (b + pcov[1, 1] ** 0.5))
fit_inf = (a - pcov[0, 0] ** 0.5) * np.exp(days * (b - pcov[1, 1] ** 0.5))

# %%
plt.figure(figsize=(10, 5))
plt.grid()
# plt.yscale("Log")
plt.plot(days[0:end] + 5, cases, "o")
plt.plot(days + 5, fit)
plt.plot(days + 5, fit_sup, linestyle="--")
plt.plot(days + 5, fit_inf, linestyle="--")
plt.fill_between(days + 5, fit_sup, fit_inf, color="gray")
plt.xlabel("Día de marzo")
plt.ylabel("N° de contagios")
plt.legend(["Datos", "Estimación", "Cota superior", "Cota inferior"])
# plt.xlim(5,20)
# plt.ylim(0,175)
plt.savefig("graphs/covid.png", dpi=200, bbox_inches="tight")
plt.show()


# %%
