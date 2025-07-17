# from Tools.scripts.generate_re_casefix import alpha
# from numpy.ma.core import cumsum

from Stochastic_interest_rates import *
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("TkAgg") # fixes plots not showing up in PyCharm

TIME_ARRAY = np.arange(1/24, 70, 1/24) #From 2 weeks to 70 years by 2 weeks

#Market rates data:
market_rates_path =  r"Market_Data\Treasury_Yields_20250618.csv"
market_rates_data = pd.read_csv(market_rates_path)
print(market_rates_data)

# Bond prices to interpolate (based on Market rates data)
Tenor_years = np.array([1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
bond_prices_data = np.exp(- Tenor_years * market_rates_data["Rate"].values/100)
log_P = np.log(bond_prices_data)
spline = scipy.interpolate.UnivariateSpline(Tenor_years, -np.gradient(log_P, Tenor_years), s=0)

f_t = spline(Tenor_years)
df_dt = spline.derivative()(Tenor_years)

plt.plot(Tenor_years, bond_prices_data)
plt.show()

interpolated_bond_prices = scipy.interpolate.PchipInterpolator(Tenor_years, bond_prices_data, extrapolate=False)
time = np.arange(0, 30, 0.1)
interpolated_bond_prices_values = [interpolated_bond_prices(t) for t in time]

plt.plot(time, interpolated_bond_prices_values)
plt.show()


# Create yield curve object
yc = yield_curve()

# Read the market rates data
yc.read_market_data(market_rates_path)

# Setup Cubic Spline Interpolation
print("#----------Interpolating_Cubic----------#")
yc.interpolate_yield_and_discount_curves("Cubic Spline")
cubic_interest_rates = np.array([yc.yield_curve["Cubic Spline"](t) for t in TIME_ARRAY])
cubic_discount_rates = np.array([yc.discount_curve["Cubic Spline"](t) for t in TIME_ARRAY])
Cubic_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":cubic_discount_rates, "Interest Rate":cubic_interest_rates})
print("#----------Cubic_Interpolation_Done----------#")
#print(Cubic_df)

# Setup PCHIP Interpolation
print("#----------Interpolating_PCHIP----------#")
yc.interpolate_yield_and_discount_curves("PCHIP")
PCHIP_interest_rates = np.array([yc.yield_curve["Cubic Spline"](t) for t in TIME_ARRAY])
PCHIP_discount_rates = np.array([yc.discount_curve["Cubic Spline"](t) for t in TIME_ARRAY])
PCHIP_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":PCHIP_discount_rates, "Interest Rate":PCHIP_interest_rates})
print("#----------PCHIP_Interpolation_Done----------#")
print(PCHIP_df)

# Read Bond Option Market Data
bond_options_md = read_bond_options_market_data(r"Market_Data\Bond_Options_20250618.csv")

# Calibrate model to the market prices
# print("#----------PCHIP----------#")
forward_rates = np.array([yc.forward_rate(T,T+1/12,"PCHIP") for T in bond_options_md['Option_Maturity_Y'].values])
bond_options_md["Price"] = bond_option_price_HW1F(K = bond_options_md['Strike'].values, opt_mat=bond_options_md['Option_Maturity_Y'].values,
                                                  bond_len= bond_options_md['Bond_Length_Y'].values,
                                                  discount_curve = yc.discount_curve["PCHIP"],
                                                  alpha = 0.07, sigma = 0.02, r_t = forward_rates)

yc.calibrate_term_structure_model(bond_options_md)
theta = yc.model_parameters["Hull-White-1F"]["PCHIP"][0]

time = np.arange(0.5, 25.5, 0.5)
yc_fun = yc.yield_curve["PCHIP"]
yc_val = np.array([yc_fun(t) for t in time])
theta_val = np.array([theta(t) for t in time])

# plt.plot(time, theta_val)
# plt.title("Theta")
# plt.show()

########################################################################################################################

# r_t path using Euler approx.

def r_path(r_0, time, dt, theta, alpha, sigma, N_paths):

    r_t = np.full((N_paths, len(time)), r_0)

    for i in range(len(time)-1):
        th = theta(time[i])
        dr = (th - alpha*r_t[:, i])*dt + sigma*np.random.normal(loc=0, scale=np.sqrt(dt), size=N_paths)
        r_t[:, i+1] = r_t[:, i] + dr

    return r_t # np.mean(r_t, axis=0)

# time = np.arange(30.25, 50, 0.25)
# plt.plot(time, [theta(t) for t in time])
# plt.show()

# time = np.arange(1, 25, 0.01)
# r_t = r_path(0.0458, time, 0.25, theta, 0.07, 0.02, 100000)
# YC = [yc.yield_curve['PCHIP'](t) for t in time]

# rate comparison
# plt.plot(time, YC, label="YC")
# plt.plot(time, np.mean(r_t, axis=0), label="Trap")
# plt.legend()
# plt.title("r_t vs YC")
# plt.show()


# PCHIP_df values check
PCHIP_df = PCHIP_df.drop(np.arange(0,9,1))
PCHIP_df = PCHIP_df.reset_index(drop=True)

# plt.plot(PCHIP_df["Tenor"].values, PCHIP_df["Interest Rate"].values)
# plt.title("PCHIP_df")
# plt.xlabel("Tenor")
# plt.ylabel("Interest Rate")
# plt.show()


# Yield_Curve values check

times = np.arange(0, 30, 0.1)
YC = [yc.yield_curve["PCHIP"](t) for t in times]
#
# plt.plot(times, YC)
# plt.title("YC")
# plt.xlabel("Time")
# plt.ylabel("Yield")
# plt.show()

# short rate

dt = 0.1
# time = np.arange(0, 30+dt, dt)
Time = [np.arange(0, T+dt, dt) for T in range(0, 30)]

r_t_paths = [r_path(0.0458, time, dt, theta, 0.07, 0.02, 1000) for time in Time]

plt.plot(Time[-1], np.mean(r_t_paths[-1], axis=0))
plt.title("r_t path")
plt.show()

# bond prices comparison
bond_path = np.array([np.mean(np.exp(-scipy.integrate.trapezoid(r_t_paths[i], Time[i]))) for i in range(len(r_t_paths))])
DC = [yc.discount_curve['PCHIP'](t) for t in range(len(r_t_paths))]

plt.plot(range(len(r_t_paths)), DC, label="DC", color="red")
plt.plot(range(len(r_t_paths)), bond_path, label="Trap_bond", color="blue")
plt.legend()
plt.title("bond vs DC")
plt.show()



# print(f"Trap: {np.mean(np.exp(-scipy.integrate.trapezoid(r_t, time)))}")
# print(f"YC: {yc.discount_curve['PCHIP'](10)}")

# def bond_prices_MC(Time_array):
#     r_t_paths = [r_path(0.0458, time, dt, theta, 0.07, 0.02, 100000) for time in Time_array]
#     bond_path = [np.mean(np.exp(-scipy.integrate.trapezoid(r_t_paths[i], Time_array[i]))) for i in range(len(r_t_paths))]

########################################################################################################################

