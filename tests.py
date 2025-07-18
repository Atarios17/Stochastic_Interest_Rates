from Stochastic_interest_rates import *
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import QuantLib as ql
matplotlib.use("TkAgg") # fixes plots not showing up in PyCharm


# setx "TCL_LIBRARY=E:\Python\tcl\tcl8.6"
# setx "TK_LIBRARY=E:\Python\tcl\tcl8.6"

TIME_ARRAY = np.arange(0, 70, 1/24) #From 2 weeks to 70 years by 2 weeks

#Market rates data:
market_rates_path =  r"Market_Data\Treasury_Yields_20250618.csv"
market_rates_data = pd.read_csv(market_rates_path)
print(market_rates_data)

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
#print(PCHIP_df)

# Setup Linear Interpolation
print("#----------Interpolating_Linear----------#")
yc.interpolate_yield_and_discount_curves("Linear")
Linear_interest_rates = np.array([yc.yield_curve["Linear"](t) for t in TIME_ARRAY])
Linear_discount_rates = np.array([yc.discount_curve["Linear"](t) for t in TIME_ARRAY])
Linear_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":Linear_discount_rates, "Interest Rate":Linear_interest_rates})
print("#----------Linear_Interpolation_Done----------#")
#print(Linear_df)

# Plot interpolated yield curves -> they should be constant before 1st and last market data tenors
plt.figure(figsize=(10, 5))
#sns.scatterplot(data=Cubic_df, x="Tenor", y="Interest Rate", color="red", label="Cubic Spline Interpolation")
sns.lineplot(data=Cubic_df, x="Tenor", y="Interest Rate", color="red", label="Cubic Spline Interpolation")
sns.lineplot(data=PCHIP_df, x="Tenor", y="Interest Rate", color="green", label="PCHIP Interpolation")
sns.lineplot(data=Linear_df, x="Tenor", y="Interest Rate", color="blue", label="Linear Interpolation")
plt.xlabel("Tenor (Years)")
plt.ylabel("Interest Rate")
plt.title("Yield curves by Interpolation")
plt.legend()
plt.grid(True)
plt.show()
plt.waitforbuttonpress()
plt.close()

# Bond Option Price
# print("#----------Cubic Spline----------#")
# print(bond_option_price_HW1F(K = 0.045, opt_mat = 10, bond_len = 12, bond_curve = yc.discount_curve["Cubic Spline"], alpha = 0.07, sigma = 0.02))
#
# print("#----------PCHIP----------#")
# print(bond_option_price_HW1F(K = 0.045, opt_mat = 10, bond_len = 12, bond_curve = yc.discount_curve["PCHIP"], alpha = 0.07, sigma = 0.02))
#
# print("#----------Linear----------#")
# print(bond_option_price_HW1F(K = 0.045, opt_mat = 10, bond_len = 12, bond_curve = yc.discount_curve["Linear"], alpha = 0.07, sigma = 0.02))


# Read Bond Option Market Data
bond_options_md = read_bond_options_market_data(r"Market_Data\Bond_Options_20250618.csv")

# Calibrate model to the market prices
print("#--Calibrating_HW_Model_to_Bond_Options_PCHIP--#")
forward_rates = np.array([yc.forward_rate(T,T+1/12,"PCHIP") for T in bond_options_md['Option_Maturity_Y'].values])
bond_options_md["Price"] = bond_option_price_HW1F(K = bond_options_md['Strike'].values, opt_mat=bond_options_md['Option_Maturity_Y'].values,
                                                  bond_len= bond_options_md['Bond_Length_Y'].values,
                                                  discount_curve = yc.discount_curve["PCHIP"],
                                                  alpha = 0.07, sigma = 0.02, r_t = forward_rates)

yc.calibrate_term_structure_model(bond_options_md)
print("#--Calibration_Finished--#")
print("#--Model Parameters: (theta, alpha, sigma)--#")
print(yc.model_parameters["Hull-White-1F"]["PCHIP"])
#yc.model_parameters["Hull-White-1F"]["PCHIP"][2] = np.float64(0.02)

# Setup Quantlib for benchmarking
print("#--Setting Up QuantLib Hull-White Model for benchmarking--#")
# -- QuantLib time settings
timestep = 360
length = 30 # in years
day_count = ql.Thirty360(ql.Thirty360.BondBasis)
todays_date = ql.Date(18, 6, 2025) # In line with input market data
ql.Settings.instance().evaluationDate = todays_date

#
sigma = 0.02 # yc.model_parameters["Hull-White-1F"]["PCHIP"][2]
alpha = 0.07 # yc.model_parameters["Hull-White-1F"]["PCHIP"][1]
forward_rate = 0.042 # yc.yield_curve["PCHIP"](0)

# QL Read market rates
ql_market_rates_dates = [todays_date + ql.Period(tenor) for tenor in market_rates_data["Tenor"].values]
# Tenor 1.5M is treated incorrectly in line above (is treated as 1M), so we shift extra 0.5M = 15d manually
ql_market_rates_dates[1] += ql.Period("15d")

ql_yield_curve_market = ql.ZeroCurve(ql_market_rates_dates, market_rates_data['Rate'].values*0.01, day_count)
ql_yield_curve_market.enableExtrapolation()
#plt.plot(yield_curve_market.times(),yield_curve_market.zeroRates())
#plt.show()
ql_spot_curve_handle = ql.YieldTermStructureHandle(ql_yield_curve_market)
# Quantlib HullWhite model
ql_hw_model = ql.HullWhite(ql_spot_curve_handle, alpha, sigma)
print("#--QuantLib HW is set up--#")

# Benchmark model Bonds pricing
print("#--Benchmark of Bond Prices--#")
Our_Bond_Prices = [bond_price_HW1F(yc.discount_curve["PCHIP"],yc.yield_curve["PCHIP"](0),0,T,alpha) for T in TIME_ARRAY]
plt.plot(TIME_ARRAY,Our_Bond_Prices, label = 'Our', color = 'blue')
plt.plot(TIME_ARRAY,PCHIP_discount_rates, label = 'Market', color = 'green')
plt.plot(TIME_ARRAY,[ql_hw_model.discountBond(0,T,0.042) for T in TIME_ARRAY], label = 'QuantLib', color = 'orange')
plt.title("Bond Prices Comparison")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()

#Git!

# Benchmark of Bond option Pricing
print("#--Benchmark of Call Bond Option Prices--#")
call_put = 1
option_mat = 3
bond_len = 4
strike = float(bond_price_HW1F(yc.discount_curve["PCHIP"],yc.yield_curve["PCHIP"](0),option_mat,option_mat+bond_len,alpha))
# Single price
print("#--Example single Bond option Price--#")
print("#-- \n option maturity: {} \n bond length: {} \n strike: {} \n --#".format(option_mat,bond_len, strike))
our_bp = bond_option_price_HW1F(strike,option_mat,bond_len,yc.discount_curve["PCHIP"],alpha, sigma, yc.yield_curve["PCHIP"](0))
print("Our Bond Price:")
print(our_bp)
ql_bp = ql_hw_model.discountBondOption(call_put,strike,option_mat,option_mat+bond_len)
print("QuantLib Bond Price:")
print(ql_bp)
print("Relative Difference: {:.2f}%".format((our_bp-ql_bp)/ql_bp*100))

#Plot depending on bond length
bond_lengths = np.arange(0.25,15,0.25)
Our_Bond_Option_Prices = [bond_option_price_HW1F(strike,option_mat,T,yc.discount_curve["PCHIP"],alpha, sigma, yc.yield_curve["PCHIP"](0)) for T in bond_lengths]
plt.plot(bond_lengths,Our_Bond_Option_Prices, label = 'Our', color = 'blue')
plt.plot(bond_lengths,[ql_hw_model.discountBondOption(call_put,strike,option_mat,option_mat+T) for T in bond_lengths], label = 'QuantLib', color = 'orange')
plt.title("Bond Option Prices Comparison")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()

# Git!
print("#--Benchmark of Jamshidian Swaption Prices--#")
notional= 1
swap_start = 5
frequency = 1
fixed_rate = 0.04

# Single price
print("#--Example single Bond option Price--#")
print("#-- \n option maturity: {} \n bond length: {} \n strike: {} \n --#".format(option_mat,bond_len, strike))
our_sp = jamshidian_swaption_price(yc.discount_curve["PCHIP"], swap_start, 5, fixed_rate, frequency, notional, alpha, sigma, True)
print("Our Bond Price:")
print(our_bp)
swap = ql_create_swap(ql_spot_curve_handle, swap_start, 5, frequency, fixed_rate)
swaption = ql_create_swaption(swap, ql_hw_model, ql_spot_curve_handle)
ql_sp = swaption.NPV()
print("QuantLib Bond Price:")
print(ql_sp)
print("Relative Difference: {:.2f}%".format((our_sp-ql_sp)/ql_sp*100))

# Plot of prices w.r.t. swap maturity
swap_lenghts = np.arange(1,15)
Our_Swaption_Prices = [jamshidian_swaption_price(yc.discount_curve["PCHIP"], swap_start, swap_len, fixed_rate, frequency, notional, alpha, sigma, True) for swap_len in swap_lenghts]

ql_swaption_prices = []
for swap_len in swap_lenghts:
    swap = ql_create_swap(ql_spot_curve_handle, swap_start, int(swap_len), frequency, fixed_rate)
    swaption = ql_create_swaption(swap, ql_hw_model, ql_spot_curve_handle)
    ql_swaption_prices.append(swaption.NPV())

plt.plot(swap_lenghts,Our_Swaption_Prices, label = 'Our', color = 'blue')
plt.plot(swap_lenghts,ql_swaption_prices, label = 'QuantLib', color = 'orange')
plt.title("Jamshidian Swaptions Comparison")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()