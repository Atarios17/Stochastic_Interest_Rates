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

# Create hull white model instances (one for each interpolation method)
hw_pchip = hull_white_model()
hw_cubic = hull_white_model()

# Read the market rates data
hw_pchip.read_market_data(market_rates_path)
hw_cubic.read_market_data(market_rates_path)

# Setup Cubic Spline Interpolation
print("#----------Interpolating_Cubic----------#")
hw_cubic.interpolate_yield_and_discount_curves(interpolation_method = "Cubic Spline")
cubic_interest_rates = hw_cubic.yield_curve(TIME_ARRAY)
cubic_discount_rates = hw_cubic.discount_curve(TIME_ARRAY)
Cubic_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":cubic_discount_rates, "Interest Rate":cubic_interest_rates})
print("#----------Cubic_Interpolation_Done----------#")

# Setup PCHIP Interpolation
print("#----------Interpolating_PCHIP----------#")
hw_pchip.interpolate_yield_and_discount_curves(interpolation_method = "PCHIP")
PCHIP_interest_rates = hw_pchip.yield_curve(TIME_ARRAY)
PCHIP_discount_rates = hw_pchip.discount_curve(TIME_ARRAY)
PCHIP_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":PCHIP_discount_rates, "Interest Rate":PCHIP_interest_rates})
print("#----------PCHIP_Interpolation_Done----------#")

# Plot interpolated yield curves -> they should be constant before 1st and last market data tenors
plt.figure(figsize=(10, 5))
sns.scatterplot(data=hw_cubic.market_rates, x="Tenor", y="Rate", color="green", label="Market Rates", s = 120)
sns.lineplot(data=Cubic_df, x="Tenor", y="Interest Rate", color="red", label="Cubic Spline Interpolation")
sns.lineplot(data=PCHIP_df, x="Tenor", y="Interest Rate", color="blue", label="PCHIP Interpolation")
plt.xlabel("Tenor (Years)")
plt.ylabel("Interest Rate")
plt.title("Yield curves by Interpolation")
plt.legend()
plt.grid(True)
plt.show()
plt.waitforbuttonpress()
plt.close()

# Plot interpolated discount curves
market_discounts = np.exp(-hw_cubic.market_rates["Tenor"].values*hw_cubic.market_rates["Rate"].values)
plt.figure(figsize=(10, 5))
sns.scatterplot(data = pd.DataFrame({"Tenor": hw_cubic.market_rates["Tenor"].values, "Discount Rate": market_discounts}), x="Tenor", y="Discount Rate", color="green", label="Market Rates", s = 120)
sns.lineplot(data=Cubic_df, x="Tenor", y="Discount Rate", color="red", label="Cubic Spline Interpolation")
sns.lineplot(data=PCHIP_df, x="Tenor", y="Discount Rate", color="blue", label="PCHIP Interpolation")
plt.xlabel("Tenor (Years)")
plt.ylabel("Discount Rate")
plt.title("Discount curves by Interpolation")
plt.legend()
plt.grid(True)
plt.show()
plt.waitforbuttonpress()
plt.close()

# Calibrating alpha and sigma to Bond Options
print("#--Calibrating_HW_Model_to_Bond_Options_PCHIP--#")
bond_option_market_data = read_bond_options_market_data(r"Market_Data\Bond_Options_20250618.csv")
hw_pchip.calibrate_model_params(bond_option_market_data,is_swaptions_market_data=False)
print("#--Calibration_Finished--#")
print("#--Model Parameters:--#")
print("alpha: {:.4f}".format(hw_pchip.alpha))
print("sigma: {:.4f}".format(hw_pchip.sigma))
print("theta: {}".format(hw_pchip.theta))

# Calibrating alpha and sigma to Swaptions
swaptions_market_data = read_swaptions_market_data(r"Market_Data\Swaptions_SABR_20250618.csv")
print("#--Calibrating_HW_Model_to_Swaptions_PCHIP--#")
hw_pchip.calibrate_model_params(swaptions_market_data)
print("#--Calibration_Finished--#")
print("#--Model Parameters:--#")
print("alpha: {:.4f}".format(hw_pchip.alpha))
print("sigma: {:.4f}".format(hw_pchip.sigma))
print("theta: {}".format(hw_pchip.theta))

# Monte Carlo Setup
dt = 1/400
T_end = 50
N_paths = 10000
print("#--Simulating_Monte_Carlo_r_paths--#")
time_vec, r_sims = hw_pchip.simulate_rate_paths(dt = dt, T = T_end, N_paths=N_paths)

# Monte Carlo Bond Prices vs Market rates:
print("#--Calculating_Monte_Carlo_bond_prices--#")
mc_bond_prices = [bond_price_mc(time_vec[:int(T/dt)], r_sims[:,:int(T/dt)]) for T in np.append(hw_pchip.market_rates["Tenor"].values,50)]
plt.figure(figsize=(10, 5))
sns.scatterplot(data = pd.DataFrame({"Tenor": np.append(hw_pchip.market_rates["Tenor"].values,50), "Discount Rate": mc_bond_prices}), x="Tenor", y="Discount Rate", color="blue", label="Monte Carlo", s = 90)
sns.lineplot(data=PCHIP_df, x="Tenor", y="Discount Rate", color="green", label="PCHIP Interpolation")
plt.xlabel("Tenor (Years)")
plt.ylabel("Discount Rate")
plt.title("Monte Carlo Discount Curve vs Interpolated PCHIP")
plt.legend()
plt.grid(True)
plt.show()
plt.waitforbuttonpress()
plt.close()

# Setup Quantlib for benchmarking
print("#--Setting Up QuantLib Hull-White Model for benchmarking--#")
# -- QuantLib time settings
timestep = 360
length = 30 # in years
day_count = ql.Thirty360(ql.Thirty360.BondBasis)
todays_date = ql.Date(18, 6, 2025) # In line with input market data
ql.Settings.instance().evaluationDate = todays_date

# QL Read market rates
ql_market_rates_dates = [todays_date + ql.Period(tenor) for tenor in market_rates_data["Tenor"].values]
# Tenor 1.5M is treated incorrectly in line above (is treated as 1M), so we shift extra 0.5M = 15d manually
ql_market_rates_dates[1] += ql.Period("15d")

#ql_yield_curve_market = ql.ZeroCurve(ql_market_rates_dates, market_rates_data['Rate'].values*0.01, day_count)
ql_yield_curve_market = ql.CubicZeroCurve(ql_market_rates_dates, market_rates_data['Rate'].values*0.01, day_count)
ql_yield_curve_market.enableExtrapolation()
ql_spot_curve_handle = ql.YieldTermStructureHandle(ql_yield_curve_market)
# Quantlib HullWhite model
ql_hw_model = ql.HullWhite(ql_spot_curve_handle, hw_pchip.alpha, hw_pchip.sigma)
print("#--QuantLib HW is set up--#")

# Benchmark model Bonds pricing
print("#--Benchmark of Bond Prices--#")
Our_Bond_Prices = bond_price_HW1F(hw_pchip.yield_curve,0,TIME_ARRAY,hw_pchip.alpha, hw_pchip.sigma, r_t = hw_pchip.yield_curve(0))
plt.plot(TIME_ARRAY,Our_Bond_Prices, label = 'Our', color = 'blue')
plt.plot(TIME_ARRAY,PCHIP_discount_rates, label = 'Market', color = 'green')
plt.plot(TIME_ARRAY,[ql_hw_model.discountBond(0,T,float(hw_pchip.yield_curve(0))) for T in TIME_ARRAY], label = 'QuantLib', color = 'orange')
plt.title("Bond Prices Comparison")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()

# Benchmark of Bond option Pricing
print("#--Benchmark of Call Bond Option Prices--#")
call_put = 1
option_mat = 3
bond_len = 4

#bond_price_HW1F(discount_curve, t, T, alpha, r_t = None):
strike = float(bond_price_HW1F(hw_pchip.yield_curve,option_mat,option_mat+bond_len,hw_pchip.alpha,hw_pchip.sigma))
# Single price
print("#--Example Bond Option Price--#")
print("#-- \n option maturity: {} \n bond length: {} \n strike: {} \n --#".format(option_mat,bond_len, strike))
our_bp = bond_option_price_HW1F(strike,option_mat,bond_len,hw_pchip.yield_curve,hw_pchip.alpha, hw_pchip.sigma, float(hw_pchip.yield_curve(0)))
print("Our Bond Option Price:")
print(our_bp)
t_vec, r_t = hw_pchip.simulate_rate_paths(T = option_mat, N_paths = 10000, dt = 0.0025)
mc_bp = bond_option_price_mc(t_vec, r_t, option_mat, bond_len, strike, hw_pchip.yield_curve, hw_pchip.alpha, hw_pchip.sigma)
print("MC Bond Option Price:")
print(mc_bp)
ql_bp = ql_hw_model.discountBondOption(call_put,strike,option_mat,option_mat+bond_len)
print("QuantLib Bond Option Price:")
print(ql_bp)
print("Relative Difference Ours to QuantLib: {:.2f}%".format((our_bp-ql_bp)/ql_bp*100))
print("Relative Difference Monte Carlo to QuantLib: {:.2f}%".format((mc_bp-ql_bp)/ql_bp*100))

# Plot of bond option prices w.r.t. bond maturity
bond_lengths = np.arange(0.25,20,0.25)
Our_Bond_Option_Prices = bond_option_price_HW1F(strike,option_mat,bond_lengths,hw_pchip.yield_curve, hw_pchip.alpha, hw_pchip.sigma)
MC_Bond_Option_Prices = np.array([bond_option_price_mc(time_vec[:int(option_mat/dt)], r_sims[:,:int(option_mat/dt)], option_mat, bond_len, strike, hw_pchip.yield_curve, hw_pchip.alpha, hw_pchip.sigma) for bond_len in bond_lengths])
plt.plot(bond_lengths,Our_Bond_Option_Prices, label = 'Our', color = 'blue')
plt.plot(bond_lengths,MC_Bond_Option_Prices, label = 'MC', color = 'green')
plt.plot(bond_lengths,[ql_hw_model.discountBondOption(call_put,strike,option_mat,option_mat+T) for T in bond_lengths], label = 'QuantLib', color = 'orange')
plt.title("Bond Option Prices Comparison w.r.t. Bond Lengths")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()

# Plot of bond option prices w.r.t. option maturity
opt_maturities = np.arange(0.25,15,0.25)
Our_Bond_Option_Prices = bond_option_price_HW1F(strike,opt_maturities,bond_len,hw_pchip.yield_curve, hw_pchip.alpha, hw_pchip.sigma)
MC_Bond_Option_Prices = np.array([bond_option_price_mc(time_vec[:int(option_mat/dt)], r_sims[:,:int(option_mat/dt)], option_mat, bond_len, strike, hw_pchip.yield_curve, hw_pchip.alpha, hw_pchip.sigma) for option_mat in opt_maturities])
plt.plot(opt_maturities,Our_Bond_Option_Prices, label = 'Our', color = 'blue')
plt.plot(opt_maturities,MC_Bond_Option_Prices, label = 'MC', color = 'green')
plt.plot(opt_maturities,[ql_hw_model.discountBondOption(call_put,strike,option_mat,option_mat+bond_len) for option_mat in opt_maturities], label = 'QuantLib', color = 'orange')
plt.title("Bond Option Prices Comparison w.r.t. Option Maturities")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()


print("#--Benchmark of Jamshidian Swaption Prices--#")
notional= 1
option_maturity = 5
swap_length = 5
frequency = 1
fixed_rate = 0.04

# Single Swaption price
print("#--Example Swaption Price--#")
print("#-- \n option maturity: {} \n swap length: {} \n fixed_strike: {:.2f}% \n --#".format(option_mat,bond_len, fixed_rate))
our_sp = jamshidian_swaption_price(hw_pchip.yield_curve, option_maturity, swap_length, fixed_rate, frequency, notional, hw_pchip.alpha, hw_pchip.sigma, True)

print("Our Swaption Price:")
print(our_sp)
swap = ql_create_swap(ql_spot_curve_handle, option_maturity, swap_length, frequency, fixed_rate)
swaption = ql_create_swaption(swap, ql_hw_model, ql_spot_curve_handle)
ql_sp = swaption.NPV()
print("QuantLib Swaption Price:")
print(ql_sp)
print("Relative Difference: {:.2f}%".format((our_sp-ql_sp)/ql_sp*100))

t_vec, r_t = hw_pchip.simulate_rate_paths(T = option_maturity, N_paths = 10000, dt = 0.0025)
mc_sp = swaption_price_mc(t_vec, r_t,option_mat = option_maturity, swap_len = swap_length, swap_freq = frequency, strike_fixed_rate = fixed_rate,
                  yield_curve = hw_pchip.yield_curve, alpha = hw_pchip.alpha, sigma = hw_pchip.sigma, is_payer=True)
print("MC Bond Option Price:")
print(mc_sp)
print("Relative Difference Monte Carlo vs QuantLib: {:.2f}%".format((mc_sp-ql_sp)/ql_sp*100))



# Plot of prices w.r.t. swap maturity
swap_lenghts = np.arange(1,15)
Our_Swaption_Prices = [jamshidian_swaption_price(hw_pchip.yield_curve, option_maturity, swap_len, fixed_rate, frequency, notional, hw_pchip.alpha, hw_pchip.sigma, True) for swap_len in swap_lenghts]

MC_Swaption_Prices = np.array([swaption_price_mc(t_vec, r_t,option_mat = option_maturity, swap_len = swap_len, swap_freq = frequency, strike_fixed_rate = fixed_rate,
                  yield_curve = hw_pchip.yield_curve, alpha = hw_pchip.alpha, sigma = hw_pchip.sigma, is_payer=True) for swap_len in swap_lenghts])

ql_swaption_prices = []
for swap_len in swap_lenghts:
    swap = ql_create_swap(ql_spot_curve_handle, option_maturity, int(swap_len), frequency, fixed_rate)
    swaption = ql_create_swaption(swap, ql_hw_model, ql_spot_curve_handle)
    ql_swaption_prices.append(swaption.NPV())

plt.plot(swap_lenghts,Our_Swaption_Prices, label = 'Our', color = 'blue')
plt.plot(swap_lenghts,MC_Swaption_Prices, label = 'Monte Carlo', color = 'green')
plt.plot(swap_lenghts,ql_swaption_prices, label = 'QuantLib', color = 'orange')
plt.title("Jamshidian Swaptions w.r.t. Swap Maturities")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()

# Plot of prices w.r.t. option maturity
option_maturities = np.arange(1,15)
Our_Swaption_Prices = [jamshidian_swaption_price(hw_pchip.yield_curve, option_maturity, swap_length, fixed_rate, frequency, notional, hw_pchip.alpha, hw_pchip.sigma, True) for option_maturity in option_maturities]

MC_Swaption_Prices = np.array([swaption_price_mc(time_vec[:int(option_maturity/dt)], r_sims[:,:int(option_maturity/dt)],option_mat = option_maturity, swap_len = swap_length, swap_freq = frequency, strike_fixed_rate = fixed_rate,
                  yield_curve = hw_pchip.yield_curve, alpha = hw_pchip.alpha, sigma = hw_pchip.sigma, is_payer=True) for option_maturity in option_maturities])

ql_swaption_prices = []
for option_maturity in option_maturities:
    swap = ql_create_swap(ql_spot_curve_handle, int(option_maturity), swap_length, frequency, fixed_rate)
    swaption = ql_create_swaption(swap, ql_hw_model, ql_spot_curve_handle)
    ql_swaption_prices.append(swaption.NPV())

plt.plot(option_maturities,Our_Swaption_Prices, label = 'Our', color = 'blue')
plt.plot(option_maturities,MC_Swaption_Prices, label = 'Monte', color = 'green')
plt.plot(option_maturities,ql_swaption_prices, label = 'QuantLib', color = 'orange')
plt.title("Jamshidian Swaptions w.r.t. Option Maturities")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()

# Plot of prices w.r.t. strike
strike_rates = np.arange(0.03,0.06,0.002)
Our_Swaption_Prices = [jamshidian_swaption_price(hw_pchip.yield_curve, 5, 5, fixed_rate, frequency, notional, hw_pchip.alpha, hw_pchip.sigma, True) for fixed_rate in strike_rates]

ql_swaption_prices = []
for fixed_rate in strike_rates:
    swap = ql_create_swap(ql_spot_curve_handle, 5, 5, frequency, fixed_rate)
    swaption = ql_create_swaption(swap, ql_hw_model, ql_spot_curve_handle)
    ql_swaption_prices.append(swaption.NPV())

plt.plot(strike_rates,Our_Swaption_Prices, label = 'Our', color = 'blue')
plt.plot(strike_rates,ql_swaption_prices, label = 'QuantLib', color = 'orange')
plt.title("Jamshidian Swaptions w.r.t. Strikes")
plt.legend()
plt.show()
plt.waitforbuttonpress()
plt.close()