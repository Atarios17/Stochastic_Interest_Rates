from Stochastic_interest_rates import *
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

# setx "TCL_LIBRARY=E:\Python\tcl\tcl8.6"
# setx "TK_LIBRARY=E:\Python\tcl\tcl8.6"

TIME_ARRAY = np.arange(1/24, 70, 1/24) #From 2 weeks to 70 years by 2 weeks

#Market rates data:
market_rates_path =  r"Market_Data\Treasury_Yields_20250618.csv"

market_rates_data = pd.read_csv(market_rates_path)
print(market_rates_data)

# Create yield curve object
yc = yield_curve()

# Read the market rates data
yc.read_market_data(market_rates_path)

# Setup Cubic Spline Interpolation
print("#----------Cubic Spline----------#")
yc.interpolate_yield_and_discount_curves("Cubic Spline")
cubic_interest_rates = [yc.yield_curve["Cubic Spline"](t) for t in TIME_ARRAY]
cubic_discount_rates = [yc.discount_curve["Cubic Spline"](t) for t in TIME_ARRAY]
Cubic_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":cubic_discount_rates, "Interest Rate":cubic_interest_rates})
print(Cubic_df)

# Setup PCHIP Interpolation
print("#----------PCHIP----------#")
yc.interpolate_yield_and_discount_curves("PCHIP")
PCHIP_interest_rates = [yc.yield_curve["Cubic Spline"](t) for t in TIME_ARRAY]
PCHIP_discount_rates = [yc.discount_curve["Cubic Spline"](t) for t in TIME_ARRAY]
PCHIP_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":PCHIP_discount_rates, "Interest Rate":PCHIP_interest_rates})
print(PCHIP_df)

# Setup Linear Interpolation
print("#----------Linear----------#")
yc.interpolate_yield_and_discount_curves("Linear")
Linear_interest_rates = [yc.yield_curve["Linear"](t) for t in TIME_ARRAY]
Linear_discount_rates = [yc.discount_curve["Linear"](t) for t in TIME_ARRAY]
Linear_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Discount Rate":Linear_discount_rates, "Interest Rate":Linear_interest_rates})
print(Linear_df)

# # Plot interpolated yield curves -> they should be constant before 1st and last market data tenors
# plt.figure(figsize=(10, 5))
# # sns.scatterplot(data=yc.bond_prices, x="Tenor", y="Bond Price", color="red", label="Original Data", s=80)
# sns.lineplot(data=Cubic_df, x="Tenor", y="Interest Rate", color="red", label="Cubic Spline Interpolation")
# sns.lineplot(data=PCHIP_df, x="Tenor", y="Interest Rate", color="green", label="PCHIP Interpolation")
# sns.lineplot(data=Linear_df, x="Tenor", y="Interest Rate", color="blue", label="Linear Interpolation")
# plt.xlabel("Tenor (Years)")
# plt.ylabel("Interest Rate")
# plt.title("Yield curves by Interpolation")
# plt.legend()
# plt.grid(True)
# plt.savefig(r"C:\Users\Artor\Desktop\Quant_Learn\Interest_Rate_Modelling\test.jpg")
# plt.show()
# plt.waitforbuttonpress()
# plt.close()

# # Plot interpolated discount curves
# plt.figure(figsize=(10, 5))
# # sns.scatterplot(data=yc.bond_prices, x="Tenor", y="Bond Price", color="red", label="Original Data", s=80)
# sns.lineplot(data=Cubic_df, x="Tenor", y="Discount Rate", color="red", label="Cubic Spline Interpolation")
# sns.lineplot(data=PCHIP_df, x="Tenor", y="Discount Rate", color="green", label="PCHIP Interpolation")
# sns.lineplot(data=Linear_df, x="Tenor", y="Discount Rate", color="blue", label="Linear Interpolation")
# plt.xlabel("Tenor (Years)")
# plt.ylabel("Discount Rate")
# plt.title("Discount curves by Interpolation")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.waitforbuttonpress()
# plt.close()

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
print("#----------PCHIP----------#")
bond_options_md["Price"] = bond_option_price_HW1F(bond_options_md['Strike'].values,bond_options_md['Option_Maturity_Y'].values,bond_options_md['Bond_Length_Y'].values,yc.discount_curve["PCHIP"],alpha = 0.07, sigma = 0.02)
yc.calibrate_term_structure_model(bond_options_md)
print(yc.model_parameters["Hull-White-1F"]["PCHIP"])

# Jamshidian swaption price
print("#----------Jamshidian----------#")
# print(f'HW bond price {bond_price_HW1F(yc, 0, 2, 0.03)}')
K = yc.discount_curve["Cubic Spline"](10) # sum(np.array([yc.discount_curve["Cubic Spline"](t) for t in np.arange(0.5, 10.5, 0.5)]))
print(f'K value: {K}')
print(f'Swaption price: {jamshidian_swaption_price(yc.yield_curve["PCHIP"], 10, 1, K, 1, 1, alpha = 0.07, sigma = 0.02)}')
print(f'Bond option price: {bond_option_price_HW1F(K, opt_mat = 10, bond_len = 1, yield_curve = yc.yield_curve["PCHIP"], alpha = 0.07, sigma = 0.02)}')
