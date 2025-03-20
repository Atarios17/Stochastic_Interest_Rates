from Stochastic_interest_rates import yield_curve
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

# setx "TCL_LIBRARY=E:\Python\tcl\tcl8.6"
# setx "TK_LIBRARY=E:\Python\tcl\tcl8.6"

TIME_ARRAY = np.arange(60, 10950, 50)

#Market rates Data:
path =  r"E:\PythonProjects\Stochastic Interest Rates\Stochastic_Interest_Rates\US_Treasury_Bonds_Live.csv"

market_data = pd.read_csv(path)
print(market_data)

# Create object
yc = yield_curve()

# Read the data
yc.read_market_data(path)

# Calculate bond prices
yc.calculate_bond_prices()

print(yc.bond_prices["Tenor"].values)
# print(yc.bond_prices["Bond Price"])

# Linear Interpolation
print("#----------Cubic Spline----------#")
yc.interpolate_bond_prices("Cubic Spline")
yc.interpolated_bond_curve["Cubic Spline"] = np.vectorize(yc.interpolated_bond_curve["Cubic Spline"])
rates_lin = [-np.log(yc.interpolated_bond_curve['Cubic Spline'](t))*(365/t) for t in TIME_ARRAY]

Lin_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Bond Price": yc.interpolated_bond_curve["Cubic Spline"](TIME_ARRAY),
                       "Rates": rates_lin})
print(Lin_df)

# PCHIP Interpolation
print("#----------PCHIP----------#")
yc.interpolate_bond_prices("PCHIP")
rates_pchip = [-np.log(yc.interpolated_bond_curve['PCHIP'](t))*(365/t) for t in TIME_ARRAY]

PCHIP_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Bond Price": yc.interpolated_bond_curve["PCHIP"](TIME_ARRAY),
                         "Rates": rates_pchip})
print(PCHIP_df)

# CH Spline Interpolation
print("#----------CH Spline----------#")

yc.interpolate_bond_prices("CH Spline")
yc.interpolated_bond_curve["CH Spline"] = np.vectorize(yc.interpolated_bond_curve["CH Spline"])
rates_ch = [-np.log(yc.interpolated_bond_curve['CH Spline'](t))*(365/t) for t in TIME_ARRAY]

CH_Spline_df = pd.DataFrame({"Tenor": TIME_ARRAY, "Bond Price": yc.interpolated_bond_curve["CH Spline"](TIME_ARRAY),
                             "Rates": rates_ch})
print(CH_Spline_df)

# Plot
plt.figure(figsize=(10, 5))
# sns.scatterplot(data=yc.bond_prices, x="Tenor", y="Bond Price", color="red", label="Original Data", s=80)
sns.lineplot(data=Lin_df, x="Tenor", y="Bond Price", color="red", label="Cubic Spline Interpolation")
sns.lineplot(data=PCHIP_df, x="Tenor", y="Bond Price", color="green", label="PCHIP Interpolation")
sns.lineplot(data=CH_Spline_df, x="Tenor", y="Bond Price", color="blue", label="CH Spline Interpolation")

plt.xlabel("Tenor (Years)")
plt.ylabel("Bond Price")
plt.title("Bond Price Interpolation")
plt.legend()
plt.grid(True)

plt.show()