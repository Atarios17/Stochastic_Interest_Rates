import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# scipy.integrate usage:
def f(x):
    return x**2
scipy.integrate.quad(f,0,1)

# Hull White One Factor: dr = (eta - gamma * r)dt + sqrt(beta)dW
# sqrt(beta) -> beta is a variance of r
# gamma ->
# H-W Bond: Z(r,t,T) = exp(A(t,T) - r* B(t,T))

class yield_curve:
    def __init__(self):
        self.market_rates = pd.DataFrame([], columns = ['Tenor', 'Rate']) # [r1,r2,..., rn]
        self.bond_prices = pd.DataFrame([], columns = ['Tenor', 'Bond Price']) # [Z1,Z2,..., Zn]
        self.interpolated_bond_curve = {} # {"Hermite": Z*(t), "Linear": Z*(t)}
        self.model_parameters = {} # {"Hull-White":{"Hermite":[eta,gamma,sigma],"Linear":[eta,gamma,sigma]},"Lee":{"Hermite":[a,b],"Linear":[a,b]}}

    def read_market_data(self, market_data_path):
        #for .csv
        self.market_rates = pd.read_csv(market_data_path)#,index_col=['Tenor', 'Rate'])

    def calculate_bond_prices(self):
        if len(self.market_rates) == 0:
            raise AssertionError('market_rates table is empty! Please use .read_market_data() first to populate it')

        self.bond_prices['Tenor'] = self.market_rates['Tenor']
        self.bond_prices['Bond Price'] = np.exp(-self.market_rates['Rate']/100 * self.market_rates['Tenor'] / 365)

    def interpolate_bond_prices(self, interpolation_method = "Linear"):

        supported_interpolation_methods = ["Linear", "PCHIP", "CH Spline"]

        if interpolation_method not in supported_interpolation_methods:
            raise ValueError(f'Inputted interpolation method is not supported. Please use any of the following ones: {str(supported_interpolation_methods)[1:-1]}')

        if len(self.bond_prices) == 0:
            raise AssertionError('bond_prices table is empty! Please use .calculate_bond_prices() first to populate it')

        if interpolation_method == "Linear": # Piecewise Linear
            params = pd.DataFrame([np.polyfit(self.bond_prices['Tenor'].values[i:(i + 2)],
                                              self.bond_prices['Bond Price'].values[i:(i + 2)], 1) for i in
                                              range(len(self.bond_prices) - 1)], columns=['a', 'b'])

            def f(t):
                if t < min(self.bond_prices['Tenor']):
                    return np.exp(-self.market_rates['Rate'][0] * t / 365)
                elif t > max(self.bond_prices['Tenor']):
                    a, b = params.iloc[-1, :].values
                    return max(a * t + b, 0)
                else:
                    wanted_row = max(np.where(t > self.bond_prices['Tenor'].values)[0])
                    a, b = params.iloc[wanted_row, :].values
                    return a * t + b

            # print(params)

            self.interpolated_bond_curve[interpolation_method] = f

        if interpolation_method == "PCHIP": # Piecewise Cubic Hermite Interpolation
            interpolator = scipy.interpolate.PchipInterpolator(self.bond_prices['Tenor'].values, self.bond_prices['Bond Price'].values)

            self.interpolated_bond_curve[interpolation_method] = interpolator

        if interpolation_method == "CH Spline": # Cubic Hermite Spline

            m = np.diff(self.bond_prices["Bond Price"]) / np.diff(self.bond_prices["Tenor"])    # slopes
            dy = np.zeros_like(self.bond_prices["Bond Price"])
            dy[1:-1] = (m[:-1] + m[1:]) / 2                                                     # interior points
            dy[0] = m[0]
            dy[-1] = m[-1]

            spline = scipy.interpolate.CubicHermiteSpline(self.bond_prices["Tenor"], self.bond_prices["Bond Price"], dy)

            def f(t):
                return spline(t)

            self.interpolated_bond_curve[interpolation_method] = f

def bond_rates_to_prices(market_rates):
    return np.exp(-market_rates['Rate'].values/100 * market_rates['Tenor'].values / 365)

#Market rates Data:
path = r"D:\Python\Pycharm\Stochastic Interest Rates\US_Treasury_Bonds_Live.csv"

# Create object
yc = yield_curve()

# Read market rates
yc.read_market_data(path)
#yc.market_rates

# Calculate Bond prices
yc.calculate_bond_prices()
#yc.bond_prices

# Interpolate bond prices
yc.interpolate_bond_prices(interpolation_method="Linear")
yc.interpolate_bond_prices(interpolation_method="PCHIP")




