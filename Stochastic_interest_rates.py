import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# scipy.integrate usage:
def f(x):
    return x ** 2


scipy.integrate.quad(f, 0, 1)


# Hull White One Factor: dr = (eta - gamma * r)dt + sqrt(beta)dW
# sqrt(beta) -> beta is a variance of r
# gamma ->
# H-W Bond: Z(r,t,T) = exp(A(t,T) - r* B(t,T))

class yield_curve:
    def __init__(self):
        self.market_rates = pd.DataFrame([], columns=['Tenor', 'Rate'])  # [r1,r2,..., rn]
        self.bond_prices = pd.DataFrame([], columns=['Tenor', 'Bond Price'])  # [Z1,Z2,..., Zn]
        self.interpolated_bond_curve = {}  # {"Hermite": Z*(t), "Linear": Z*(t)}
        self.model_parameters = {}  # {"Hull-White":{"Hermite":[eta,gamma,sigma],"Linear":[eta,gamma,sigma]},"Lee":{"Hermite":[a,b],"Linear":[a,b]}}

    def read_market_data(self, market_data_path):
        # for .csv
        self.market_rates = pd.read_csv(market_data_path)  # ,index_col=['Tenor', 'Rate'])

    def calculate_bond_prices(self):
        if len(self.market_rates) == 0:
            raise AssertionError('market_rates table is empty! Please use .read_market_data() first to populate it')

        self.bond_prices['Tenor'] = self.market_rates['Tenor']
        self.bond_prices['Bond Price'] = np.exp(-self.market_rates['Rate'] / 100 * self.market_rates['Tenor'] / 365)

    def interpolate_bond_prices(self, interpolation_method="Cubic Spline"):

        supported_interpolation_methods = ["Cubic Spline", "PCHIP", "CH Spline"]

        if interpolation_method not in supported_interpolation_methods:
            raise ValueError(
                f'Inputted interpolation method is not supported. Please use any of the following ones: {str(supported_interpolation_methods)[1:-1]}')

        if len(self.bond_prices) == 0:
            raise AssertionError('bond_prices table is empty! Please use .calculate_bond_prices() first to populate it')

        if interpolation_method == "Cubic Spline":  # Piecewise Cubic Hermite Interpolation
            interpolator = scipy.interpolate.CubicSpline(self.bond_prices['Tenor'].values,
                                                         self.bond_prices['Bond Price'].values)

            self.interpolated_bond_curve[interpolation_method] = interpolator

        if interpolation_method == "PCHIP":  # Piecewise Cubic Hermite Interpolation
            interpolator = scipy.interpolate.PchipInterpolator(self.bond_prices['Tenor'].values,
                                                               self.bond_prices['Bond Price'].values)

            self.interpolated_bond_curve[interpolation_method] = interpolator

        if interpolation_method == "CH Spline":  # Cubic Hermite Spline

            m = np.diff(self.bond_prices["Bond Price"]) / np.diff(self.bond_prices["Tenor"])  # slopes
            dy = np.zeros_like(self.bond_prices["Bond Price"])
            dy[1:-1] = (m[:-1] + m[1:]) / 2  # interior points
            dy[0] = m[0]
            dy[-1] = m[-1]

            spline = scipy.interpolate.CubicHermiteSpline(self.bond_prices["Tenor"], self.bond_prices["Bond Price"], dy)

            def f(t):
                return spline(t)

            self.interpolated_bond_curve[interpolation_method] = f

    def _calibrate(self, interpolation_method, gamma, alpha, sigma):

        def eta(t):
            index = min(np.where(t < self.bond_prices['Tenor'].values)[0])
            T = self.bond_prices["Tenor"][index]

            return (-self.interpolated_bond_curve[interpolation_method].derivative(2)(t) - self.interpolated_bond_curve[
                interpolation_method].derivative(1)(t)
                    + (sigma ** 2 / 2 * alpha) * (1 - np.exp(-2 * alpha * (T - t))))

        self.model_parameters[interpolation_method] = [eta, gamma, sigma]

