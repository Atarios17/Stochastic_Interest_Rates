from asyncore import poll2

import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math


class yield_curve:
    def __init__(self):
        self.market_rates = pd.DataFrame([], columns=['Tenor', 'Rate'])  # [r1,r2,..., rn]
        self.bond_prices = pd.DataFrame([], columns=['Tenor', 'Bond Price'])  # [Z1,Z2,..., Zn]
        self.interpolated_bond_curve = {}  # {"Cubic Spline": Z*(t), "PCHIIP": Z*(t)}
        self.model_parameters = {}  # {"Hull-White":{"Hermite":[eta,gamma,sigma],"Linear":[eta,gamma,sigma]},"Lee":{"Hermite":[a,b],"Linear":[a,b]}}

    def read_market_data(self, market_data_path):
        # for .csv
        self.market_rates = pd.read_csv(market_data_path)  # ,index_col=['Tenor', 'Rate'])

    def calculate_bond_prices(self):
        """Calculates Bond prices based on data containing Tenor and Rate."""

        if len(self.market_rates) == 0:
            raise AssertionError('market_rates table is empty! Please use .read_market_data() first to populate it')

        self.bond_prices['Tenor'] = self.market_rates['Tenor']
        self.bond_prices['Bond Price'] = np.exp(-self.market_rates['Rate'] / 100 * self.market_rates['Tenor'] / 365)

    def interpolate_bond_prices(self, interpolation_method="Cubic Spline"):
        """This function interpolates the Bond prices. Interpolation allows us to estimate the rates of the Bond. The default interpolation method is "Cubic Spline"."""

        supported_interpolation_methods = ["Cubic Spline", "PCHIP"]

        if interpolation_method not in supported_interpolation_methods:
            raise ValueError(
                f'Inputted interpolation method is not supported. Please use any of the following ones: {str(supported_interpolation_methods)[1:-1]}')

        if len(self.bond_prices) == 0:
            raise AssertionError('bond_prices table is empty! Please use .calculate_bond_prices() first to populate it')

        if interpolation_method == "Cubic Spline":  # Piecewise Cubic Hermite Interpolation
            cubicspline = scipy.interpolate.CubicSpline(self.bond_prices['Tenor'].values,
                                                         self.bond_prices['Bond Price'].values)

            self.interpolated_bond_curve[interpolation_method] = cubicspline

        if interpolation_method == "PCHIP":  # Piecewise Cubic Hermite Interpolation
            pchip = scipy.interpolate.PchipInterpolator(self.bond_prices['Tenor'].values,
                                                               self.bond_prices['Bond Price'].values)

            self.interpolated_bond_curve[interpolation_method] = pchip

    def _calibrate(self, interpolation_method, gamma, alpha, sigma):
        """This function calibrates parameters: eta, gamma, sigma using estimated Bond rates."""

        def eta(t):
            index = min(np.where(t < self.bond_prices['Tenor'].values)[0])
            T = self.bond_prices["Tenor"][index]

            dev_1 = self.interpolated_bond_curve[interpolation_method].derivative(1)
            dev_2 = self.interpolated_bond_curve[interpolation_method].derivative(2)

            return -dev_2(t) - alpha*dev_1(t) + (sigma ** 2 / 2 * alpha) * (1 - np.exp(-2 * alpha * (T - t)))

        self.model_parameters[interpolation_method] = [eta, gamma, sigma]

    def bond_option_price(self, interpolation_method, a, sigma, K, T1, T2):

        p1 = self.interpolated_bond_curve[interpolation_method](T1)
        p2 = self.interpolated_bond_curve[interpolation_method](T2)

        Sigma2 = sigma**2 / (2 * a**3) * (1 - np.exp(-2*a*T1)) * (1 - np.exp(-a*(T2-T1)))**2

        d2 = (np.log(p2 / (K*p1)) - Sigma2/2) / np.sqrt(Sigma2)
        d1 = d2 + np.sqrt(Sigma2)

        return p2 * scipy.stats.norm.cdf(d1) - K * p1 * scipy.stats.norm.cdf(d2)



# ⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⠟⢉⣕⡞⣼⣿⣿⣿⣿⣿⡟⠛⠛⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣳⡀⢻⣤⣴⣴⣶⣶⣶⣿⣿⣷⣤⣼⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⠀⠀⣼⢧⣿⣦⠬⡜⣼⡝⣿⢹⣿⣿⡟⣁⡀⣴⣿⡟⣹⠈⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣗⠾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⠀⢰⣿⢀⣿⢀⠨⢳⣷⢴⣧⣸⣟⡟⠉⢁⡾⠋⣀⡴⢃⡠⣠⣿⣻⠿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡝⠋⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⠀⣾⣧⢸⣻⠀⢩⡏⠀⣸⣯⣿⡿⠒⣲⠟⠘⣩⠿⣠⠏⣴⣿⢯⣟⣹⣄⡿⣿⣿⣿⠏⣿⣿⣿⣧⠀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⠀⣿⣿⣿⡇⠘⢸⡇⣰⣿⠏⣿⣀⡼⠧⢶⣶⣳⡞⢳⠞⣿⣣⠏⠀⢠⣞⣀⣸⣿⣿⠀⢻⣿⣿⣿⣯⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡦⣿⣷⣿⢹⠀⣯⣞⣧⣔⣯⣿⣯⡾⢃⣾⠟⠁⢀⣴⡟⢀⣼⡟⣾⡿⣐⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⣸⣿⣧⢹⣾⣴⣿⠋⡇⢸⣸⣿⣿⣿⣿⣿⣿⣯⣶⠟⠋⢀⣴⣫⢏⣴⠟⣽⣿⡟⠀⠈⢹⣿⣟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⠀⢳⠇⣿⣿⣿⣿⢹⡀⠃⢸⡜⣿⢹⣿⣷⣿⣿⡿⣷⠶⠚⠛⢛⡿⠋⣠⣾⣿⣿⣤⣦⣀⢸⡟⣠⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⢠⡞⣸⢻⣿⣿⣿⣾⣗⠄⠘⡇⣿⠚⠻⠿⠯⠿⠅⠀⠠⢄⠍⢨⡙⠊⠑⣿⣿⣯⣿⡿⠻⣿⢡⠇⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⠀⣾⢣⡟⢸⠀⣿⣿⣷⣿⣦⠀⣷⣹⣣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠛⠿⠿⡿⣁⣼⣿⡟⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⢠⣇⣼⣿⣾⠀⣿⣿⣮⡛⢿⣧⣽⡿⣿⣷⡤⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⠀⠀⠀⠶⠆⣰⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⠀⡼⣼⣿⡏⡟⣼⣿⣿⣿⡻⣿⠃⣸⡟⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣺⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⢰⢳⢻⡿⣠⡷⣿⣿⣿⣿⡏⢻⣴⠏⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⣼⣿⡛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⡞⡞⣼⣷⢻⡇⣸⣿⣿⣿⡿⡟⠅⣴⣿⡟⢤⣀⠀⠀⠀⠀⠈⢿⡟⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣷⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⢠⢻⢁⡿⠇⣸⠇⣿⣿⣿⣿⣇⣿⠞⣡⣿⣧⣤⣽⣷⣀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣶⣿⣿⣿⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⡞⣸⠃⢀⣿⢀⣿⣿⣿⣿⣿⢥⣾⣿⣿⣿⣿⣿⣿⣿⣷⣦⣄⣀⣤⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⠒⠺⠷⣏⡀⣼⡿⢸⣿⣿⣿⣏⣥⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣯⠀⠀⠀⠀⣀⠀⣿⣿⣿⣿⣿⣿⣿⠿⠿⠛⠻⣿⣿
# ⠀⠀⠀⠀⠈⠙⠳⢿⣿⣿⡟⢉⣰⣿⣿⣿⣿⡟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠻⢿⣿⣿⣷⣤⠖⠛⠉⠉⠀⠀⢸⡏⠁⠀⠀⠀⠀⠀⠀⣿⣿
# ⠀⠀⠀⠀⠤⢄⣀⠀⣨⡿⠟⢋⣿⡿⢯⣶⣝⣛⣛⣿⠿⢿⣿⢥⣠⠼⠻⣿⣟⣿⣿⣿⢿⣿⣿⡋⢤⡀⠀⠻⣌⣉⣻⣦⣀⠀⠀⠀⠀⣼⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿
# ⠀⠀⠀⠀⠀⠀⢀⣼⡟⠛⣛⣿⡿⠀⠉⢛⣻⣿⠿⠋⠀⢸⢹⠀⠀⠀⠀⠘⣾⣦⠉⠛⠷⢿⣿⣻⣦⡌⠢⡀⠘⣧⣙⡛⣿⣗⡚⠉⠉⠉⡇⠀⠀⢀⣀⣀⣠⣴⣿⣿
# ⠀⠀⠀⢀⡤⡶⣿⠋⣽⠛⣿⡿⢁⣤⢶⡋⠉⠁⠀⠀⠀⢸⠾⠆⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠈⠙⠛⣿⣦⡙⣏⠁⠈⣯⣙⠛⢿⣆⠀⣼⣷⣶⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⠀⢠⡟⢸⣷⣿⠉⡇⢀⣯⣴⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠻⢿⣟⢧⡤⠶⠛⠋⠉⠻⣿⣽⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣴⣶⣿⣧⣿⢃⠏⠀⡇⣼⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠲⠤⣀⠀⠀⠀⢻⡝⢿⣾⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣟⢿⣯⣎⠤⠔⡇⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠲⢴⣿⣿⣆⢻⣿⣿⣿⣿⣿⣿⣿⣿
# ⠀⣿⣿⡄⢿⠦⠀⢠⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⢄⡉⢿⣻⡄⢻⣿⣿⣿⣿⣿⣿⣿
# ⠀⢸⣿⣿⣾⣇⠀⠋⡏⠀⠀⠀⠀⣀⡤⠴⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣭⣻⣎⠀⠹⣵⡟⠙⠛⢿⣿⣿⣿⣿
# ⣴⣿⣿⣿⣻⣿⣆⢰⠁⠀⠀⠀⠘⠁⣶⠿⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠛⠋⠀⠀⢻⣠⣴⠀⣼⡟⠛⠛⠛
# ⣿⣟⠙⣿⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⠙⠻⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣾⣿⠁⠀⠀⠀
# ⡏⠁⠀⠘⢿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠀⠀⢠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⠃⠀⠀⠀⠀
# ⣄⠀⢀⡀⠈⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠃⠀⠀⠈⠻⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣷⣤⣀⠀⠀⠀
# ⣿⣷⣌⢿⣷⣍⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠃⠀⠀⠀⠀⠈⠊⣙⠷⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⡟⣎⣻⣼⣷⡆⠀
# ⣿⣿⣿⣷⣽⣿⣿⣿⣷⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠙⠊⠉⠛⠷⢶⣤⣄⠀⠀⠀⢀⣠⣼⣿⣿⣿⣿⣷⣿⣿⡿⠿⠃⠀
# ⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⡦⢶⣄⠀⠀⠀⠀⠀⠀⢀⣀⡤⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⢿⣿⣿⣿⠿⠟⠛⠉⠁⠀⠙⣄⠀⠀
# ⣸⣷⣿⣿⣿⢿⣿⣿⣿⣿⣿⡄⠈⠉⠉⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀
# ⠿⠿⢿⣿⣿⣾⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳
# ⠀⠀⠀⠁⠚⠉⠛⠙⠛⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⢹⡟⣿⣿⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡏⠉⢹⣏⠀⠉⠓⠤⣄⣀⡠⠖⠋⢳⣀⣀⣀⣀⣀⣀⣾⣐⣈⣦⣀⣀⡤⠤⠖⠒⠒⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢸⠇⠸⣿⡄⡇⠀⠀⠀⠀⠀⠀⠀⢀⣠⠤⠷⠶⠾⡊⠦⣄⣀⡤⠖⠉⠀⣀⣴⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⣾⢧⠟⠀⠀⠙⢧⡵⠀⢀⣀⠠⠤⠒⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠢⣤⣤⡴⠛⠉⠙⡆⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠒⠚⠛⠛⠛⠛⠒⠒⠒⠒⠋⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡨⠋⠀⢀⣴⡾⠃⢀⠇⠀⠀⢸⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡀⠀⠀⠀⠀⠀⠀⠀⣜⠁⢀⣼⠛⠛⠤⣰⣫⠖⠀⡄⢸⡦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⡀⠀⠀⠀⠘⢄⢸⢑⡷⢿⡿⢶⣄⠀⢈⠳⡄⣴⡧⢸⡏⠉⠳⡖⠲⠤⠤⢤⣄⣀⣠⡤⠴⠚⠉
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡠⠤⠖⣚⣿⠀⠀⢄⡀⠙⠲⣍⣠⠋⠐⣠⣾⣿⢧⣴⣣⠏⣰⡿⠀⠀⠀⠹⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠴⠒⠉⠁⠀⠀⠀⠀⠀⠈⢷⡀⠐⠭⣗⡀⡴⢁⢀⣼⣿⠂⠉⠢⣴⣁⡾⠋⠀⠀⠀⠀⠀⠹⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠑⠦⢄⣀⠀⠀⠀⠀⠀⢀⣠⡤⢒⠯⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠦⢄⡀⠙⠧⣴⡋⣠⠟⠒⣤⣠⢀⠝⣄⠀⠀⠀⠀⠀⠀⠀⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠈⠉⠉⠉⠉⠉⢉⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠦⣜⡽⠡⣤⣼⡯⣍⡉⢲⠊⠀⠀⠀⠀⠀⠀⠀⢸⡀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⢠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⡰⠟⠋⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣠⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⡴⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡀⠀⠀⠀⠀⠀⠀⠀
# ⡀⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢧⠀⠀⠀⠀⠀⠀⠀
# ⠃⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⢀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀
# ⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⡄⠀⠀⠀⠀
# ⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡇⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣆⣀⣀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠾⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠛⠉
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡆⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⠀⠀⢀⡇⡀⠀⠀⡰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡄⣤⢞⡞⠀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡉⡏⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡍⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀