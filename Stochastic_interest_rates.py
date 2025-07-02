import pandas as pd
import numpy as np
import scipy

tenor_years_map = pd.DataFrame([["1M" , 1/12],
                                ["1.5M", 0.125],
                                ["2M", 1/6],
                                ["3M" , 0.25],
                                ["4M" , 1/3],
                                ["6M" , 0.5],
                                ["1Y" , 1],
                                ["2Y" , 2],
                                ["3Y" , 3],
                                ["4Y" , 4],
                                ["5Y" , 5],
                                ["6Y" , 6],
                                ["7Y" , 7],
                                ["8Y" , 8],
                                ["9Y" , 9],
                                ["10Y" , 10],
                                ["15Y" , 15],
                                ["20Y" , 20],
                                ["25Y" , 25],
                                ["30Y" , 30],
                                ], columns=["Tenor","Years"])

def read_swaptions_market_data(path):
    global tenor_years_map
    swaptions_md = pd.read_csv(path)

    # Translate tenors to number of years, accordingly to above mapping
    swaptions_md['Option_Maturity_Y'] = pd.merge(swaptions_md, tenor_years_map, left_on='Option_Maturity', right_on='Tenor')['Years']
    swaptions_md['Swap_Length_Y'] = pd.merge(swaptions_md, tenor_years_map, left_on='Swap_Length', right_on='Tenor')['Years']
    swaptions_md['Swap_Freq_Y'] = pd.merge(swaptions_md, tenor_years_map, left_on='Swap_Freq', right_on='Tenor')['Years']

    # Change format of Strikes and Vols so they are in decimals instead of percentage points
    swaptions_md['Strike'] = swaptions_md['Strike'] * 0.01
    swaptions_md['LogNormal_Vol'] = swaptions_md['LogNormal_Vol'] * 0.01

    return swaptions_md

def read_bond_options_market_data(path):
    global tenor_years_map
    bond_options_md = pd.read_csv(path)

    # Translate tenors to number of years, accordingly to above mapping
    bond_options_md['Option_Maturity_Y'] = pd.merge(bond_options_md, tenor_years_map, left_on='Option_Maturity', right_on='Tenor')['Years']
    bond_options_md['Bond_Length_Y'] = pd.merge(bond_options_md, tenor_years_map, left_on='Bond_Length', right_on='Tenor')['Years']
    bond_options_md['Coupon_Freq_Y'] = pd.merge(bond_options_md, tenor_years_map, left_on='Coupon_Freq', right_on='Tenor')['Years']

    # Change format of Strikes and Vols so they are in decimals instead of percentage points
    bond_options_md['Strike'] = bond_options_md['Strike'] * 0.01
    bond_options_md['LogNormal_Vol'] = bond_options_md['LogNormal_Vol'] * 0.01

    return bond_options_md

# Logic behind Black Swaptions formula:
# https://www.quantpie.co.uk/black_formula/swaption_price.php
# def bond_curve(t):
#     if t == 1: return 0.97
#     elif t == 2: return 0.93
#     elif t == 3: return 0.88
#
# #vol,K,T1,T2,swap_freq = 0.2, 0.05, 1, 3,1

def swaption_price_black_model(K,vol,opt_mat,swap_len,swap_freq,bond_curve):
    A = np.sum(swap_freq*np.array([bond_curve(t) for t in np.arange(swap_freq,swap_len+swap_freq,swap_freq)]))
    F = (bond_curve(opt_mat)-bond_curve(opt_mat+swap_len))/A
    d1 = (np.log(F/K) + vol**2 * 0.5 * opt_mat) / np.sqrt(vol * np.sqrt(opt_mat))
    d2 = d1 - vol * np.sqrt(opt_mat)
    return A * bond_curve(opt_mat) * (F * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2))

swaption_price_black_model = np.vectorize(swaption_price_black_model)

def bond_price_HW1F_iter(r_t, r_T, t, T, alpha, sigma):
    """This function prices bond using Hull-White bond price formula: A(t, T) * np.exp(-B(t, T) * r)."""

    # A and B functions for Hull-White bond price formula
    B = (1 - np.exp(-alpha * (T - t))) / alpha
    A = np.exp((r_t - sigma ** 2 / (2 * alpha ** 2)) * (B - (T - t)) - (sigma ** 2 / (4 * alpha)) * B ** 2)

    return A * np.exp(-B * r_T)

bond_price_HW1F_iter = np.vectorize(bond_price_HW1F_iter)

def bond_price_HW1F(yield_curve, t, T, alpha, sigma):
    """This function prices bond using Hull-White bond price formula: A(t, T) * np.exp(-B(t, T) * r)."""

    # A and B functions for Hull-White bond price formula
    B = (1 - np.exp(-alpha * (T - t))) / alpha
    A = np.exp((yield_curve(t) - sigma ** 2 / (2 * alpha ** 2)) * (B - (T - t)) - (sigma ** 2 / (4 * alpha)) * B ** 2)

    return A * np.exp(-B * yield_curve(T))

bond_price_HW1F = np.vectorize(bond_price_HW1F)

def bond_option_price_HW1F(K, opt_mat, bond_len, yield_curve, alpha, sigma):

    p1 = bond_price_HW1F(yield_curve, 0, opt_mat, alpha, sigma)
    p2 = bond_price_HW1F(yield_curve, 0, opt_mat + bond_len, alpha, sigma)

    Sigma2 = sigma ** 2 / (2 * alpha ** 3) * (1 - np.exp(-2 * alpha * opt_mat)) * (1 - np.exp(-alpha * (bond_len))) ** 2

    d2 = (np.log(p2 / (K * p1)) - Sigma2 / 2) / np.sqrt(Sigma2)
    d1 = d2 + np.sqrt(Sigma2)

    return p2 * scipy.stats.norm.cdf(d1) - K * p1 * scipy.stats.norm.cdf(d2)

bond_option_price_HW1F = np.vectorize(bond_option_price_HW1F)

def jamshidian_swaption_price(yield_curve, option_mat, swap_len, K, delta, notional, alpha, sigma):
    """This function prices a European call option on a swap (in the Hull-White one-factor model) using so called Jamshidian trick."""

    # delta = time step
    payment_times = np.arange(option_mat+delta, option_mat+delta+swap_len, delta)

    # looking for r_star
    def mean_fit(r_T):
        return sum(delta * bond_price_HW1F_iter(yield_curve(0), r_T, option_mat, payment_times, alpha, sigma) - K)**2

    # We eed to pass "initial guess" values.
    sol = scipy.optimize.minimize(mean_fit, yield_curve(0))
    # r_star = sol.root
    #
    # sol = scipy.optimize.root_scalar(lambda r_T: sum(delta * bond_price_HW1F_iter(yield_curve(0), r_T,option_mat, payment_times, alpha, sigma) - K), bracket=[0.01, 0.15], method='brentq')
    r_star = sol.x[0]

    print(f"r_star = {r_star:.2f}")

    K_i = bond_price_HW1F_iter(yield_curve(0), r_star, option_mat, payment_times, alpha, sigma)
    bond_call_prices = bond_option_price_HW1F(K_i, option_mat, payment_times, yield_curve, alpha, sigma)
    swaption_price = sum(delta * bond_call_prices) * notional

    return  swaption_price

class yield_curve:
    def __init__(self):
        self.market_rates = pd.DataFrame([], columns=['Tenor', 'Rate'])  # [r1,r2,..., rn]
        self.yield_curve = {}  # {"Cubic Spline": Z*(t), "PCHIIP": Z*(t)}
        self.discount_curve = {}  # {"Cubic Spline": Z*(t), "PCHIIP": Z*(t)}
        self.model_parameters = {}  # {"Hull-White":{"Hermite":[eta,gamma,sigma],"Linear":[eta,gamma,sigma]},"Lee":{"Hermite":[a,b],"Linear":[a,b]}}

    def read_market_data(self, market_data_path):
        # We assume market rates are in format of two columns table:
        # col 1 should be named "Tenor" and should contain Tenor name contained in global tenors-years mapping.
        # col 2 should be named "Rate" and should contain rate in % points. For example 4.53 means 4.53%.
        self.market_rates = pd.read_csv(market_data_path)

        # Translate Tenor names to numbers (parts of year), using global tenor-years mapping
        global tenor_years_map
        self.market_rates['Tenor'] = pd.merge(self.market_rates['Tenor'], tenor_years_map, on='Tenor')['Years']

        # Translate rates to decimal numbers
        self.market_rates['Rate'] = self.market_rates['Rate']*0.01

    def interpolate_yield_and_discount_curves(self, interpolation_method="Cubic Spline"):
        """This function interpolates the Bond prices. Interpolation allows us to estimate the rates of the Bond. The default interpolation method is "Cubic Spline"."""

        supported_interpolation_methods = ["Linear","Cubic Spline", "PCHIP"]

        if interpolation_method not in supported_interpolation_methods:
            raise ValueError(f'Inputted interpolation method is not supported. Please use any of the following ones: {str(supported_interpolation_methods)[1:-1]}')

        if len(self.market_rates) == 0:
            raise AssertionError('market_rates table is empty! Please read market data using .read_market_data() to populate it first.')

        if interpolation_method == "Linear":  # Piecewise Linear Interpolation with flat ends
            class LinearFlatEnds:
                def __init__(self,x,y):
                    self.x = x
                    self.y = y
                    self.coeffs = pd.DataFrame(columns=['a','b'])
                    self.coeffs['a'] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
                    self.coeffs['b'] = y[1:] - self.coeffs['a']*x[1:]

                def __call__(self,t):
                    if t <= self.x[0]:
                        return self.y[0]

                    elif t > self.x[-1]:
                        return self.y[-1]

                    else:
                        # find which in which part lies the argument
                        idx = int(np.where(self.x < t)[0][-1])
                        return self.coeffs['a'].values[idx]*t+self.coeffs['b'].values[idx]

                def derivative(self, n):
                    if n == 1:
                        def f(t):
                            if t < self.x[0] or t > self.x[-1]:
                                return 0
                            else:
                                idx = int(np.where(self.x <= t)[0][-1])
                                return self.coeffs['a'].values[idx]

                        return f

                    elif n >= 2:
                        return lambda x: 0

                    else:
                        raise ValueError("Derivative is defined only for positive integer numbers")

            interpolated_curve = LinearFlatEnds(self.market_rates['Tenor'].values,self.market_rates['Rate'].values)


        elif interpolation_method == "Cubic Spline":  # Piecewise Cubic Hermite Interpolation
            # We overwrite behaviour of calling interpolated function in base scipy class
            # to obtain flat lines below first and above last point
            class CublicSplineFlatEnds(scipy.interpolate.CubicSpline):
                def __call__(self, t):
                    if t < self.x[0]:
                        return super().__call__(self.x[0])
                    elif t > self.x[-1]:
                        return super().__call__(self.x[-1])
                    else:
                        return super().__call__(t)

            interpolated_curve = CublicSplineFlatEnds(self.market_rates['Tenor'].values,self.market_rates['Rate'].values, extrapolate=False)

        elif interpolation_method == "PCHIP":  # Piecewise Cubic Hermite Interpolation
            class PchipInterpolatorFlatEnds(scipy.interpolate.PchipInterpolator):
                # We overwrite behaviour of calling interpolated function in base scipy class
                # to obtain flat lines below first and above last point
                def __call__(self, t):
                    if t < self.x[0]:
                        return super().__call__(self.x[0])
                    elif t > self.x[-1]:
                        return super().__call__(self.x[-1])
                    else:
                        return super().__call__(t)

            interpolated_curve = PchipInterpolatorFlatEnds(self.market_rates['Tenor'].values,self.market_rates['Rate'].values, extrapolate=False)

        class discount_curve:
            def __init__(self,interpolated_yield_curve):
                self.interpolated_yield_curve = interpolated_yield_curve

            def __call__(self, t):
                return np.exp(-t*self.interpolated_yield_curve(t))

            # First derivative is calculated manually:
            # d exp(-r(t)*t)/dt = exp(-r(t)*t)*(-r'(t)*t-r(t)*1) = -(r'(t)*t+r(t)) * exp(-r(t)*t)
            # Second derivative written down from Wolfram
            def derivative(self,k):
                if k not in (1,2):
                    raise NotImplementedError("Only first and second derivative of discount curve are supported")
                elif k == 1:
                    return lambda t: -(self.interpolated_yield_curve.derivative(1)(t)*t + self.interpolated_yield_curve(t)) * np.exp(-self.interpolated_yield_curve(t)*t)
                elif k == 2:
                    return lambda t: (-self.interpolated_yield_curve.derivative(2)(t) * t + self.interpolated_yield_curve.derivative(1)(t)**2 * t**2 +
                            2*t*self.interpolated_yield_curve(t)*self.interpolated_yield_curve.derivative(1)(t) - 2*self.interpolated_yield_curve.derivative(1)(t)
                            + self.interpolated_yield_curve(t)**2) * np.exp(-self.interpolated_yield_curve(t)*t)

        self.yield_curve[interpolation_method] = interpolated_curve
        self.discount_curve[interpolation_method] = discount_curve(interpolated_curve)

    def calibrate_term_structure_model(self, options_market_data, model = "Hull-White-1F", bond_curve_interpolation = "PCHIP"):
        """This function calibrates parameters of chosen term structure model. In particular for Hull-White 1-Factor Model it calibrates theta, alpha and sigma parameters using market swaption and Bond prices."""
        supported_term_structure_models = ["Hull-White-1F"] # "Ho-Lee"

        if model not in supported_term_structure_models:
            raise ValueError(f'Inputted term structure model is not supported. Please use any of the following ones: {str(supported_term_structure_models)[1:-1]}')

        if model == "Hull-White-1F":
            # 1-factor Hull White Model has three parameters to calibrate: function theta(t) and constants alpha and sigma.

            #Now we can fit alpha and sigma to market prices of chosen derivative instrument (example swaptions, bond options or bond future options)
            def mean_fit(params):
                alpha, sigma = params
                pred = bond_option_price_HW1F(options_market_data["Strike"].values, options_market_data["Option_Maturity_Y"].values, options_market_data["Bond_Length_Y"].values, self.discount_curve[bond_curve_interpolation], alpha, sigma)
                return np.mean((pred - options_market_data["Price"]) ** 2)

            # We eed to pass "initial guess" values.
            fitted_params = scipy.optimize.minimize(mean_fit,np.array([0.07,0.01]))
            alpha, sigma = fitted_params.x[0], fitted_params.x[1]

            # Theta function is calibrated by making model's bond prices equal market bond prices
            # This way theta can be expressed by market bond prices + alpha and sigma parameters.
            # We call theta(t) that assures matching market bond prices theta_star(t) and formula is:
            # theta_star(t) = d^2log(Z_star)/dt^2 - alpha * dlog(Z_star)/dt + sigma^2/(2*alpha) * (1-exp(-2*alpha*(T-t)))
            # Where Z_star is market bond price for tenor t. Derivatives are as follows:

            # dlog(Z_star)/dt = 1/Z_star * dZ_star/dt

            # d^2log(Z_star)/dt^2 =  -1/Z_star^2 * dZ_star/dt + 1/Z_star * d^2Z_star/dt^2 =
            # = - 1/Z_star * dlog(Z_star)/dt + 1/Z_star * d^2Z_star/dt^2 =
            # = ( d^2Z_star/dt^2 - dlog(Z_star)/dt ) / Z_star

            # Start with assignment of dZ_star/dt and d^2Z_star/dt^2:
            dZ_dt = self.discount_curve[bond_curve_interpolation].derivative(1)
            dZ_dt2 = self.discount_curve[bond_curve_interpolation].derivative(2)

            # then derivatives of logarithms:
            dlogZ_dt = lambda t: dZ_dt(t)/self.discount_curve[bond_curve_interpolation](t)
            dlogZ_dt2 = lambda t: (dZ_dt2-dlogZ_dt(t))/self.discount_curve[bond_curve_interpolation](t)

            # and we can express theta using the derivatives and alpha + sigma parameters:
            def theta_star(t):
                return -dlogZ_dt2(t) - alpha * dlogZ_dt(t) + 0.5 * sigma**2/alpha * (1 - np.exp(-2 * alpha * t))

            calibrated_params = [theta_star, alpha, sigma]

        # Add chosen term structure model to dictionary of calibrated models (if not present yet):
        if model not in self.model_parameters.keys():
            self.model_parameters[model] = {}

        # Assign calibrated parameters
        self.model_parameters[model][bond_curve_interpolation] = calibrated_params

    def forward_rate(self,t1,t2,interpolation_method="PCHIP"):
        return (np.log(self.discount_curve[interpolation_method](t1)) - np.log(self.discount_curve[interpolation_method](t2)))/(t2-t1)