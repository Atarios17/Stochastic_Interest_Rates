import pandas as pd
import numpy as np
import scipy
import QuantLib as ql

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

# Logic behind Black Swaptions formula:
# https://www.quantpie.co.uk/black_formula/swaption_price.php

def swaption_price_black_model(K,vol,opt_mat,swap_len,swap_freq,discount_curve):
    A = swap_freq*np.sum(np.array([discount_curve(opt_mat+t)/discount_curve(opt_mat) for t in np.arange(swap_freq,swap_len+swap_freq,swap_freq)]))
    S0 = 100*(1/discount_curve(0.01)-1)
    F = S0/discount_curve(opt_mat)
    d1 = (np.log(F/K) + vol**2 * 0.5 * opt_mat) / np.sqrt(vol * np.sqrt(opt_mat))
    d2 = d1 - vol * np.sqrt(opt_mat)
    return A * discount_curve(opt_mat) * (F * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2))

swaption_price_black_model = np.vectorize(swaption_price_black_model)

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

def bond_option_price_black_model(K,vol,opt_mat,bond_len,discount_curve):
    F = discount_curve(opt_mat+bond_len)/discount_curve(opt_mat)
    d1 = (np.log(F/K) + vol**2 * 0.5 * opt_mat) / np.sqrt(vol * np.sqrt(opt_mat))
    d2 = d1 - vol * np.sqrt(opt_mat)
    return discount_curve(opt_mat) * (F * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2))

# r_t and alpha are used only if we want to use r_t not matching market
def bond_price_HW1F(discount_curve, t, T, alpha, r_t = None):
    """This function prices bond using Hull-White bond price formula: np.exp(A(t, T) - B(t, T) * r)"""

    if r_t is None:
        return discount_curve(T)/discount_curve(t)

    market_r_t = (discount_curve(t) / discount_curve(t + 0.0001) - 1) / 0.0001

    # A and B functions for Hull-White bond price formula
    B = (1 - np.exp(-alpha * (T - t))) / alpha
    A = np.log(discount_curve(T)/discount_curve(t)) + market_r_t * B

    return np.exp(A - r_t * B)

def bond_option_price_HW1F(K, opt_mat, bond_len, discount_curve, alpha, sigma, is_call = True,r_t = None):
    p1 = bond_price_HW1F(discount_curve = discount_curve, t = 0, T = opt_mat, alpha = alpha, r_t = r_t)
    p2 = bond_price_HW1F(discount_curve = discount_curve, t = 0, T = opt_mat + bond_len, alpha= alpha, r_t = r_t)

    Sigma2 = sigma ** 2 / (2 * alpha ** 3) * (1 - np.exp(-2 * alpha * opt_mat)) * (1 - np.exp(-alpha * (bond_len))) ** 2

    d2 = (np.log(p2 / (K * p1)) - Sigma2 / 2) / np.sqrt(Sigma2)
    d1 = d2 + np.sqrt(Sigma2)

    if is_call:
        return p2 * scipy.stats.norm.cdf(d1) - K * p1 * scipy.stats.norm.cdf(d2)

    else :
        return K * p1 * scipy.stats.norm.cdf(-d2) - p2 * scipy.stats.norm.cdf(-d1)

# r_t = forward_rate (t, t+1/12, "PCHIP")

# alpha = 0.07
# sigma = 0.01
# discount_curve = yc.discount_curve[bond_curve_interpolation],
# option_mat = swaptions_market_data["Option_Maturity_Y"].values,
# swap_len = swaptions_market_data["Swap_Length_Y"].values,
# strike_fixed_rate = swaptions_market_data["Strike"].values,
# swaptions_market_data["Swap_Freq_Y"].values, 1, alpha, sigma, True)

def jamshidian_swaption_price(discount_curve, option_mat, swap_len, strike_fixed_rate, delta, notional, alpha, sigma,
                              is_payer=False):
    #"""This function prices a European call option on a swap (in the Hull-White one-factor model) using so called Jamshidian trick."""

    # delta = swap payment frequency
    payment_times = np.arange(option_mat + delta, option_mat + delta + swap_len, delta)

    # looking for r_star
    def mean_fit(r_star):
        return (sum(bond_price_HW1F(discount_curve, option_mat, option_mat + np.arange(delta, swap_len + delta, delta), alpha,r_t=r_star)) * strike_fixed_rate * delta + bond_price_HW1F(discount_curve, option_mat,option_mat + swap_len, alpha,r_t=r_star) - 1) ** 2

    # We need to pass "initial guess" value, so we take short forward rate at maturity
    sol = scipy.optimize.minimize(mean_fit, 100 * (discount_curve(option_mat) / discount_curve(option_mat + 0.01) - 1))
    r_star = sol.x[0]

    K_i = bond_price_HW1F(discount_curve=discount_curve, r_t=r_star, t=option_mat, T=payment_times, alpha=alpha)

    if is_payer:
        bond_call_prices = bond_option_price_HW1F(K=K_i, opt_mat=option_mat, bond_len=payment_times - option_mat,
                                                  discount_curve=discount_curve, alpha=alpha, sigma=sigma,
                                                  is_call=False)
    else:
        bond_call_prices = bond_option_price_HW1F(K=K_i, opt_mat=option_mat, bond_len=payment_times - option_mat,
                                                  discount_curve=discount_curve, alpha=alpha, sigma=sigma, is_call=True)

    return sum(delta * strike_fixed_rate * bond_call_prices) * notional + bond_call_prices[-1] * notional

jamshidian_swaption_price = np.vectorize(jamshidian_swaption_price)

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

                # Works both on t being a number and t being a vector
                def __call__(self,t):
                    t = np.clip(t,self.x[0],self.x[-1])
                    # Find index/indices in which intervals argument(s) lie (searchsored-1)
                    # and since for arguments above self.x[-1] it gives index that is one too big, we cut it with minimum
                    idx = np.minimum(np.searchsorted(self.x, t, side='right') - 1, len(self.coeffs)-1)
                    return self.coeffs['a'].values[idx]*t+self.coeffs['b'].values[idx]

                def derivative(self, n):
                    if n == 1:
                        def f(t):

                            if np.size(t) == 1:
                                if t < self.x[0] or t >= self.x[-1]:
                                    return 0

                                else:
                                    return self.coeffs['a'].values[np.searchsorted(self.x, np.clip(t, self.x[0], self.x[-1]), side='right') - 1]

                            else:
                                # Find indices and assign "a" coefficient as derivative:
                                idx = np.minimum(np.searchsorted(self.x, np.clip(t, self.x[0], self.x[-1]), side='right') - 1, 2)
                                output = self.coeffs['a'].values[idx]
                                # But for values that are outside of interpolation bounds (i.e. lie on the flat ends) put 0 as derivative
                                output[np.where(np.logical_and(t < self.x[0], t >= self.x[-1]))] = 0
                                return output

                        return f

                    elif n >= 2:
                        def f(t):
                            if np.size(t) == 1:
                                return lambda t: 0
                            else:
                                return np.repeat(0,np.size(t))
                        return f

                    else:
                        raise ValueError("Derivative is defined only for positive integer numbers")

            interpolated_curve = LinearFlatEnds(self.market_rates['Tenor'].values,self.market_rates['Rate'].values)


        elif interpolation_method == "Cubic Spline":  # Piecewise Cubic Hermite Interpolation
            # We overwrite behaviour of calling interpolated function in base scipy class
            # to obtain flat lines below first and above last point
            class CublicSplineFlatEnds(scipy.interpolate.CubicSpline):
                def __call__(self, t):
                    t = np.clip(t, self.x[0], self.x[-1])
                    return super().__call__(t)

            interpolated_curve = CublicSplineFlatEnds(self.market_rates['Tenor'].values,self.market_rates['Rate'].values, extrapolate=False)

        elif interpolation_method == "PCHIP":  # Piecewise Cubic Hermite Interpolation
            class PchipInterpolatorFlatEnds(scipy.interpolate.PchipInterpolator):
                # We overwrite behaviour of calling interpolated function in base scipy class
                # to obtain flat lines below first and above last point
                def __call__(self, t):
                    t = np.clip(t,self.x[0],self.x[-1])
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

    def calibrate_term_structure_model(self, options_market_data, model = "Hull-White-1F", bond_curve_interpolation = "PCHIP", is_swaptions_market_data = True):
        """This function calibrates parameters of chosen term structure model. In particular for Hull-White 1-Factor Model it calibrates theta, alpha and sigma parameters using market swaption and Bond prices."""
        supported_term_structure_models = ["Hull-White-1F"] # "Ho-Lee"

        if is_swaptions_market_data:
            options_market_data["Price"] = swaption_price_black_model(options_market_data["Strike"],options_market_data["LogNormal_Vol"],options_market_data["Option_Maturity_Y"],options_market_data["Swap_Length_Y"],options_market_data["Swap_Freq_Y"],self.discount_curve[bond_curve_interpolation])

        else:
            options_market_data["Price"] = bond_option_price_black_model(options_market_data["Strike"],
                                                                         options_market_data["LogNormal_Vol"],
                                                                         options_market_data["Option_Maturity_Y"],
                                                                         options_market_data["Bond_Length_Y"],
                                                                         self.discount_curve[bond_curve_interpolation])

        if model not in supported_term_structure_models:
            raise ValueError(f'Inputted term structure model is not supported. Please use any of the following ones: {str(supported_term_structure_models)[1:-1]}')

        if model == "Hull-White-1F":
            # 1-factor Hull White Model has three parameters to calibrate: function theta(t) and constants alpha and sigma.

            if is_swaptions_market_data:
                def mean_fit(params):
                    alpha, sigma = params
                    pred = jamshidian_swaption_price(self.discount_curve[bond_curve_interpolation],options_market_data["Option_Maturity_Y"].values,
                                                     options_market_data["Swap_Length_Y"].values,options_market_data["Strike"].values,
                                                     options_market_data["Swap_Freq_Y"].values,1,alpha,sigma,True)
                    return 100*np.mean((pred - options_market_data["Price"]) ** 2)
            else:
                def mean_fit(params):
                    alpha, sigma = params
                    pred = bond_option_price_HW1F(K=options_market_data["Strike"].values,
                                                  opt_mat=options_market_data["Option_Maturity_Y"].values,
                                                  bond_len=options_market_data["Bond_Length_Y"].values,
                                                  discount_curve=self.discount_curve[bond_curve_interpolation],
                                                  alpha=alpha, sigma=sigma)
                    return 100*np.mean((pred - options_market_data["Price"]) ** 2)

            # We eed to pass "initial guess" values.
            fitted_params = scipy.optimize.minimize(mean_fit,np.array([0.05,0.05]), bounds=[(0.01,0.20),(0.01,0.20)])
            print("Calibration Finished with diff values: " + str(fitted_params.fun))
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
            # For clarity
            Z = self.discount_curve[bond_curve_interpolation]
            # Then we assign dZ_star/dt and d^2Z_star/dt^2:
            dZ_dt = self.discount_curve[bond_curve_interpolation].derivative(1)
            dZ_dt2 = self.discount_curve[bond_curve_interpolation].derivative(2)

            # and we calculate derivatives of logarithms:
            dlogZ_dt = lambda t: dZ_dt(t)/self.discount_curve[bond_curve_interpolation](t)
            dlogZ_dt2 = lambda t: (Z(t)*dZ_dt2(t) - dZ_dt(t)**2) / (Z(t)**2)

            # Now theta can be expressed using above derivatives and alpha + sigma parameters:
            def theta(t):
                return -dlogZ_dt2(t) - alpha * dlogZ_dt(t) + 0.5 * sigma**2/alpha * (1 - np.exp(-2 * alpha * t))

            calibrated_params = {"theta":theta, "alpha": alpha, "sigma": sigma}

        # Add chosen term structure model to dictionary of calibrated models (if not present yet):
        if model not in self.model_parameters.keys():
            self.model_parameters[model] = {}

        # Assign calibrated parameters
        self.model_parameters[model][bond_curve_interpolation] = calibrated_params

    def forward_rate(self,t1,t2,interpolation_method="PCHIP"):
        return (self.discount_curve[interpolation_method](t1)/self.discount_curve[interpolation_method](t2)-1)/(t2-t1)


def ql_create_swap(ytsh, swap_start: int = 2, swap_len: int = 2, frequency=0.25, fixed_rate=0.04,
                   todays_date=ql.Date(18, 6, 2025), notional=1, r_0=0.042):

    day_count = ql.Thirty360(ql.Thirty360.BondBasis)
    start = todays_date + ql.Period(swap_start, ql.Years)
    maturity = start + ql.Period(swap_len, ql.Years)
    fix_schedule = ql.MakeSchedule(start, maturity, ql.Period(int(frequency * 12), ql.Months))
    float_schedule = ql.MakeSchedule(start, maturity, ql.Period(int(frequency * 12), ql.Months))

    customIndex = ql.IborIndex('MyIndex', ql.Period('1Y'), 0, ql.USDCurrency(), ql.UnitedStates(0),
                               ql.ModifiedFollowing, False, day_count, ytsh)

    customIndex.addFixing(todays_date, r_0)

    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, notional, fix_schedule, fixed_rate, day_count,
                          float_schedule, customIndex, 0, day_count)

    swap.setPricingEngine(ql.DiscountingSwapEngine(ytsh))

    return swap

def ql_create_swaption(ql_swap, ql_hull_white, ytsh):
    exercise = ql.EuropeanExercise(ql_swap.startDate())
    swaption = ql.Swaption(ql_swap, exercise)
    Jamshidian_engine = ql.JamshidianSwaptionEngine(ql_hull_white, ytsh)
    swaption.setPricingEngine(Jamshidian_engine)
    return swaption