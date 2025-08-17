import pandas as pd
import numpy as np
import scipy
import QuantLib as ql
from joblib import Parallel, delayed

def translate_tenors(tenor: str) -> float:
    """This function translates tenors from strings of the form of '6D' / '6M' / '6Y'
    into float representing fraction of year"""

    if tenor[-1] == 'M':
        return float(tenor[:(-1)])/12

    elif tenor[-1] == 'Y':
        return float(tenor[:(-1)])

    elif tenor[-1] == 'D':
        return float(tenor[:(-1)])/365

translate_tenors = np.vectorize(translate_tenors)

def read_swaptions_market_data(path: str) -> pd.DataFrame:
    """This function reads swaptions market data in format in line with our input and translates tenors into floats"""

    swaptions_md = pd.read_csv(path)

    # Translate tenors to number of years, accordingly to above mapping
    swaptions_md['Option_Maturity_Y'] = translate_tenors(swaptions_md['Option_Maturity'].values)

    swaptions_md['Swap_tenor_Y'] = translate_tenors(swaptions_md['Swap_tenor'].values)

    swaptions_md['Swap_Freq_Y'] = translate_tenors(swaptions_md['Swap_Freq'].values)

    # Change format of Strikes and Vols so they are in decimals instead of percentage points
    swaptions_md['Strike'] = swaptions_md['Strike'] * 0.01
    swaptions_md['LogNormal_Vol'] = swaptions_md['LogNormal_Vol'] * 0.01

    return swaptions_md

# Logic behind Black Swaptions formula:
# https://www.quantpie.co.uk/black_formula/swaption_price.php


def swaption_price_black_model(K: float, vol: float, opt_mat: float, swap_tenor: float,
                               swap_freq: float, discount_curve: callable) -> float:
    """The function calculates swaption price with Black Formula"""

    A = swap_freq*np.sum(np.array([discount_curve(opt_mat+t)/discount_curve(opt_mat)
                                   for t in np.arange(swap_freq,swap_tenor+swap_freq,swap_freq)]))
    S0 = 100*(1/discount_curve(0.01)-1)
    F = S0/discount_curve(opt_mat)
    d1 = (np.log(F/K) + vol**2 * 0.5 * opt_mat) / np.sqrt(vol * np.sqrt(opt_mat))
    d2 = d1 - vol * np.sqrt(opt_mat)
    return A * discount_curve(opt_mat) * (F * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2))


swaption_price_black_model = np.vectorize(swaption_price_black_model)


def read_bond_options_market_data(path: str) -> pd.DataFrame:
    """The function calculates swaption price with Black Formula"""

    bond_options_md = pd.read_csv(path)

    # Translate tenors to number of years, accordingly to above mapping
    bond_options_md['Option_Maturity_Y'] = translate_tenors(bond_options_md['Option_Maturity'].values)

    bond_options_md['Bond_tenor_Y'] = translate_tenors(bond_options_md['Bond_tenor'].values)

    # Change format of Strikes and Vols so they are in decimals instead of percentage points
    bond_options_md['Strike'] = bond_options_md['Strike'] * 0.01
    bond_options_md['LogNormal_Vol'] = bond_options_md['LogNormal_Vol'] * 0.01

    return bond_options_md


def bond_option_price_black_model(K: float, vol: float, opt_mat: float,
                                  bond_tenor: float, discount_curve: callable) -> float:
    """The function calculates bond option price with Black Formula"""

    F = discount_curve(opt_mat+bond_tenor)/discount_curve(opt_mat)
    d1 = (np.log(F/K) + vol**2 * 0.5 * opt_mat) / np.sqrt(vol * np.sqrt(opt_mat))
    d2 = d1 - vol * np.sqrt(opt_mat)
    return discount_curve(opt_mat) * (F * scipy.stats.norm.cdf(d1) - K * scipy.stats.norm.cdf(d2))


# r_t and alpha are used only if we want to use r_t not matching market
def bond_price_HW1F(yield_curve: callable, t: float, T: float, alpha: float, sigma: float, r_t: float = None) -> float:
    """This function prices bond using Hull-White bond price formula: np.exp(A(t, T) - B(t, T) * r)"""

    discount_curve = lambda t: np.exp(-t * yield_curve(t))

    market_r_t = (discount_curve(t) / discount_curve(t + 0.0001) - 1) / 0.0001

    if r_t is None:
        r_t = market_r_t

    # A and B functions for Hull-White bond price formula
    B = (1 - np.exp(-alpha * (T - t))) / alpha
    A = np.log(discount_curve(T)/discount_curve(t)) + market_r_t * B

    # Because Z(t) = exp(-r_t * t) then log(Z(t)) = -r_t * t
    # dlog(Z(t))/dt = - t * r_t' - r_t
    # d^2log(Z_star)/dt^2 =  -t * r_t'' - 2 * r_t'
    # Start with assignment of dZ_star/dt and d^2Z_star/dt^2:
    d_rate_dt = yield_curve.derivative(1)
    dlogZ_dt = lambda t: -t * d_rate_dt(t) - yield_curve(t)

    A = np.log(discount_curve(T) / discount_curve(t)) - B * dlogZ_dt(t) \
        - 0.25*sigma**2/alpha**3*(np.exp(-alpha*T)-np.exp(-alpha*t))**2*(np.exp(2*alpha*t)-1)

    return np.exp(A - r_t * B)


def bond_option_price_HW1F(K: float, opt_mat: float, bond_tenor: float, yield_curve: callable,
                           alpha: float, sigma: float, is_call: bool = True, r_t: float = None) -> float:
    """The function calculates bond option price for Hull White Model"""

    p1 = bond_price_HW1F(yield_curve = yield_curve, t = 0, T = opt_mat, alpha = alpha, sigma = sigma, r_t = r_t)
    p2 = bond_price_HW1F(yield_curve = yield_curve, t = 0, T = opt_mat + bond_tenor, alpha= alpha, sigma = sigma, r_t = r_t)

    Sigma2 = sigma ** 2 / (2 * alpha ** 3) * (1 - np.exp(-2 * alpha * opt_mat)) * (1 - np.exp(-alpha * (bond_tenor))) ** 2

    d2 = (np.log(p2 / (K * p1)) - Sigma2 / 2) / np.sqrt(Sigma2)
    d1 = d2 + np.sqrt(Sigma2)

    if is_call:
        return p2 * scipy.stats.norm.cdf(d1) - K * p1 * scipy.stats.norm.cdf(d2)

    else :
        return K * p1 * scipy.stats.norm.cdf(-d2) - p2 * scipy.stats.norm.cdf(-d1)


def jamshidian_swaption_price(yield_curve: callable, option_mat: float, swap_tenor: float, strike_fixed_rate: float,
                              delta: float, notional: float, alpha: float, sigma: float, is_payer: bool = False) -> float:
    """This function calculates swaption prices in the Hull-White model using so called Jamshidian trick."""

    # delta = swap payment frequency
    payment_times = np.arange(option_mat + delta, option_mat + delta + swap_tenor, delta)

    # looking for r_star
    def mean_fit(r_star):
        return (sum(bond_price_HW1F(yield_curve, option_mat, option_mat + np.arange(delta, swap_tenor + delta, delta),
                                    alpha, sigma, r_t=r_star)) * strike_fixed_rate * delta
                + bond_price_HW1F(yield_curve, option_mat, option_mat + swap_tenor, alpha, sigma, r_t=r_star) - 1) ** 2

    # We need to pass "initial guess" value, so we take short forward rate at maturity
    sol = scipy.optimize.minimize(mean_fit, 100 * (np.exp(yield_curve(option_mat+0.01)-yield_curve(option_mat))- 1))
    r_star = sol.x[0]

    K_i = bond_price_HW1F(yield_curve=yield_curve, r_t=r_star, t=option_mat, T=payment_times, alpha=alpha, sigma=sigma)

    if is_payer:
        bond_call_prices = bond_option_price_HW1F(K=K_i, opt_mat=option_mat, bond_tenor=payment_times - option_mat,
                                                  yield_curve=yield_curve, alpha=alpha, sigma=sigma,
                                                  is_call=False)
    else:
        bond_call_prices = bond_option_price_HW1F(K=K_i, opt_mat=option_mat, bond_tenor=payment_times - option_mat,
                                                  yield_curve=yield_curve, alpha=alpha, sigma=sigma, is_call=True)

    return sum(delta * strike_fixed_rate * bond_call_prices) * notional + bond_call_prices[-1] * notional


jamshidian_swaption_price = np.vectorize(jamshidian_swaption_price)


class HullWhiteModel:
    def __init__(self):
        self.market_rates = pd.DataFrame([], columns=['Tenor', 'Rate'])  # [r1,r2,..., rn]
        self.yield_curve = None
        self.discount_curve = None
        self.alpha = None
        self.sigma = None
        self.theta = None

    def read_market_data(self, market_data_path: str) -> None:
        # We assume market rates are in format of two columns table:
        # col 1 should be named "Tenor" and should contain Tenor name contained in tenors-years mapping.
        # col 2 should be named "Rate" and should contain rate in % points. For example 4.53 means 4.53%.
        self.market_rates = pd.read_csv(market_data_path)

        # Translate Tenor names to numbers (parts of year), using tenor-years mapping
        self.market_rates['Tenor'] = translate_tenors(self.market_rates['Tenor'].values)

        # Translate rates to decimal numbers
        self.market_rates['Rate'] = self.market_rates['Rate']*0.01

    def interpolate_yield_and_discount_curves(self, interpolation_method: str = "Cubic Spline") -> None:
        """This function interpolates yield curve and bond prices to market quotes.
        The default interpolation method is "Cubic Spline"."""

        supported_interpolation_methods = ["Cubic Spline", "PCHIP"]

        if len(self.market_rates) == 0:
            raise AssertionError('market_rates table is empty! Please read market data using '
                                 '.read_market_data() to populate it first.')

        if interpolation_method not in supported_interpolation_methods:
            raise ValueError(f'Inputted interpolation method is not supported. Please use any of '
                             f'the following ones: {str(supported_interpolation_methods)[1:-1]}')

        elif interpolation_method == "Cubic Spline":  # Piecewise Cubic Hermite Interpolation
            # We overwrite behaviour of calling interpolated function in base scipy class
            # to obtain flat lines above last point
            class CublicSplineFlatEnds(scipy.interpolate.CubicSpline):
                def __call__(self, t:  float | np.ndarray) -> np.float64 | np.ndarray:
                    t = np.minimum(t, self.x[-1])
                    return 1*super().__call__(t)
                    # Multiplication by 1, so in case of t being a float it will return a float

            interpolated_curve = CublicSplineFlatEnds(self.market_rates['Tenor'].values,
                                                      self.market_rates['Rate'].values,
                                                      extrapolate=True)

        elif interpolation_method == "PCHIP":  # Piecewise Cubic Hermite Interpolation
            class PchipInterpolatorFlatEnds(scipy.interpolate.PchipInterpolator):
                # We overwrite behaviour of calling interpolated function in base scipy class
                # to obtain flat lines above last point
                def __call__(self, t:  float | np.ndarray) -> np.float64 | np.ndarray:
                    t = np.minimum(t, self.x[-1])
                    # We multiply output by 1, so in case of t being a float it will return a float
                    return 1*super().__call__(t)

            interpolated_curve = PchipInterpolatorFlatEnds(self.market_rates['Tenor'].values,
                                                           self.market_rates['Rate'].values,
                                                           extrapolate=True)

        def discount_curve(t: float | np.ndarray) -> float | np.ndarray:
            """This function calculates discount rate(s) for given time moment(s)"""
            return np.exp(-t*interpolated_curve(t))

        self.yield_curve = interpolated_curve
        self.discount_curve = discount_curve

    def calibrate_model_params(self, options_market_data: pd.DataFrame, is_swaptions_market_data: bool = True) -> None:
        """This function calibrates parameters alpha, sigma and theta of Hull White 1-FactOr model. Parameters alpha and
         sigma can be calibrated either to bond options or swaptions market data (which is default setting),
          while theta is calibrated to market yield curve."""

        # 1-factor Hull White Model has three parameters to calibrate: function theta(t) and constants alpha and sigma.

        # If calibration to swaptions is chosen
        if is_swaptions_market_data:
            options_market_data["Price"] = swaption_price_black_model(options_market_data["Strike"],
                                                                      options_market_data["LogNormal_Vol"],
                                                                      options_market_data["Option_Maturity_Y"],
                                                                      options_market_data["Swap_tenor_Y"],
                                                                      options_market_data["Swap_Freq_Y"],
                                                                      self.discount_curve)
            def mean_fit(params):
                alpha, sigma = params

                # Parallel over each option instead of calling jamshidian for full arrays | n_jobs=-1 <=> use all cores
                prices = Parallel(n_jobs=-1)(
                    delayed(jamshidian_swaption_price)(self.yield_curve,[maturity], [swap_tenor], [strike],
                                                       [freq], 1, alpha, sigma, True)
                    for maturity, swap_tenor, strike, freq
                    in zip(options_market_data["Option_Maturity_Y"].values,
                           options_market_data["Swap_tenor_Y"].values,
                           options_market_data["Strike"].values,
                           options_market_data["Swap_Freq_Y"].values)
                    )

                pred = np.array(prices).flatten()
                return 100 * np.mean((pred - options_market_data["Price"].values) ** 2)

        #If calibration to bond options is chosen
        else:
            options_market_data["Price"] = bond_option_price_black_model(options_market_data["Strike"],
                                                                         options_market_data["LogNormal_Vol"],
                                                                         options_market_data["Option_Maturity_Y"],
                                                                         options_market_data["Bond_tenor_Y"],
                                                                         self.discount_curve)

            def mean_fit(params):
                alpha, sigma = params
                pred = bond_option_price_HW1F(K=options_market_data["Strike"].values,
                                              opt_mat=options_market_data["Option_Maturity_Y"].values,
                                              bond_tenor=options_market_data["Bond_tenor_Y"].values,
                                              yield_curve=self.yield_curve,
                                              alpha=alpha, sigma=sigma)
                return 100*np.mean((pred - options_market_data["Price"]) ** 2)

        # We need to pass "initial guess" values.
        fitted_params = scipy.optimize.minimize(mean_fit, np.array([0.05,0.05]), bounds=[(0.01,0.20),(0.01,0.20)])

        print("Calibration Finished with mean price diff: {:.4f}".format(fitted_params.fun))
        alpha, sigma = fitted_params.x[0], fitted_params.x[1]

        # Theta function is calibrated by making model's bond prices equal market bond prices
        # This way theta can be expressed by market bond prices + alpha and sigma parameters.
        # We call theta(t) that assures matching market bond prices theta_star(t) and formula is:
        # theta_star(t) = d^2log(Z_star)/dt^2 - alpha * dlog(Z_star)/dt + sigma^2/(2*alpha) * (1-exp(-2*alpha*(T-t)))
        # Where Z_star is market bond price for tenor t. Derivatives are as follows:

        # Because Z(t) = exp(-r_t * t) then log(Z(t)) = -r_t * t
        # dlog(Z(t))/dt = - t * r_t' - r_t
        # d^2log(Z_star)/dt^2 =  -t * r_t'' - 2 * r_t'

        # Start with assignment of dZ_star/dt and d^2Z_star/dt^2:
        d_rate_dt = self.yield_curve.derivative(1)
        d_rate_dt2 = self.yield_curve.derivative(2)

        # then derivatives of logarithms:
        dlogZ_dt = lambda t: -t*d_rate_dt(t) - self.yield_curve(t)
        dlogZ_dt2 = lambda t: -t*d_rate_dt2(t) - 2*d_rate_dt(t)

        # and we can express theta using the derivatives and alpha + sigma parameters:
        def theta(t):
           return -dlogZ_dt2(t) - alpha * dlogZ_dt(t) + 0.5 * sigma**2/alpha * (1 - np.exp(-2 * alpha * t))

        # Assign calibrated parameters
        self.alpha, self.sigma, self.theta = alpha, sigma, theta

    def forward_rate(self, t1: float, t2 : float) -> float:
        return (self.discount_curve(t1)/self.discount_curve(t2)-1)/(t2-t1)

    def simulate_rate_paths(self, T: float, N_paths: int, dt: float = 0.0025) -> tuple:
        n_steps = int(T / dt)
        time_vec = np.linspace(0, T, n_steps + 1)

        theta_vals = self.theta(time_vec)
        rates_sims = np.zeros((N_paths, n_steps + 1))
        rates_sims[:, 0] = self.yield_curve(0)

        for i in range(1, n_steps + 1):
            drift = theta_vals[i - 1] - self.alpha*rates_sims[:, i - 1]
            dW = np.random.normal(0, np.sqrt(dt), N_paths)
            rates_sims[:, i] = rates_sims[:, i - 1] + drift*dt + self.sigma*dW

        return (time_vec, rates_sims)


def ql_create_swap(ytsh, swap_start: int = 2, swap_tenor: int = 2, frequency: float = 0.25, fixed_rate: float = 0.04,
                   todays_date: ql.Date = ql.Date(18, 6, 2025), notional: float = 1.0, r_0: float = 0.042,
                   is_payer: bool = True) -> ql.VanillaSwap:

    day_count = ql.Thirty360(ql.Thirty360.BondBasis)
    start = todays_date + ql.Period(swap_start, ql.Years)
    maturity = start + ql.Period(swap_tenor, ql.Years)
    fix_schedule = ql.MakeSchedule(start, maturity, ql.Period(int(frequency * 12), ql.Months))
    float_schedule = ql.MakeSchedule(start, maturity, ql.Period(int(frequency * 12), ql.Months))

    customIndex = ql.IborIndex('MyIndex', ql.Period('1Y'), 0, ql.USDCurrency(), ql.UnitedStates(0),
                               ql.ModifiedFollowing, False, day_count, ytsh)

    customIndex.addFixing(todays_date, r_0)

    if is_payer:
        swap = ql.VanillaSwap(ql.VanillaSwap.Payer, notional, fix_schedule, fixed_rate, day_count,
                              float_schedule, customIndex, 0, day_count)
    else:
        swap = ql.VanillaSwap(ql.VanillaSwap.Receiver, notional, fix_schedule, fixed_rate, day_count,
                              float_schedule, customIndex, 0, day_count)

    swap.setPricingEngine(ql.DiscountingSwapEngine(ytsh))

    return swap

def ql_create_swaption(ql_swap: ql.VanillaSwap, ql_hull_white: ql.HullWhite,
                       ytsh: ql.YieldTermStructureHandle) -> ql.Swaption:
    exercise = ql.EuropeanExercise(ql_swap.startDate())
    swaption = ql.Swaption(ql_swap, exercise)
    jamshidian_engine = ql.JamshidianSwaptionEngine(ql_hull_white, ytsh)
    swaption.setPricingEngine(jamshidian_engine)
    return swaption

def bond_price_mc(time_vec: np.array, r_sims: pd.DataFrame) -> np.array:
    return np.mean(np.exp(-scipy.integrate.trapezoid(r_sims, time_vec)))


def bond_option_price_mc(time_vec: float | np.ndarray, r_sims: float | np.ndarray, option_mat: float,
                         bond_mat: float, strike: float, yield_curve, alpha: float, sigma: float,
                         is_call: bool = True) -> float | np.ndarray:

    bond_price = bond_price_HW1F(yield_curve=yield_curve, t=option_mat, T=option_mat+bond_mat,
                                 alpha=alpha, sigma = sigma, r_t=r_sims[:, -1])

    if is_call:
        payoff = np.maximum(bond_price - strike, 0)
    else:
        payoff = np.maximum(strike - bond_price, 0)

    return np.mean(np.exp(-scipy.integrate.trapezoid(r_sims, time_vec)) * payoff)


def swaption_price_mc(time_vec: np.array, r_sims: pd.DataFrame, option_mat: float, swap_tenor: float, swap_freq: float,
                      strike_fixed_rate: float, yield_curve: float, alpha: float, sigma: float,
                      is_payer: bool = True) -> float:
    if is_payer:
        def payoff(r_T: float) -> float:
            return 1  - bond_price_HW1F(yield_curve, option_mat, option_mat + swap_tenor, alpha, sigma, r_t=r_T) \
                   - sum(bond_price_HW1F(yield_curve, option_mat,
                   option_mat + np.arange(swap_freq, swap_tenor + swap_freq, swap_freq),
                   alpha, sigma, r_t=r_T)) * strike_fixed_rate * swap_freq
    else:
        def payoff(r_T: float) -> float:
            return sum(bond_price_HW1F(yield_curve, option_mat, option_mat +
                   np.arange(swap_freq, swap_tenor + swap_freq, swap_freq), alpha, sigma, r_t=r_T)) * \
                   strike_fixed_rate * swap_freq + \
                   bond_price_HW1F(yield_curve, option_mat,option_mat + swap_tenor, alpha, sigma,r_t=r_T) - 1

    payoff = np.vectorize(payoff)
    payoffs = np.maximum(payoff(r_sims[:,-1]),0)

    return np.mean(np.exp(-scipy.integrate.trapezoid(r_sims, time_vec))*payoffs)

