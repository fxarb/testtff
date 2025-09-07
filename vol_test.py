
"""Root finder functions using newton method."""

import tensorflow as tf

import numpy as np

def default_relative_root_tolerance(dtype):
  """Returns the default relative root tolerance used for a TensorFlow dtype."""
  if dtype is None:
    dtype = tf.float64
  return 4 * np.finfo(dtype.as_numpy_dtype(0)).eps

# TODO(b/179451420): Refactor BrentResults as RootSearchResults and return it
# for newton method as well.
def root_finder(value_and_grad_func,
                initial_values,
                max_iterations=20,
                tolerance=2e-7,
                relative_tolerance=None,
                dtype=None,
                name='root_finder'):
  """Finds roots of a scalar function using Newton's method.

  This method uses Newton's algorithm to find values `x` such that `f(x)=0` for
  some real-valued differentiable function `f`. Given an initial value `x_0` the
  values are iteratively updated as:

    `x_{n+1} = x_n - f(x_n) / f'(x_n),`

  for further details on Newton's method, see [1]. The implementation accepts
  array-like arguments and assumes that each cell corresponds to an independent
  scalar model.

  #### Examples
  ```python
  # Set up the problem of finding the square roots of three numbers.
  constants = np.array([4.0, 9.0, 16.0])
  initial_values = np.ones(len(constants))
  def objective_and_gradient(values):
    objective = values**2 - constants
    gradient = 2.0 * values
    return objective, gradient

  # Obtain and evaluate a tensor containing the roots.
  roots = tff.math.root_search.newton_root(objective_and_gradient,
    initial_values)
  print(root_values)  # Expected output: [ 2.  3.  4.]
  print(converged)  # Expected output: [ True  True  True]
  print(failed)  # Expected output: [False False False]
  ```

  #### References
  [1] Luenberger, D.G., 1984. 'Linear and Nonlinear Programming'. Reading, MA:
  Addison-Wesley.

  Args:
    value_and_grad_func: A python callable that takes a `Tensor` of the same
      shape and dtype as the `initial_values` and which returns a two-`tuple` of
      `Tensors`, namely the objective function and the gradient evaluated at the
      passed parameters.
    initial_values: A real `Tensor` of any shape. The initial values of the
      parameters to use (`x_0` in the notation above).
    max_iterations: positive `int`. The maximum number of
      iterations of Newton's method.
      Default value: 20.
    tolerance: positive scalar `Tensor`. The absolute tolerance for the root
      search. Search is judged to have converged  if
      `|f(x_n) - f(x_n-1)|` < |x_n| * `relative_tolerance` + `tolerance`
      (using the notation above), or if `x_n` becomes `nan`. When an element is
      judged to have converged it will no longer be updated. If all elements
      converge before `max_iterations` is reached then the root finder will
      return early. If None, it would be set according to the `dtype`,
      which is 4 * np.finfo(dtype.as_numpy_dtype(0)).eps.
      Default value: 2e-7.
    relative_tolerance: positive `double`, default 0. See the document for
      `tolerance`.
      Default value: None.
    dtype: optional `tf.DType`. If supplied the `initial_values` will be coerced
      to this data type.
      Default value: None.
    name: `str`, to be prefixed to the name of
      TensorFlow ops created by this function.
      Default value: 'root_finder'.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `initial_values`. It
    contains the found roots (same dtype as `initial_values`), a boolean
    `Tensor` indicating whether the corresponding root results in an objective
    function value less than the tolerance, and a boolean `Tensor` which is true
    where the corresponding 'root' is not finite.
  """
  if tolerance is None:
    tolerance = default_relative_root_tolerance(dtype)
  if relative_tolerance is None:
    relative_tolerance = default_relative_root_tolerance(dtype)

  with tf.compat.v1.name_scope(
      name,
      default_name='newton_root_finder',
      values=[initial_values, tolerance]):

    initial_values = tf.convert_to_tensor(
        initial_values, dtype=dtype, name='initial_values')

    def _condition(counter, parameters, converged, failed):
      del parameters
      early_stop = tf.reduce_all(converged | failed)
      return ~((counter >= max_iterations) | early_stop)

    def _updater(counter, parameters, converged, failed):
      """Updates each parameter via Newton's method."""
      values, gradients = value_and_grad_func(parameters)
      deltas = tf.math.divide(values, gradients)

      converged = tf.abs(
          deltas) < relative_tolerance * tf.abs(values) + tolerance

      # Used to zero out updates to cells that have converged.
      update_mask = tf.cast(~converged, dtype=parameters.dtype)
      increment = -update_mask * deltas
      updated_parameters = parameters + increment
      failed = ~tf.math.is_finite(updated_parameters)

      return counter + 1, updated_parameters, converged, failed

    starting_position = (tf.constant(0, dtype=tf.int32), initial_values,
                         tf.zeros_like(initial_values, dtype=tf.bool),
                         tf.math.is_nan(initial_values))

    return tf.while_loop(_condition, _updater, starting_position,
                         maximum_iterations=max_iterations)[1:]



"""Calculation of the Black-Scholes implied volatility via Newton's method."""


_SQRT_2 = np.sqrt(2., dtype=np.float64)
_SQRT_2_PI = np.sqrt(2 * np.pi, dtype=np.float64)
_NORM_PDF_AT_ZERO = 1. / _SQRT_2_PI


def _cdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


def _pdf(x):
  return tf.math.exp(-0.5 * x ** 2) / _SQRT_2_PI


def implied_vol(*,
                prices,
                strikes,
                expiries,
                spots=None,
                forwards=None,
                discount_factors=None,
                is_call_options=None,
                initial_volatilities=tf.constant(0.2, dtype=tf.float64),
                tolerance=1e-8,
                max_iterations=20,
                validate_args=False,
                dtype=None,
                name=None):
  """Computes implied volatilities from given call or put option prices.

  This method applies a Newton root search algorithm to back out the implied
  volatility given the price of either a put or a call option.

  The implementation assumes that each cell in the supplied tensors corresponds
  to an independent volatility to find.

  Args:
    prices: A real `Tensor` of any shape. The prices of the options whose
      implied vol is to be calculated.
    strikes: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The strikes of the options.
    expiries: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The expiry for each option. The units should be
      such that `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
      `prices`. The current spot price of the underlying. Either this argument
      or the `forwards` (but not both) must be supplied.
      Default value: None.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `prices`. The forwards to maturity. Either this argument or the `spots`
      must be supplied but both must not be supplied.
      Default value: None.
    discount_factors: An optional real `Tensor` of same dtype as the `prices`.
      If not None, these are the discount factors to expiry (i.e. e^(-rT)). If
      None, no discounting is applied (i.e. it is assumed that the undiscounted
      option prices are provided ). If `spots` is supplied and
      `discount_factors` is not None then this is also used to compute the
      forwards to expiry.
      Default value: None, equivalent to discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.
      Indicates whether the option is a call (if True) or a put (if False). If
      not supplied, call options are assumed.
      Default value: None.
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting positions for Newton's method.
      Default value: None. If not supplied, the starting point is chosen using
        the Stefanica-Radoicic scheme. See `polya_approx.implied_vol` for
        details.
      Default value: None.
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    max_iterations: `int`. The maximum number of iterations of Newton's method.
      Default value: 20.
    validate_args: A Python bool. If True, indicates that arguments should be
      checked for correctness before performing the computation. The checks
      performed are: (1) Forwards and strikes are positive. (2) The prices
        satisfy the arbitrage bounds (i.e. for call options, checks the
        inequality `max(F-K, 0) <= Price <= F` and for put options, checks that
        `max(K-F, 0) <= Price <= K`.). (3) Checks that the prices are not too
        close to the bounds. It is numerically unstable to compute the implied
        vols from options too far in the money or out of the money.
      Default value: False.
    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not
      supplied, the default TensorFlow conversion will take place. Note that
      this argument does not do any casting for `Tensor`s or numpy arrays.
      Default value: None.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'implied_vol' is used.
      Default value: None.

  Returns:
    A 3-tuple containing the following items in order:
       (a) implied_vols: A `Tensor` of the same dtype as `prices` and shape as
         the common broadcasted shape of
         `(prices, spots/forwards, strikes, expiries)`. The implied vols as
         inferred by the algorithm. It is possible that the search may not have
         converged or may have produced NaNs. This can be checked for using the
         following return values.
       (b) converged: A boolean `Tensor` of the same shape as `implied_vols`
         above. Indicates whether the corresponding vol has converged to within
         tolerance.
       (c) failed: A boolean `Tensor` of the same shape as `implied_vols` above.
         Indicates whether the corresponding vol is NaN or not a finite number.
         Note that converged being True implies that failed will be false.
         However, it may happen that converged is False but failed is not True.
         This indicates the search did not converge in the permitted number of
         iterations but may converge if the iterations are increased.

  Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
      supplied.
  """
  if (spots is None) == (forwards is None):
    raise ValueError('Either spots or forwards must be supplied but not both.')

  with tf.compat.v1.name_scope(
      name,
      default_name='implied_vol',
      values=[
          prices, spots, forwards, strikes, expiries, discount_factors,
          is_call_options, initial_volatilities
      ]):
    prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
    dtype = prices.dtype
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    if discount_factors is None:
      discount_factors = tf.convert_to_tensor(
          1.0, dtype=dtype, name='discount_factors')
    else:
      discount_factors = tf.convert_to_tensor(
          discount_factors, dtype=dtype, name='discount_factors')

    if forwards is not None:
      forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    else:
      spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
      forwards = spots / discount_factors


    initial_volatilities = tf.convert_to_tensor(
          initial_volatilities, dtype=dtype, name='initial_volatilities')

    implied_vols, converged, failed = _newton_implied_vol(
        prices, strikes, expiries, forwards, discount_factors, is_call_options,
        initial_volatilities,
        tolerance, max_iterations)
    return implied_vols, converged, failed


def _newton_implied_vol(prices, strikes, expiries, forwards, discount_factors,
                        is_call_options, initial_volatilities, tolerance, max_iterations):
  """Uses Newton's method to find Black Scholes implied volatilities of options.

  Finds the volatility implied under the Black Scholes option pricing scheme for
  a set of European options given observed market prices. The implied volatility
  is found via application of Newton's algorithm for locating the root of a
  differentiable function.

  The implementation assumes that each cell in the supplied tensors corresponds
  to an independent volatility to find.

  Args:
    prices: A real `Tensor` of any shape. The prices of the options whose
      implied vol is to be calculated.
    strikes: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The strikes of the options.
    expiries: A real `Tensor` of the same dtype as `prices` and a shape that
      broadcasts with `prices`. The expiry for each option. The units should be
      such that `expiry * volatility**2` is dimensionless.
    forwards: A real `Tensor` of any shape that broadcasts to the shape of
      `prices`. The forwards to maturity.
    discount_factors: An optional real `Tensor` of same dtype as the `prices`.
      If not None, these are the discount factors to expiry (i.e. e^(-rT)). If
      None, no discounting is applied (i.e. it is assumed that the undiscounted
      option prices are provided ).
    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.
      Indicates whether the option is a call (if True) or a put (if False). If
      not supplied, call options are assumed.
    initial_volatilities: A real `Tensor` of the same shape and dtype as
      `forwards`. The starting positions for Newton's method.
    tolerance: `float`. The root finder will stop where this tolerance is
      crossed.
    max_iterations: `int`. The maximum number of iterations of Newton's method.

  Returns:
    A three tuple of `Tensor`s, each the same shape as `forwards`. It
    contains the implied volatilities (same dtype as `forwards`), a boolean
    `Tensor` indicating whether the corresponding implied volatility converged,
    and a boolean `Tensor` which is true where the corresponding implied
    volatility is not a finite real number.
  """
  pricer = _make_black_lognormal_objective_and_vega_func(
        prices, forwards, strikes, expiries, is_call_options,
        discount_factors)

  results = root_finder(
      pricer,
      initial_volatilities,
      max_iterations=max_iterations,
      tolerance=tolerance)
  return results


def _get_normalizations(prices, forwards, strikes, discount_factors):
  """Returns the normalized prices, normalization factors, and discount_factors.

  The normalization factors is the larger of strikes and forwards.
  If `discount_factors` is not None, these are the discount factors to expiry.
  If None, no discounting is applied and 1's are returned.

  Args:
    prices: A real `Tensor` of any shape. The observed market prices of the
      assets.
    forwards: A real `Tensor` of the same shape and dtype as `prices`. The
      current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike
      prices of the options.
    discount_factors: A real `Tensor` of same dtype as the `prices`.

  Returns:
    the normalized prices, normalization factors, and discount_factors.
  """
  strikes_abs = tf.abs(strikes)
  forwards_abs = tf.abs(forwards)
  # orientations will decide the normalization strategy.
  orientations = strikes_abs >= forwards_abs
  # normalization is the greater of strikes or forwards
  normalization = tf.where(orientations, strikes_abs, forwards_abs)
  normalization = tf.where(tf.equal(normalization, 0),
                           tf.ones_like(normalization), normalization)
  normalized_prices = prices / normalization
  if discount_factors is not None:
    normalized_prices /= discount_factors
  else:
    discount_factors = tf.ones_like(normalized_prices)

  return normalized_prices, normalization, discount_factors


def _make_black_lognormal_objective_and_vega_func(
    prices, forwards, strikes, expiries, is_call_options, discount_factors):
  """Produces an objective and vega function for the Black Scholes model.

  The returned function maps volatilities to a tuple of objective function
  values and their gradients with respect to the volatilities. The objective
  function is the difference between Black Scholes prices and observed market
  prices, whereas the gradient is called vega of the option. That is:

  ```
  g(s) = (f(s) - a, f'(s))
  ```

  Where `g` is the returned function taking volatility parameter `s`, `f` the
  Black Scholes price with all other variables curried and `f'` its derivative,
  and `a` the observed market prices of the options. Hence `g` calculates the
  information necessary for finding the volatility implied by observed market
  prices for options with given terms using first order methods.

  #### References
  [1] Hull, J., 2018. Options, Futures, and Other Derivatives. Harlow, England.
  Pearson. (p.358 - 361)

  Args:
    prices: A real `Tensor` of any shape. The observed market prices of the
      assets.
    forwards: A real `Tensor` of the same shape and dtype as `prices`. The
      current forward prices to expiry.
    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike
      prices of the options.
    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    is_call_options: A boolean `Tensor` of same shape and dtype as `forwards`.
      Positive one where option is a call, negative one where option is a put.
    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.
      The total discount factors to apply.

  Returns:
    A function from volatilities to a Black Scholes objective and its
    derivative (which is coincident with Vega).
  """
  normalized_prices, normalization, discount_factors = _get_normalizations(
      prices, forwards, strikes, discount_factors)

  norm_forwards = forwards / normalization
  norm_strikes = strikes / normalization
  lnz = tf.math.log(forwards) - tf.math.log(strikes)
  sqrt_t = tf.sqrt(expiries)
  if is_call_options is not None:
    is_call_options = tf.convert_to_tensor(is_call_options,
                                           dtype=tf.bool,
                                           name='is_call_options')
  def _black_objective_and_vega(volatilities):
    """Calculate the Black Scholes price and vega for a given volatility.

    This method returns normalized results.

    Args:
      volatilities: A real `Tensor` of same shape and dtype as `forwards`. The
        volatility to expiry.

    Returns:
      A tuple containing (value, gradient) of the black scholes price, both of
        which are `Tensor`s of the same shape and dtype as `volatilities`.
    """
    vol_t = volatilities * sqrt_t
    d1 = (lnz / vol_t + vol_t / 2)
    d2 = d1 - vol_t
    implied_prices = norm_forwards * _cdf(d1) - norm_strikes * _cdf(d2)
    if is_call_options is not None:
      put_prices = implied_prices - norm_forwards + norm_strikes
      implied_prices = tf.where(
          tf.broadcast_to(is_call_options, tf.shape(put_prices)),
          implied_prices, put_prices)
    vega = norm_forwards * _pdf(d1) * sqrt_t / discount_factors
    return implied_prices - normalized_prices, vega

  return _black_objective_and_vega


import math
import numpy as np
from scipy.stats import mvn, norm

class _GBS_Limits:
    # An GBS model will return an error if an out-of-bound input is input
    MAX32 = 2147483248.0

    MIN_T = 1.0 / 1000.0  # requires some time left before expiration
    MIN_X = 0.01
    MIN_FS = 0.01

    # Volatility smaller than 0.5% causes American Options calculations
    # to fail (Number to large errors).
    # GBS() should be OK with any positive number. Since vols less
    # than 0.5% are expected to be extremely rare, and most likely bad inputs,
    # _gbs() is assigned this limit too
    MIN_V = 0.005

    MAX_T = 100
    MAX_X = MAX32
    MAX_FS = MAX32

    # Asian Option limits
    # maximum TA is time to expiration for the option
    MIN_TA = 0

    # This model will work with higher values for b, r, and V. However, such values are extremely uncommon.
    # To catch some common errors, interest rates and volatility is capped to 200%
    # This reason for 2 (200%) is mostly to cause the library to throw an exceptions
    # if a value like 15% is entered as 15 rather than 0.15)
    MIN_b = -1
    MIN_r = -1

    MAX_b = 1
    MAX_r = 2
    MAX_V = 2

# The primary class for calculating Generalized Black Scholes option prices and deltas
# It is not intended to be part of this module's public interface

# Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
#         b = cost of carry, v = implied volatility
# Outputs: value, delta, gamma, theta, vega, rho
def _gbs(option_type, fs, x, t, r, b, v):
    # -----------

    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        # it's a call
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)


    return value, delta, gamma, theta, vega, rho



def _newton_implied_vol(val_fn, option_type, x, fs, t, b, r, cp, precision=.00001, max_steps=100):

    # Estimate starting Vol, making sure it is allowable range
    v = _approx_implied_vol(option_type, fs, x, t, r, b, cp)
    v = max(_GBS_Limits.MIN_V, v)
    v = min(_GBS_Limits.MAX_V, v)

    # Calculate the value at the estimated vol
    value, delta, gamma, theta, vega, rho = val_fn(option_type, fs, x, t, r, b, v)
    min_diff = abs(cp - value)


    # Newton-Raphson Search
    countr = 0
    while precision <= abs(cp - value) <= min_diff and countr < max_steps:

        v = v - (value - cp) / vega
        if (v > _GBS_Limits.MAX_V) or (v < _GBS_Limits.MIN_V):
            break

        value, delta, gamma, theta, vega, rho = val_fn(option_type, fs, x, t, r, b, v)
        min_diff = min(abs(cp - value), min_diff)

        # keep track of how many loops
        countr += 1


    # check if function converged and return a value
    if abs(cp - value) < precision:
        # the search function converged
        return v
    else:
        # if the search function didn't converge, try a bisection search
        return np.nan




# Find the Implied Volatility of an European (GBS) Option given a price
# using Newton-Raphson method for greater speed since Vega is available
def _gbs_implied_vol(option_type, fs, x, t, r, b, cp, precision=.00001, max_steps=100):
    return _newton_implied_vol(_gbs, option_type, x, fs, t, b, r, cp, precision, max_steps)


def euro_implied_vol(option_type, fs, x, t, r, q, cp):
    """Implied volatility calculator for European options.

    Args:
        option_type (str): Type of the option. "p" for put and "c" for call options.
        fs (float): Price of underlying asset.
        x (float): Strike price.
        t (float): Time to expiration in years. 1 for one year, 0.5 for 6 months.
        r (float): Risk free rate.
        q (float): Dividend payment. Set q=0 for non-dividend paying options.
        cp (float): The price of the call or put observed in the market.

    Returns:
        value (float): Implied volatility.
    """
    b = r - q
    return _gbs_implied_vol(option_type, fs, x, t, r, b, cp)
