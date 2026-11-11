# Implied volatility solver for Black-Scholes, Newton with a safe fallback

Given an option's market price, the implied volatility is the sigma that makes Black-Scholes give back that price. There's no closed form for it, so you solve for it numerically, and it's a common quant interview exercise because the easy version breaks in ways you only notice later. Newton's method on vega is fast near the money. Far out of the money, or very close to expiry, vega is tiny, and a plain Newton step can shoot off to a negative or huge sigma.

`iv.py` keeps a bracket around the answer the whole time. It takes the Newton step when that step stays inside the bracket and halves the bracket when it doesn't, so it's as quick as Newton when Newton behaves and it still converges when Newton doesn't. Before it starts, it checks the price against the no-arbitrage range (for a call, between the discounted intrinsic value and the spot after dividends). A price outside that range has no implied volatility at all, so you get a `ValueError` and not a made-up number.

```python
from iv import bs_price, implied_vol

p = bs_price(100, 110, 0.04, 0.27, 0.5)   # 4.596269, a six-month call at 27% vol
implied_vol(p, 100, 110, 0.04, 0.5)       # 0.27
implied_vol(5.2, 100, 95, 0.02, 0.25, kind="put", q=0.01)
```

Arguments are spot, strike, rate, time in years, then `kind` ("call" or "put") and a continuous dividend yield `q`.

## Tests

`python3 -m unittest test_iv` prices options over a grid of strikes from 50 to 200, expiries from a week to five years and vols from 5% to 150%, for calls and puts. It solves each one back and checks it gets the vol it started from. It skips the few cases where the price sits on a no-arbitrage bound to machine precision, because the vol really can't be recovered there, and it checks that more than 120 cases were tested. It also checks that a call and a put at the same strike give the same implied vol, as put-call parity says they must, and that prices outside the bounds are rejected. You only need the Python standard library.

## More practice

Quant interview questions on options, probability and stochastic calculus, with worked answers: https://geodesicpartners.co.uk/quant-questions-stochastic-calculus

MIT licence.
