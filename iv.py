"""Implied volatility for European options under Black-Scholes: Newton's method
on vega, with a bisection fallback that always converges inside the no-arbitrage bounds."""
import math


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S, K, r, sigma, T, kind="call", q=0.0):
    fwd_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    if sigma <= 0.0:
        intrinsic = fwd_s - disc_k if kind == "call" else disc_k - fwd_s
        return max(intrinsic, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(fwd_s / disc_k) + 0.5 * v * v) / v
    d2 = d1 - v
    if kind == "call":
        return fwd_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - fwd_s * norm_cdf(-d1)


def vega(S, K, r, sigma, T, q=0.0):
    fwd_s = S * math.exp(-q * T)
    v = sigma * math.sqrt(T)
    d1 = (math.log(fwd_s / (K * math.exp(-r * T))) + 0.5 * v * v) / v
    return fwd_s * norm_pdf(d1) * math.sqrt(T)


def bounds(S, K, r, T, kind="call", q=0.0):
    """No-arbitrage range for the option price."""
    fwd_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    if kind == "call":
        return max(fwd_s - disc_k, 0.0), fwd_s
    return max(disc_k - fwd_s, 0.0), disc_k


def implied_vol(price, S, K, r, T, kind="call", q=0.0, tol=1e-10, max_iter=100):
    """Return sigma with bs_price(..., sigma) == price, or raise ValueError
    if the price is outside the no-arbitrage bounds."""
    lo_p, hi_p = bounds(S, K, r, T, kind, q)
    if not (lo_p < price < hi_p):
        raise ValueError(f"price {price} is outside the no-arbitrage range ({lo_p}, {hi_p})")
    lo, hi = 1e-9, 1.0
    while bs_price(S, K, r, hi, T, kind, q) < price:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError("no volatility below 10,000% reproduces this price")
    sigma = 0.5 * (lo + hi)
    for _ in range(max_iter):
        diff = bs_price(S, K, r, sigma, T, kind, q) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        v = vega(S, K, r, sigma, T, q)
        step = sigma - diff / v if v > 1e-12 else None
        # take the Newton step only when it stays inside the bracket
        sigma = step if step is not None and lo < step < hi else 0.5 * (lo + hi)
    return sigma


if __name__ == "__main__":
    p = bs_price(100, 110, 0.04, 0.27, 0.5)
    print(f"call priced at 27% vol: {p:.6f}")
    print(f"implied vol recovered: {implied_vol(p, 100, 110, 0.04, 0.5):.10f}")
