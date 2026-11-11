import itertools
import math
import unittest

from iv import bounds, bs_price, implied_vol


class TestImpliedVol(unittest.TestCase):
    def test_round_trip_over_a_grid(self):
        count = 0
        for S, K, T, sigma, kind in itertools.product(
            [100.0], [50.0, 80.0, 100.0, 120.0, 200.0], [0.02, 0.25, 1.0, 5.0],
            [0.05, 0.2, 0.6, 1.5], ["call", "put"]
        ):
            price = bs_price(S, K, 0.03, sigma, T, kind, q=0.01)
            lo, hi = bounds(S, K, 0.03, T, kind, q=0.01)
            # skip prices that sit on a bound to machine precision: vol is not identifiable there
            if price - lo < 1e-9 * S or hi - price < 1e-9 * S:
                continue
            got = implied_vol(price, S, K, 0.03, T, kind, q=0.01)
            self.assertAlmostEqual(bs_price(S, K, 0.03, got, T, kind, q=0.01), price, delta=1e-8)
            self.assertAlmostEqual(got, sigma, delta=1e-5 * max(1.0, sigma))
            count += 1
        self.assertGreater(count, 120)

    def test_call_and_put_give_the_same_vol(self):
        S, K, r, T, sigma = 100.0, 95.0, 0.02, 0.75, 0.33
        c = bs_price(S, K, r, sigma, T, "call")
        p = bs_price(S, K, r, sigma, T, "put")
        self.assertAlmostEqual(c - p, S - K * math.exp(-r * T), places=10)
        self.assertAlmostEqual(implied_vol(c, S, K, r, T, "call"), implied_vol(p, S, K, r, T, "put"), places=8)

    def test_rejects_arbitrage_prices(self):
        with self.assertRaises(ValueError):
            implied_vol(0.5, 100, 50, 0.0, 1.0)      # below intrinsic value 50
        with self.assertRaises(ValueError):
            implied_vol(100.5, 100, 50, 0.0, 1.0)    # above the spot


if __name__ == "__main__":
    unittest.main()
