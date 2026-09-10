"""Tests for BPP extension and input-domain guards."""
import numpy as np
import pytest

from src.analytical.kaufman_roberts import kaufman_roberts_bpp


class TestBpp:

    def test_bpp_binomial_requires_source_counts(self):
        with pytest.raises(ValueError, match="source_counts"):
            kaufman_roberts_bpp(
                10,
                np.array([5.0, 3.0]),
                np.array([1, 2]),
                traffic_types=['binomial', 'poisson'],
            )

    def test_bpp_pascal_requires_source_counts(self):
        with pytest.raises(ValueError, match="source_counts"):
            kaufman_roberts_bpp(
                10,
                np.array([5.0, 3.0]),
                np.array([1, 2]),
                traffic_types=['poisson', 'pascal'],
            )

    def test_bpp_pascal_smoke(self):
        P, B = kaufman_roberts_bpp(
            20,
            np.array([3.0, 2.0]),
            np.array([1, 2]),
            traffic_types=['pascal', 'poisson'],
            source_counts=np.array([10.0, np.inf]),
        )
        assert np.all(np.isfinite(P)), "P contains non-finite values"
        assert np.all(np.isfinite(B)), "B contains non-finite values"

    def test_bpp_poisson_only_no_source_counts(self):
        P, B = kaufman_roberts_bpp(
            15,
            np.array([4.0, 2.0]),
            np.array([1, 2]),
            traffic_types=['poisson', 'poisson'],
        )
        assert np.all(np.isfinite(B)), "Poisson-only BPP returned non-finite blocking"
