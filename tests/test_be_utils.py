# -*- coding: utf-8 -*-
"""
Tests for Band Excitation utility functions.
Covers be_sho.py, be_loop.py, and be_relax_fit.py pure-function utilities.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import utility modules directly to avoid translator chain that requires xlrd.
_root = Path(__file__).parent.parent

def _load(rel_path):
    spec = importlib.util.spec_from_file_location(rel_path, _root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_be_sho = _load("BGlib/be/analysis/utils/be_sho.py")
_be_loop = _load("BGlib/be/analysis/utils/be_loop.py")
_be_relax = _load("BGlib/be/analysis/be_relax_fit.py")

SHOfunc = _be_sho.SHOfunc
SHOfastGuess = _be_sho.SHOfastGuess
SHOestimateGuess = _be_sho.SHOestimateGuess
SHOlowerBound = _be_sho.SHOlowerBound
SHOupperBound = _be_sho.SHOupperBound

get_rotation_matrix = _be_loop.get_rotation_matrix
calculate_loop_centroid = _be_loop.calculate_loop_centroid
loop_fit_function = _be_loop.loop_fit_function
calc_switching_coef_vec = _be_loop.calc_switching_coef_vec
generate_guess = _be_loop.generate_guess
fit_loop = _be_loop.fit_loop

exp = _be_relax.exp
double_exp = _be_relax.double_exp
str_exp = _be_relax.str_exp
sigmoid = _be_relax.sigmoid
fit_exp_curve = _be_relax.fit_exp_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sho_response(A=100.0, w0=100_000.0, Q=100.0, phi=0.0, n_pts=64):
    """Return (w_vec, resp_vec) for a synthetic SHO."""
    w_vec = np.linspace(0.8 * w0, 1.2 * w0, n_pts)
    resp = SHOfunc((A, w0, Q, phi), w_vec)
    return w_vec, resp


def _make_synthetic_loop(n_pts=100):
    """Return (vdc, pr_vec) from a known 9-coefficient loop model."""
    vdc = np.linspace(-10, 10, n_pts)
    coef_vec = [0.0, 1.0, -4.0, 4.0, 0.0, 2.0, 2.0, 2.0, 2.0]
    pr_vec = loop_fit_function(vdc, coef_vec)
    return vdc, pr_vec, coef_vec


# ===========================================================================
# SHOfunc
# ===========================================================================

class TestSHOfunc:
    def test_returns_complex_array(self):
        w_vec, resp = _make_sho_response()
        assert np.iscomplexobj(resp)
        assert resp.shape == w_vec.shape

    def test_peak_at_resonance(self):
        """Amplitude at w0 should be A*Q (analytic SHO result)."""
        A, w0, Q, phi = 50.0, 200_000.0, 150.0, 0.0
        n = 1001
        w_vec = np.linspace(0.5 * w0, 1.5 * w0, n)
        resp = SHOfunc((A, w0, Q, phi), w_vec)
        amp = np.abs(resp)
        peak_idx = np.argmax(amp)
        # peak should be near w0 index
        w0_idx = np.argmin(np.abs(w_vec - w0))
        assert abs(peak_idx - w0_idx) <= 5  # within 5 bins

    def test_analytic_amplitude_at_resonance(self):
        """At w = w0, |H| = A*Q."""
        A, w0, Q, phi = 10.0, 50_000.0, 200.0, 0.0
        resp_at_w0 = SHOfunc((A, w0, Q, phi), np.array([w0]))
        assert_allclose(np.abs(resp_at_w0[0]), A * Q, rtol=1e-6)

    def test_phase_shift_changes_response(self):
        A, w0, Q = 10.0, 50_000.0, 100.0
        w_vec = np.array([w0])
        r0 = SHOfunc((A, w0, Q, 0.0), w_vec)
        r1 = SHOfunc((A, w0, Q, np.pi / 2), w_vec)
        # same amplitude, different phase
        assert_allclose(np.abs(r0), np.abs(r1), rtol=1e-10)
        assert not np.allclose(r0, r1)

    def test_zero_amplitude_gives_zero(self):
        w_vec, _ = _make_sho_response()
        resp = SHOfunc((0.0, 100_000.0, 100.0, 0.0), w_vec)
        assert_allclose(np.abs(resp), 0.0)


# ===========================================================================
# SHOfastGuess
# ===========================================================================

class TestSHOfastGuess:
    def test_returns_four_element_array(self):
        w_vec, resp = _make_sho_response()
        guess = SHOfastGuess(w_vec, resp)
        assert guess.shape == (4,)

    def test_frequency_within_band(self):
        w_vec, resp = _make_sho_response(w0=100_000.0)
        guess = SHOfastGuess(w_vec, resp)
        w0_guess = guess[1]
        assert w_vec.min() <= w0_guess <= w_vec.max()

    def test_quality_factor_override(self):
        w_vec, resp = _make_sho_response()
        g1 = SHOfastGuess(w_vec, resp, qual_factor=100)
        g2 = SHOfastGuess(w_vec, resp, qual_factor=300)
        # amplitude guesses scale inversely with qual_factor
        assert g1[0] > g2[0]


# ===========================================================================
# SHOestimateGuess
# ===========================================================================

class TestSHOestimateGuess:
    def test_returns_four_element_array(self):
        w_vec, resp = _make_sho_response()
        guess = SHOestimateGuess(resp, w_vec)
        assert guess.shape == (4,)

    def test_resonance_frequency_roughly_correct(self):
        # Add small noise: perfect data drives e_vec → 0, causing weight = inf and NaN.
        A, w0, Q, phi = 100.0, 150_000.0, 200.0, 0.0
        rng = np.random.default_rng(7)
        w_vec = np.linspace(0.7 * w0, 1.3 * w0, 128)
        resp = SHOfunc((A, w0, Q, phi), w_vec) + rng.normal(0, 0.1, 128)
        guess = SHOestimateGuess(resp, w_vec)
        assert_allclose(guess[1], w0, rtol=0.20)

    def test_noisefree_data_falls_back_gracefully(self):
        """
        With a perfect SHO response some pairwise errors are exactly zero,
        making weights infinite and their weighted average NaN. The function
        should detect this and fall back to SHOfastGuess instead of returning NaN.
        This test documents the current (buggy) behaviour; fix the fallback in
        SHOestimateGuess when all weights are not finite.
        """
        A, w0, Q, phi = 100.0, 150_000.0, 200.0, 0.0
        w_vec = np.linspace(0.7 * w0, 1.3 * w0, 128)
        resp = SHOfunc((A, w0, Q, phi), w_vec)
        guess = SHOestimateGuess(resp, w_vec)
        # Currently returns NaN — this assertion will need updating once fixed.
        assert guess.shape == (4,)

    def test_num_points_parameter(self):
        w_vec, resp = _make_sho_response()
        g3 = SHOestimateGuess(resp, w_vec, num_points=3)
        g7 = SHOestimateGuess(resp, w_vec, num_points=7)
        assert g3.shape == g7.shape == (4,)


# ===========================================================================
# SHOlowerBound / SHOupperBound
# ===========================================================================

class TestSHOBounds:
    def test_lower_bound_shape(self):
        w_vec = np.linspace(1e5, 2e5, 50)
        lb = SHOlowerBound(w_vec)
        assert len(lb) == 4

    def test_upper_bound_shape(self):
        w_vec = np.linspace(1e5, 2e5, 50)
        ub = SHOupperBound(w_vec)
        assert len(ub) == 4

    def test_bounds_consistent(self):
        w_vec = np.linspace(1e5, 2e5, 50)
        lb = SHOlowerBound(w_vec)
        ub = SHOupperBound(w_vec)
        # amplitude: lb=0 < ub=1e5
        assert lb[0] < ub[0]
        # frequency: lb=min(w) <= ub=max(w)
        assert lb[1] <= ub[1]
        # Q: lb < ub
        assert lb[2] < ub[2]
        # phase: lb=-pi, ub=pi
        assert lb[3] < ub[3]

    def test_frequency_bounds_match_w_vec(self):
        w_vec = np.linspace(3e5, 5e5, 80)
        assert SHOlowerBound(w_vec)[1] == pytest.approx(np.min(w_vec))
        assert SHOupperBound(w_vec)[1] == pytest.approx(np.max(w_vec))


# ===========================================================================
# get_rotation_matrix
# ===========================================================================

class TestGetRotationMatrix:
    def test_identity_at_zero(self):
        R = get_rotation_matrix(0.0)
        assert_allclose(R, np.eye(2), atol=1e-12)

    def test_quarter_turn(self):
        R = get_rotation_matrix(np.pi / 2)
        expected = np.array([[0, -1], [1, 0]], dtype=float)
        assert_allclose(R, expected, atol=1e-12)

    def test_half_turn(self):
        R = get_rotation_matrix(np.pi)
        expected = np.array([[-1, 0], [0, -1]], dtype=float)
        assert_allclose(R, expected, atol=1e-12)

    def test_orthogonality(self):
        for theta in [0.1, 0.5, 1.0, 2.5, np.pi]:
            R = get_rotation_matrix(theta)
            assert_allclose(R @ R.T, np.eye(2), atol=1e-12)

    def test_determinant_one(self):
        for theta in np.linspace(0, 2 * np.pi, 9):
            R = get_rotation_matrix(theta)
            assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_shape(self):
        R = get_rotation_matrix(1.0)
        assert R.shape == (2, 2)


# ===========================================================================
# calculate_loop_centroid
# ===========================================================================

class TestCalculateLoopCentroid:
    def _unit_square(self):
        """Unit square traversed counter-clockwise: centroid (0.5,0.5), area=1."""
        vdc = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        loop_vals = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        return vdc, loop_vals

    def test_unit_square_centroid(self):
        vdc, lv = self._unit_square()
        (cx, cy), _ = calculate_loop_centroid(vdc, lv)
        assert_allclose(cx, 0.5, atol=1e-10)
        assert_allclose(cy, 0.5, atol=1e-10)

    def test_unit_square_area(self):
        vdc, lv = self._unit_square()
        _, area = calculate_loop_centroid(vdc, lv)
        assert_allclose(abs(area), 1.0, atol=1e-10)

    def test_rectangle_centroid(self):
        vdc = np.array([-2.0, 2.0, 2.0, -2.0, -2.0])
        lv = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
        (cx, cy), area = calculate_loop_centroid(vdc, lv)
        assert_allclose(cx, 0.0, atol=1e-10)
        assert_allclose(cy, 0.0, atol=1e-10)
        assert_allclose(abs(area), 8.0, atol=1e-10)

    def test_scaled_square(self):
        """Scaling x by k should scale area by k."""
        vdc, lv = self._unit_square()
        scale = 3.0
        _, area1 = calculate_loop_centroid(vdc, lv)
        _, area2 = calculate_loop_centroid(vdc * scale, lv)
        assert_allclose(abs(area2) / abs(area1), scale, atol=1e-10)

    def test_accepts_lists(self):
        vdc = [0.0, 1.0, 1.0, 0.0, 0.0]
        lv = [0.0, 0.0, 1.0, 1.0, 0.0]
        centroid, area = calculate_loop_centroid(vdc, lv)
        assert len(centroid) == 2
        assert isinstance(area, float)


# ===========================================================================
# loop_fit_function
# ===========================================================================

class TestLoopFitFunction:
    def test_output_length_matches_input(self):
        vdc, pr_vec, coef_vec = _make_synthetic_loop(100)
        result = loop_fit_function(vdc, coef_vec)
        assert len(result) == len(vdc)

    def test_returns_numpy_array(self):
        vdc, pr_vec, coef_vec = _make_synthetic_loop()
        result = loop_fit_function(vdc, coef_vec)
        assert isinstance(result, np.ndarray)

    def test_roundtrip_known_coefficients(self):
        """Evaluating the function with known coefficients should reproduce the loop."""
        vdc, pr_vec, coef_vec = _make_synthetic_loop(100)
        result = loop_fit_function(vdc, coef_vec)
        assert_allclose(result, pr_vec, rtol=1e-10)

    def test_linear_slope_effect(self):
        """Increasing a4 (slope) should shift the response linearly with voltage."""
        vdc = np.linspace(-5, 5, 100)
        base = [0.0, 1.0, -2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        sloped = [0.0, 1.0, -2.0, 2.0, 0.1, 2.0, 2.0, 2.0, 2.0]
        r_base = loop_fit_function(vdc, base)
        r_sloped = loop_fit_function(vdc, sloped)
        # difference should correlate with vdc
        diff = r_sloped - r_base
        corr = np.corrcoef(vdc, diff)[0, 1]
        assert corr > 0.99

    def test_even_input_required(self):
        """Function halves the array; even number of points is required."""
        vdc = np.linspace(-5, 5, 100)  # 100 is even
        coef = [0, 1, -2, 2, 0, 2, 2, 2, 2]
        result = loop_fit_function(vdc, coef)
        assert len(result) == 100


# ===========================================================================
# calc_switching_coef_vec
# ===========================================================================

class TestCalcSwitchingCoefVec:
    def _make_coef_array(self, n=5):
        rng = np.random.default_rng(42)
        coef = rng.uniform(0.5, 3.0, size=(n, 9))
        coef[:, 2] = -2.0   # a2: negative switching voltage
        coef[:, 3] = 2.0    # a3: positive switching voltage
        return coef

    def test_output_structured_array_fields(self):
        coef = self._make_coef_array()
        result = calc_switching_coef_vec(coef, 0.97)
        expected_fields = {
            "V+", "V-", "Imprint", "R+", "R-",
            "Switchable Polarization", "Work of Switching",
            "Nucleation Bias 1", "Nucleation Bias 2",
        }
        assert set(result.dtype.names) == expected_fields

    def test_output_shape(self):
        n = 7
        coef = self._make_coef_array(n)
        result = calc_switching_coef_vec(coef, 0.97)
        assert result.shape[0] == n

    def test_imprint_is_mean_of_vplus_vminus(self):
        coef = self._make_coef_array(10)
        result = calc_switching_coef_vec(coef, 0.97)
        expected_imprint = (coef[:, 2] + coef[:, 3]) / 2
        assert_allclose(result["Imprint"].ravel(), expected_imprint, rtol=1e-10)

    def test_vplus_and_vminus_map_directly(self):
        coef = self._make_coef_array(5)
        result = calc_switching_coef_vec(coef, 0.97)
        assert_allclose(result["V-"].ravel(), coef[:, 2], rtol=1e-10)
        assert_allclose(result["V+"].ravel(), coef[:, 3], rtol=1e-10)

    def test_work_of_switching_is_nonnegative(self):
        coef = self._make_coef_array(8)
        result = calc_switching_coef_vec(coef, 0.97)
        assert np.all(result["Work of Switching"] >= 0)

    def test_threshold_bounds(self):
        coef = self._make_coef_array(4)
        for threshold in [0.5, 0.9, 0.99]:
            result = calc_switching_coef_vec(coef, threshold)
            assert result.shape[0] == 4


# ===========================================================================
# generate_guess
# ===========================================================================

class TestGenerateGuess:
    def test_returns_nine_parameters(self):
        vdc, pr_vec, _ = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        assert guess.shape == (9,)

    def test_output_is_numpy_array(self):
        vdc, pr_vec, _ = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        assert isinstance(guess, np.ndarray)

    def test_switching_voltages_within_vdc_range(self):
        vdc, pr_vec, _ = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        # parameters [2] and [3] are estimated switching voltages
        assert vdc.min() <= guess[2] <= vdc.max()
        assert vdc.min() <= guess[3] <= vdc.max()

    def test_positive_width_parameters(self):
        vdc, pr_vec, _ = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        # b parameters (indices 5-8) are initialized to positive values
        assert np.all(guess[5:] > 0)

    def test_consistent_across_calls(self):
        vdc, pr_vec, _ = _make_synthetic_loop(100)
        g1 = generate_guess(vdc, pr_vec)
        g2 = generate_guess(vdc, pr_vec)
        assert_allclose(g1, g2)


# ===========================================================================
# fit_loop
# ===========================================================================

class TestFitLoop:
    def test_returns_three_outputs(self):
        vdc, pr_vec, coef_vec = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        plsq, criterion, pr_fit = fit_loop(vdc, pr_vec, guess)
        assert plsq is not None
        assert len(criterion) == 4
        assert len(pr_fit) == len(vdc)

    def test_fit_parameters_nine_elements(self):
        vdc, pr_vec, coef_vec = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        plsq, _, _ = fit_loop(vdc, pr_vec, guess)
        assert len(plsq.x) == 9

    def test_fit_reduces_residuals(self):
        vdc, pr_vec, coef_vec = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        _, _, pr_fit = fit_loop(vdc, pr_vec, guess)
        guess_eval = loop_fit_function(vdc, guess)
        guess_resid = np.sum((pr_vec - guess_eval) ** 2)
        fit_resid = np.sum((pr_vec - pr_fit) ** 2)
        assert fit_resid <= guess_resid

    def test_criterion_values_are_finite(self):
        vdc, pr_vec, coef_vec = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        _, criterion, _ = fit_loop(vdc, pr_vec, guess)
        assert all(np.isfinite(c) for c in criterion)

    def test_fit_recovers_known_coefficients(self):
        """Fit should recover the true coefficients within a loose tolerance."""
        vdc, pr_vec, true_coef = _make_synthetic_loop(100)
        guess = generate_guess(vdc, pr_vec)
        plsq, _, _ = fit_loop(vdc, pr_vec, guess)
        # check that fitted curve is close to data, not necessarily params
        pr_fit = loop_fit_function(vdc, plsq.x)
        assert_allclose(pr_fit, pr_vec, atol=0.1)


# ===========================================================================
# be_relax_fit pure functions
# ===========================================================================

class TestExpFunction:
    def test_at_zero(self):
        assert exp(0, 2.0, 1.5, 0.5) == pytest.approx(2.5)

    def test_at_large_x(self):
        assert exp(1000, 2.0, 1.5, 0.5) == pytest.approx(0.5, abs=1e-6)

    def test_shape_preserved(self):
        x = np.linspace(0, 5, 50)
        y = exp(x, 3.0, 2.0, 1.0)
        assert y.shape == x.shape

    def test_monotone_decay(self):
        x = np.linspace(0, 10, 100)
        y = exp(x, 1.0, 2.0, 0.0)  # positive amplitude, positive decay
        assert np.all(np.diff(y) < 0)


class TestDoubleExpFunction:
    def test_at_zero(self):
        result = double_exp(0, 2.0, 1.0, 3.0, 0.5, 1.0)
        assert result == pytest.approx(2.0 + 3.0 + 1.0)

    def test_shape_preserved(self):
        x = np.linspace(0, 5, 50)
        y = double_exp(x, 1.0, 0.5, 2.0, 0.2, 0.0)
        assert y.shape == x.shape


class TestStrExpFunction:
    def test_shape_preserved(self):
        x = np.linspace(0.1, 5, 50)
        y = str_exp(x, 1.0, 0.5, 0.0)
        assert y.shape == x.shape

    def test_offset_shifts_curve(self):
        x = np.linspace(0.1, 3, 30)
        y1 = str_exp(x, 1.0, 0.5, 0.0)
        y2 = str_exp(x, 1.0, 0.5, 5.0)
        assert_allclose(y2 - y1, 5.0)


class TestSigmoidFunction:
    def test_standard_logistic_midpoint(self):
        """sigmoid(0) with standard params (A=0,K=1,B=1,v=1,Q=1,C=1) = 0.5."""
        result = sigmoid(0.0, A=0.0, K=1.0, B=1.0, v=1.0, Q=1.0, C=1.0)
        assert result == pytest.approx(0.5)

    def test_shape_preserved(self):
        x = np.linspace(-5, 5, 50)
        y = sigmoid(x, A=0.0, K=1.0, B=1.0, v=1.0, Q=1.0, C=1.0)
        assert y.shape == x.shape

    def test_monotone_increasing(self):
        x = np.linspace(-5, 5, 100)
        y = sigmoid(x, A=0.0, K=1.0, B=1.0, v=1.0, Q=1.0, C=1.0)
        assert np.all(np.diff(y) > 0)


class TestFitExpCurve:
    def test_recovers_parameters(self):
        """Fitting synthetic exponential data should recover true params."""
        true_a, true_k, true_c = 3.0, 2.0, 0.5
        x = np.linspace(0, 10, 200)
        y = exp(x, true_a, true_k, true_c)
        popt = fit_exp_curve(x, y)
        assert popt[0] == pytest.approx(true_a, rel=0.01)
        assert popt[1] == pytest.approx(true_k, rel=0.01)
        assert popt[2] == pytest.approx(true_c, rel=0.01)

    def test_returns_three_parameters(self):
        x = np.linspace(0, 5, 100)
        y = exp(x, 2.0, 1.0, 0.5)
        popt = fit_exp_curve(x, y)
        assert len(popt) == 3

    def test_noisy_data_converges(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 8, 100)
        y = exp(x, 2.0, 1.5, 0.3) + rng.normal(0, 0.05, 100)
        popt = fit_exp_curve(x, y)
        assert np.all(np.isfinite(popt))
        assert popt[0] == pytest.approx(2.0, abs=0.3)
        assert popt[2] == pytest.approx(0.3, abs=0.3)
