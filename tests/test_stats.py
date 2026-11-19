import math

import pytest

from crsi.stats import paired, signed_rank, t_quantile, t_two_sided_p


def test_t_distribution_matches_the_papers_reported_test():
    # Section 4.4: paired t(9) = 4.06, p = 0.003; 95% CI [1.05, 3.68] around a mean of 2.36.
    assert t_two_sided_p(4.06, 9) == pytest.approx(0.00284, abs=5e-5)
    critical = t_quantile(0.975, 9)
    assert critical == pytest.approx(2.2622, abs=1e-4)
    half = critical * 2.36 / 4.06
    assert (round(2.36 - half, 2), round(2.36 + half, 2)) == (1.05, 3.67)  # 3.68 in the paper, from unrounded inputs


def test_t_distribution_reference_values():
    assert t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    assert t_two_sided_p(2.0, 1) == pytest.approx(0.2952, abs=1e-4)
    assert t_two_sided_p(-2.0, 30) == pytest.approx(0.0546, abs=1e-4)
    assert t_quantile(0.975, 1) == pytest.approx(12.706, abs=1e-3)
    assert t_quantile(0.95, 120) == pytest.approx(1.6577, abs=1e-4)


def test_paired_interval_on_a_worked_example():
    result = paired([11, 12, 13, 14, 15], [10, 10, 10, 10, 10])
    assert (result.n, result.df, result.wins) == (5, 4, 5)
    assert result.mean == pytest.approx(3.0) and result.sd == pytest.approx(math.sqrt(2.5))
    assert result.t == pytest.approx(4.2426, abs=1e-4) and result.p_t == pytest.approx(0.01324, abs=1e-4)
    assert result.ci == pytest.approx((1.0368, 4.9632), abs=1e-3)
    assert result.d_z == pytest.approx(3.0 / math.sqrt(2.5))
    assert result.p_signed_rank == pytest.approx(2 / 32)  # all five positive: 2 * (1/2)^5
    assert "paired t(4) = 4.24" in result.describe()


def test_signed_rank_is_exact_and_matches_the_papers_check():
    # Nine of ten pairs favour SIM. With the one loss ranked second smallest, W- = 2 and the
    # exact two-sided p is 2 * 3/1024 = 0.006, the value the paper reports.
    diffs = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    w_plus, w_minus, p = signed_rank(diffs)
    assert (w_plus, w_minus) == (53.0, 2.0) and p == pytest.approx(6 / 1024)
    assert signed_rank([-d for d in diffs])[2] == pytest.approx(6 / 1024)


def test_signed_rank_handles_ties_and_zero_differences():
    w_plus, w_minus, p = signed_rank([0.0, 1.0, 1.0, -1.0, 2.0])
    assert (w_plus, w_minus) == (8.0, 2.0)  # ranks 2, 2, 2, 4 after dropping the zero
    assert p == pytest.approx(2 * 4 / 16)  # W+ >= 8 in 4 of 16 sign assignments: three ways to 8, one to 10
    assert signed_rank([0.0, 0.0]) == (0.0, 0.0, 1.0)


def test_paired_rejects_unpaired_or_too_few_scores():
    with pytest.raises(ValueError):
        paired([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired([1.0], [0.0])
    constant = paired([3.0, 3.0, 3.0], [1.0, 1.0, 1.0])
    assert constant.ci == (2.0, 2.0) and constant.p_t == 0.0
