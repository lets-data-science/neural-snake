"""Exercise 3 (learner/survival.py): Kaplan-Meier survival. Run: python -m pytest tests/test_learner_survival.py -q"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from learner.survival import kaplan_meier  # noqa: E402


def test_a_small_example_by_hand():
    # Five rollouts: mismatches at steps 3, 5 and 8; censored at 5 and 10.
    km = kaplan_meier([3, 5, 5, 8, 10], [True, False, True, True, False], horizons=(2, 3, 5, 8, 10))
    # step 3: 5 still observed, 1 mismatch -> 4/5. Step 5: 4 still observed, 1 mismatch -> 4/5 * 3/4.
    # Step 8: 2 still observed (8 and 10), 1 mismatch -> 3/5 * 1/2.
    assert km["S"] == {"2": 1.0, "3": 0.8, "5": 0.6, "8": 0.3, "10": 0.3}
    assert km["rollouts_still_observed_at_step"] == {"2": 5, "3": 5, "5": 4, "8": 2, "10": 1}
    assert km["median_step"] == 8


def test_censored_rollouts_are_not_counted_as_mismatches():
    km = kaplan_meier([20, 40, 60], [False, False, False], horizons=(10, 50, 100))
    assert km["S"] == {"10": 1.0, "50": 1.0, "100": 1.0}
    assert km["median_step"] is None  # "not reached": the curve never drops below 0.5


def test_the_median_is_reached_only_when_the_curve_drops_below_one_half():
    km = kaplan_meier([1, 1, 2, 9], [True, True, False, False], horizons=(1, 5))
    assert km["S"] == {"1": 0.5, "5": 0.5}  # exactly 0.5 is not below 0.5
    assert km["median_step"] is None


def test_it_reproduces_the_published_validation_curve():
    fixture = json.load(open(os.path.join(ROOT, "tests", "fixtures", "validation_rollouts.json")))
    km = kaplan_meier(fixture["times"], fixture["events"])
    published = fixture["published_kaplan_meier"]
    assert km["S"] == published["S"]
    assert km["rollouts_still_observed_at_step"] == published["rollouts_still_observed_at_step"]
    assert km["median_step"] is None and published["median"] == "not reached"
