"""Exercise 3: report how long the model keeps matching the real engine, without overstating it.

Each test game (a "rollout") ends in one of three ways:
  - the model's board first differs from the engine's at some step   -> an event at that step
  - the engine's game ends and the model matched every step           -> censored at that step
  - the rollout reaches the step cap with no difference                -> censored at the cap

A censored rollout tells you "no mismatch up to here", not "never mismatches". The Kaplan-Meier
estimate uses exactly that information. S(t) is the estimated chance a rollout is still matching
after step t:

    S(t) = product over event steps u <= t of (1 - mismatches at u / rollouts still observed at u)

A rollout is still observed at step u when its time is at least u.
"""

HORIZONS = (50, 100, 200, 500, 1000)


def kaplan_meier(times, events, horizons=HORIZONS):
    """times[i]: the step rollout i mismatched or was censored. events[i]: True if it mismatched.

    Returns:
      "S": {str(h): S(h) rounded to 4 places} for each horizon h,
      "rollouts_still_observed_at_step": {str(h): number of rollouts with time >= h},
      "median_step": the first event step where S drops below 0.5, or None if it never does.
    """
    event_steps = sorted({t for t, e in zip(times, events) if e})
    curve, s = [], 1.0
    for u in event_steps:
        at_risk = sum(1 for t in times if t >= u)
        mismatches = sum(1 for t, e in zip(times, events) if e and t == u)
        s *= 1 - mismatches / at_risk
        curve.append((u, s))

    def survival_at(h):
        value = 1.0
        for u, v in curve:
            if u > h:
                break
            value = v
        return value

    return {
        "S": {str(h): round(survival_at(h), 4) for h in horizons},
        "rollouts_still_observed_at_step": {str(h): sum(1 for t in times if t >= h) for h in horizons},
        "median_step": next((u for u, v in curve if v < 0.5), None),
    }
