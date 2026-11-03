"""Write tests/fixtures/validation_rollouts.json: when each of the 200 validation rollouts of the shipped model mismatched
or was censored, so tests/test_learner_survival.py can check kaplan_meier() against Dev Suthar's published dry run
(eval/dryrun_run1_validation.json). Validation rows only (episode seed % 5 == 1); held-out rows are never read.

  python scripts/make_survival_fixture.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import eval_heldout as ev  # noqa: E402


def main():
    d = ev.load_split("run1", "validation")
    x14, _ = ev.common.make_14(d)
    sess = ev.make_session()
    per = ev.rollouts(sess, d, x14, "bfs", ev.ROLLOUTS_PER_KIND, ev.ROLLOUT_CAP, seed=1)
    per += ev.rollouts(sess, d, x14, "random", ev.ROLLOUTS_PER_KIND, ev.ROLLOUT_CAP, seed=2)
    published = json.load(open(os.path.join(ROOT, "eval", "dryrun_run1_validation.json")))["rollout_all_200"]
    mine = ev.summarise(per, ev.ROLLOUT_CAP)
    assert mine["headline"] == published["headline"], "these rollouts do not reproduce the published dry run"
    out = {"source": "200 validation rollouts (100 BFS, 100 random non-suicidal) of web/model/snake.onnx, cap 1000, as in scripts/eval_heldout.py",
           "times": [r["first_mismatch"] or r["steps"] for r in per], "events": [r["first_mismatch"] is not None for r in per],
           "published_kaplan_meier": published["headline"]["kaplan_meier_P_no_mismatch_yet"]}
    os.makedirs(os.path.join(ROOT, "tests", "fixtures"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "tests", "fixtures", "validation_rollouts.json"), "w"), indent=1)
    print("wrote tests/fixtures/validation_rollouts.json; matches the published dry run:", out["published_kaplan_meier"]["S"])


if __name__ == "__main__":
    main()
