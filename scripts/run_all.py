"""Run the whole analysis (Steps 1-4) and regenerate every table and figure.

Run from the repository root:   python scripts/run_all.py
Takes ~15 s (most of it is the 4,000-draw Monte Carlo in Step 4).
"""
import time

import _common  # noqa: F401
import step1_validate_s2f2
import step2_mission_emissions
import step3_contrails
import step4_climate

if __name__ == "__main__":
    t0 = time.time()
    for step in (step1_validate_s2f2, step2_mission_emissions, step3_contrails, step4_climate):
        print("\n" + "=" * 100)
        step.main()
    print(f"\nAll steps done in {time.time() - t0:.1f} s. See results/figures, results/tables "
          "and results/key_metrics.md")
