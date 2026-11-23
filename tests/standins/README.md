Stand-ins for the parts of Booster's booster_deploy that deploy/booster_t2 uses, so its glue can be tested without
running vendor code. Each mirrors the vendor's names (tests/test_booster_t2.py compares them with the vendor source,
read as text, when ~/booster_deploy exists); the bodies are the minimum the task touches.
