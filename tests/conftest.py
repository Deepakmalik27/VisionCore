"""Make check() failures actually fail the test.

50 test files in this directory share a hand-rolled harness:

    FAILED = []
    def check(cond, label, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            FAILED.append(label)

It prints and appends. It does not raise. The `if FAILED: return 1` that
consumes the list lives in each file's `__main__` block, which pytest never
runs -- so under pytest a failing check printed "FAIL" and the run still
reported PASSED. `pytest tests/ -q` said "389 passed" while three checks were
failing, and had been for some time.

This fixture reads each test's own module-level FAILED list and fails the test
if the list grew while the test ran. No change to the 50 files, and their
`python tests/test_x.py` entry points keep working exactly as before.
"""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Run in the CALL phase, so a failure reports as a failure.

    A teardown fixture would work too, but pytest classifies a teardown raise as
    an ERROR, which reads like harness trouble rather than a failed assertion.
    """
    failed = getattr(item.module, "FAILED", None)
    before = len(failed) if isinstance(failed, list) else None

    outcome = yield

    if before is None:
        return
    if outcome.excinfo is not None:      # the test already failed on its own
        return
    new = failed[before:]
    if new:
        pytest.fail(
            f"{len(new)} check() assertion(s) failed but did not raise:\n  - "
            + "\n  - ".join(str(x) for x in new),
            pytrace=False)


_PRISTINE = {}


def pytest_configure(config):
    """Snapshot kevacv's constants BEFORE collection imports any test module.

    tests/test_phantom_fast_patience.py lines 86-113 call run() at MODULE level,
    and run() does `ph.PHANTOM_FAST_CV_RATIO = fast_ratio` on the real module
    global. Module level means import time, and pytest imports every test module
    during collection -- so by the time the first test starts, the flag is
    already stuck at the last value that file set (0.30), for the whole session.

    A per-test snapshot cannot see this: it snapshots the ALREADY-polluted value
    and faithfully restores the pollution. So the pristine copy has to be taken
    here, in pytest_configure, which runs before collection.
    """
    import importlib
    for name in ("kevacv.config", "kevacv.phantoms"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr, val in vars(mod).items():
            if attr.isupper() and isinstance(val, (bool, int, float, str, type(None))):
                _PRISTINE[(name, attr)] = val


@pytest.fixture(autouse=True)
def _restore_kevacv_constants():
    """Give every test the pristine constants, whatever ran before it.

    Restored on the way IN as well as out: the import-time damage above is
    already done before the first test, so only a setup-side restore undoes it.
    """
    import sys

    def _restore():
        for (name, attr), val in _PRISTINE.items():
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, attr, object()) != val:
                setattr(mod, attr, val)

    _restore()
    yield
    _restore()
