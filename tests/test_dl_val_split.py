"""DL early-stopping validation must come from the train tail, not the test set."""

from src.models.trainer import DL_VALIDATION_FRACTION, carve_dl_validation


def test_carve_proportional():
    assert carve_dl_validation(1000) == int(1000 * DL_VALIDATION_FRACTION)


def test_carve_minimum_one_sample():
    assert carve_dl_validation(2) == 1
    assert carve_dl_validation(10) == 1


def test_carve_never_consumes_entire_train():
    for n in (2, 5, 17, 100, 960):
        val = carve_dl_validation(n)
        assert 0 <= val < n
        assert n - val >= 1


def test_carve_zero_for_degenerate_input():
    assert carve_dl_validation(0) == 0
    assert carve_dl_validation(1) == 0


def test_val_window_is_train_tail_not_test():
    """Documented invariant: with split=100 sequences and fraction 15%,
    training uses rows [0, 85) and validation rows [85, 100) — the TEST
    region (>=100) is never touched during model selection."""
    split = 100
    val = carve_dl_validation(split)
    assert (split - val, split) == (85, 100)
