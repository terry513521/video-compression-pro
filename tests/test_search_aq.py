"""Search algorithms that do not need ffmpeg."""

from vidopt.config import Config
from vidopt.encoding.encoders import LibSvtAv1, LibX265
from vidopt.encoding.params import EncodeParams
from vidopt.search.optimizer import _crf_guess, _screen_plan
from vidopt.search.samplers import SAMPLERS, get_sampler, params_from_unit, params_to_unit


def test_svt_aq_grid_is_the_eight_integer_pairs() -> None:
    grid = LibSvtAv1.space.aq_grid()
    assert grid == [
        (0, 1.0), (0, 2.0), (0, 3.0), (0, 4.0),
        (1, 1.0), (1, 2.0), (1, 3.0), (1, 4.0),
    ]
    assert (1, 2.0) in LibSvtAv1.space.aq_neighbors(1, 3.0)
    assert (0, 3.0) in LibSvtAv1.space.aq_neighbors(1, 3.0)


def test_x265_float_grid_respects_strength_steps() -> None:
    grid = LibX265.space.aq_grid(n_strength_steps=3)
    modes = {m for m, _ in grid}
    strengths = {s for _, s in grid}
    assert modes == {0, 1, 2, 3, 4}
    assert len(strengths) == 3
    assert len(grid) == 15


def test_svt_screen_fits_two_crfs_under_default_budget() -> None:
    aq, crfs = _screen_plan(LibSvtAv1.space, Config())
    assert len(aq) == 8
    assert len(crfs) == 2
    assert all(LibSvtAv1.space.crf_min < c < LibSvtAv1.space.crf_max for c in crfs)


def test_every_sampler_returns_clamped_points() -> None:
    space = LibSvtAv1.space
    for name in SAMPLERS:
        points = get_sampler(name)(space, 8, 1)
        assert points, name
        for p in points:
            assert space.crf_min <= p.crf <= space.crf_max
            assert p.aq_mode in space.aq_modes


def test_unit_cube_roundtrip_crf() -> None:
    space = LibSvtAv1.space
    p = space.clamp(EncodeParams(crf=40.0, aq_mode=1, aq_strength=3.0))
    back = params_from_unit(space, params_to_unit(space, p))
    assert back.aq_mode == p.aq_mode
    assert abs(back.crf - p.crf) < 0.6


def test_crf_guess_golden_is_inside_bracket() -> None:
    g = _crf_guess("golden", 20.0, 90.0, 50.0, 70.0, [(20.0, 90.0), (50.0, 70.0)], 85.0)
    assert 20.0 < g < 50.0


def test_crf_guess_bisect_uses_secant() -> None:
    g = _crf_guess("bisect", 20.0, 90.0, 50.0, 70.0, [(20.0, 90.0), (50.0, 70.0)], 85.0)
    # 20 + (90-85)*(50-20)/(90-70) = 20 + 5*30/20 = 27.5
    assert abs(g - 27.5) < 1e-6


def test_crf_guess_brent_uses_inverse_quadratic() -> None:
    history = [(20.0, 95.0), (40.0, 80.0), (60.0, 65.0)]
    g = _crf_guess("brent", 20.0, 95.0, 60.0, 65.0, history, 85.0)
    assert 20.0 < g < 60.0


def test_adaptive_strategies_spend_the_explore_budget() -> None:
    from dataclasses import replace

    from vidopt.search.adaptive import explore_adaptive
    from vidopt.search.cache import TrialRecord

    def evaluate(params: EncodeParams) -> TrialRecord:
        return TrialRecord(
            segment_hash="seg",
            encoder="libsvtav1",
            params=params,
            vmaf_model="vmaf_v0.6.1neg",
            n_subsample=2,
            ref_bytes=1_000_000,
            out_bytes=max(1_000, int(400_000 - 2_000 * params.crf)),
            vmaf=max(0.0, 110.0 - params.crf),
        )

    config = replace(
        Config(),
        search=replace(Config().search, n_explore=8, n_init=4, sampler="lhs"),
    )
    for strategy in ("bayes", "tpe", "cmaes"):
        trials = explore_adaptive(evaluate, LibSvtAv1(), config, strategy, 85.0)
        assert 4 <= len(trials) <= 8, strategy
        assert all(t.vmaf >= 0 for t in trials)
