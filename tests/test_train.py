from poker_ai.train import KuhnCFRTrainer, train_kuhn_cfr


def test_kuhn_cfr_converges_near_known_game_value() -> None:
    result = train_kuhn_cfr(5_000)
    assert abs(result.expected_value_player0 - (-1.0 / 18.0)) < 0.03
    assert len(result.strategy) == 12


def test_kuhn_cfr_is_deterministic() -> None:
    assert train_kuhn_cfr(50) == train_kuhn_cfr(50)


def test_reusing_trainer_starts_a_fresh_self_contained_run() -> None:
    trainer = KuhnCFRTrainer()
    first = trainer.train(50)
    second = trainer.train(50)
    assert first == second
