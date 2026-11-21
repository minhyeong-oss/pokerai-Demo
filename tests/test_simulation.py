from poker_ai.simulation import evaluate_practice_match, play_practice_hand


def test_end_to_end_hand_is_reproducible_and_zero_sum() -> None:
    settings = dict(
        seed=23,
        hero_index=1,
        opponent_style="calling",
        equity_simulations=40,
        rollout_simulations=20,
        baseline_simulations=30,
    )
    first = play_practice_hand(**settings)
    second = play_practice_hand(**settings)
    assert first.hero_cards == second.hero_cards
    assert first.final_board == second.final_board
    assert first.action_history == second.action_history
    assert first.payoffs_bb == second.payoffs_bb
    assert sum(first.payoffs_bb) == 0
    assert sum(first.final_stacks_bb) == 200
    assert first.decision_records


def test_small_match_reports_bb_per_100() -> None:
    result = evaluate_practice_match(
        hands=2,
        base_seed=3,
        opponent_style="random",
        equity_simulations=20,
        rollout_simulations=10,
        baseline_simulations=10,
    )
    assert result.report.hands == 2
    assert len(result.hands) == 2
