from poker_ai.simulation import evaluate_practice_match, play_practice_hand


def test_end_to_end_hand_is_reproducible_and_zero_sum() -> None:
    settings = dict(
        environment_master_seed=23,
        hero_policy_master_seed=24,
        villain_policy_master_seed=25,
        run_id="simulation-test",
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
        environment_master_seed=3,
        hero_policy_master_seed=4,
        villain_policy_master_seed=5,
        run_id="match-test",
        opponent_style="random",
        equity_simulations=20,
        rollout_simulations=10,
        baseline_simulations=10,
    )
    assert result.report.hands == 2
    assert len(result.hands) == 2
    assert result.hands[0].deck_seed == result.hands[1].deck_seed


def test_run_id_does_not_change_cards_actions_or_settlement() -> None:
    settings = dict(
        environment_master_seed=31,
        hero_policy_master_seed=32,
        villain_policy_master_seed=33,
        hero_index=1,
        opponent_style="random",
        equity_simulations=20,
        rollout_simulations=10,
        baseline_simulations=10,
    )
    first = play_practice_hand(**settings, run_id="run-a")
    second = play_practice_hand(**settings, run_id="run-b")

    assert first.hero_cards == second.hero_cards
    assert first.final_board == second.final_board
    assert first.payoffs_bb == second.payoffs_bb
    assert [event.type for event in first.action_history] == [
        event.type for event in second.action_history
    ]
    assert [getattr(event, "action", None) for event in first.action_history] == [
        getattr(event, "action", None) for event in second.action_history
    ]


def test_policy_master_seed_cannot_change_the_deck_stream() -> None:
    common = dict(
        environment_master_seed=41,
        villain_policy_master_seed=43,
        run_id="seed-separation",
        equity_simulations=20,
        rollout_simulations=10,
        baseline_simulations=10,
    )
    first = play_practice_hand(**common, hero_policy_master_seed=42)
    second = play_practice_hand(**common, hero_policy_master_seed=999)
    assert first.deck_seed == second.deck_seed
    assert first.hero_cards == second.hero_cards
