from poker_ai.opponents import make_range_profile


def test_facing_bet_action_probabilities_are_normalized() -> None:
    for style in ("balanced", "tight", "calling", "aggressive"):
        profile = make_range_profile(style, board_cards=("2c", "7d", "Jh"), facing_bet=True)
        for combo in (("As", "Ah"), ("3c", "4c"), ("Kd", "Qd")):
            total = sum(profile.likelihood(action, combo) for action in ("fold", "call", "raise"))
            assert abs(total - 1.0) < 1e-12
            assert profile.likelihood("check", combo) == 0.0


def test_unopened_action_probabilities_are_normalized() -> None:
    profile = make_range_profile("aggressive", facing_bet=False)
    for combo in (("As", "Ah"), ("3c", "4c")):
        assert abs(
            profile.likelihood("check", combo) + profile.likelihood("raise", combo) - 1.0
        ) < 1e-12
        assert profile.likelihood("fold", combo) == 0.0
        assert profile.likelihood("call", combo) == 0.0
