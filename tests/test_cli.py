import json

from poker_ai.cli import main


def test_equity_command_outputs_json(capsys) -> None:
    assert main([
        "equity", "--hero", "As", "Ah", "--simulations", "20", "--seed", "4"
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["simulations"] == 20
    assert payload["seed"] == 4


def test_advise_command_logs_chosen_action(capsys) -> None:
    code = main([
        "advise",
        "--hero", "As", "Qs",
        "--pot", "300",
        "--call-cost", "100",
        "--hero-stack", "9900",
        "--villain-stack", "9800",
        "--hero-contribution", "100",
        "--villain-contribution", "200",
        "--min-raise-to", "400",
        "--max-raise-to", "10000",
        "--simulations", "20",
        "--rollouts", "10",
        "--hero-policy-master-seed", "9",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"]["action"]["kind"] in {"fold", "call", "raise"}


def test_play_compact_command(capsys) -> None:
    code = main([
        "play", "--environment-master-seed", "2", "--simulations", "20", "--rollouts", "10",
        "--compact",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert sum(payload["payoffs_bb"]) == 0
    assert "pokerkit_hand_history" not in payload
