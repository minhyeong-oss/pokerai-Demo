"""Command-line interface for the equity, advice, play, and evaluation flows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .actions import LegalActions
from .agent import ResearchPokerAgent
from .equity import monte_carlo_equity
from .logging import to_jsonable
from .observation import Observation
from .simulation import evaluate_practice_match, play_practice_hand
from .seeding import derive_seed
from .train import train_kuhn_cfr


def _print_json(value: object) -> None:
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def _street_from_board(board: Sequence[str]) -> str:
    try:
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)]
    except KeyError as exc:
        raise ValueError("board must contain 0, 3, 4, or 5 cards") from exc


def _two_player_values(hero_index: int, hero: int, villain: int) -> tuple[int, int]:
    return (hero, villain) if hero_index == 0 else (villain, hero)


def _add_simulation_options(parser: argparse.ArgumentParser, *, equity_default: int) -> None:
    parser.add_argument("--simulations", type=int, default=equity_default, help="equity samples")
    parser.add_argument("--rollouts", type=int, default=500, help="action rollout samples")
    parser.add_argument(
        "--hero-policy-master-seed", type=int, default=1,
        help="hero policy master seed",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.0,
        help="0=max mean EV, 1=max lower confidence bound",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker-ai",
        description="Offline heads-up no-limit Hold'em research AI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    equity = subparsers.add_parser("equity", help="estimate showdown equity")
    equity.add_argument("--hero", nargs=2, required=True, metavar=("CARD1", "CARD2"))
    equity.add_argument("--board", nargs="*", default=[], metavar="CARD")
    equity.add_argument("--opponents", type=int, default=1, choices=range(1, 10))
    equity.add_argument("--simulations", type=int, default=10_000)
    equity.add_argument("--seed", type=int)

    advise = subparsers.add_parser("advise", help="compare legal actions for one observation")
    advise.add_argument("--hero", nargs=2, required=True, metavar=("CARD1", "CARD2"))
    advise.add_argument("--board", nargs="*", default=[], metavar="CARD")
    advise.add_argument("--pot", type=int, required=True, help="pot in chips")
    advise.add_argument("--call-cost", type=int, required=True, help="call cost in chips")
    advise.add_argument("--hero-stack", type=int, required=True, help="hero stack in chips")
    advise.add_argument("--villain-stack", type=int, required=True, help="villain stack in chips")
    advise.add_argument("--hero-contribution", type=int, default=0)
    advise.add_argument("--villain-contribution", type=int, default=0)
    advise.add_argument("--hero-hand-contribution", type=int)
    advise.add_argument("--villain-hand-contribution", type=int)
    advise.add_argument("--min-raise-to", type=int)
    advise.add_argument("--max-raise-to", type=int)
    advise.add_argument("--hero-index", type=int, choices=(0, 1), default=0)
    advise.add_argument("--button-index", type=int, choices=(0, 1), default=1)
    advise.add_argument(
        "--opponent",
        choices=("balanced", "tight", "calling", "aggressive", "random"),
        default="balanced",
    )
    advise.add_argument("--audit", type=Path, help="append the decision to this JSONL file")
    _add_simulation_options(advise, equity_default=10_000)

    play = subparsers.add_parser("play", help="play one reset-stack PokerKit hand")
    play.add_argument("--hero-index", type=int, choices=(0, 1), default=1)
    play.add_argument(
        "--opponent",
        choices=("balanced", "tight", "calling", "aggressive", "random"),
        default="balanced",
    )
    play.add_argument("--audit", type=Path, help="append hero decisions to JSONL")
    play.add_argument("--compact", action="store_true", help="omit full decision and PHH payloads")
    _add_simulation_options(play, equity_default=2_000)
    play.add_argument("--environment-master-seed", type=int, default=0)
    play.add_argument("--villain-policy-master-seed", type=int, default=2)

    evaluate = subparsers.add_parser("evaluate", help="measure bb/100 against a baseline")
    evaluate.add_argument("--hands", type=int, default=20)
    evaluate.add_argument("--environment-master-seed", type=int, default=0)
    evaluate.add_argument("--hero-policy-master-seed", type=int, default=1)
    evaluate.add_argument("--villain-policy-master-seed", type=int, default=2)
    evaluate.add_argument(
        "--opponent",
        choices=("balanced", "tight", "calling", "aggressive", "random"),
        default="balanced",
    )
    evaluate.add_argument("--simulations", type=int, default=500)
    evaluate.add_argument("--rollouts", type=int, default=200)
    evaluate.add_argument("--risk-aversion", type=float, default=0.0)
    evaluate.add_argument("--audit", type=Path)
    evaluate.add_argument("--include-hands", action="store_true")

    train = subparsers.add_parser("train-kuhn", help="validate learning with tabular Kuhn CFR")
    train.add_argument("--iterations", type=int, default=10_000)
    return parser


def _run_equity(args: argparse.Namespace) -> int:
    result = monte_carlo_equity(
        args.hero,
        args.board,
        opponents=args.opponents,
        simulations=args.simulations,
        seed=args.seed,
    )
    _print_json(result.as_dict())
    return 0


def _run_advise(args: argparse.Namespace) -> int:
    if (args.min_raise_to is None) != (args.max_raise_to is None):
        raise ValueError("--min-raise-to and --max-raise-to must be supplied together")
    hero_index = args.hero_index
    hero_hand = (
        args.hero_contribution
        if args.hero_hand_contribution is None
        else args.hero_hand_contribution
    )
    villain_hand = (
        args.villain_contribution
        if args.villain_hand_contribution is None
        else args.villain_hand_contribution
    )
    observation = Observation(
        hand_id="cli:h0",
        decision_id="cli:h0:d0",
        hole_cards=tuple(args.hero),
        board=tuple(args.board),
        street=_street_from_board(args.board),
        hero_index=hero_index,
        button_index=args.button_index,
        actor_index=hero_index,
        pot=args.pot,
        call_cost=args.call_cost,
        stacks=_two_player_values(hero_index, args.hero_stack, args.villain_stack),
        street_contributions=_two_player_values(
            hero_index,
            args.hero_contribution,
            args.villain_contribution,
        ),
        hand_contributions=_two_player_values(hero_index, hero_hand, villain_hand),
        statuses=("active", "active"),
        public_history=(),
        legal_actions=LegalActions(
            check=args.call_cost == 0,
            call=args.call_cost > 0,
            fold=args.call_cost > 0,
            min_raise_to=args.min_raise_to,
            max_raise_to=args.max_raise_to,
        ),
    )
    with ResearchPokerAgent(
        opponent_style=args.opponent,
        equity_simulations=args.simulations,
        rollout_simulations=args.rollouts,
        policy_hand_seed=derive_seed(
            args.hero_policy_master_seed, "single", 0, hero_index, "hero"
        ),
        risk_aversion=args.risk_aversion,
        audit_path=args.audit,
    ) as agent:
        result = agent.decide(observation)
    _print_json(result.audit_record)
    return 0


def _run_play(args: argparse.Namespace) -> int:
    result = play_practice_hand(
        environment_master_seed=args.environment_master_seed,
        hero_policy_master_seed=args.hero_policy_master_seed,
        villain_policy_master_seed=args.villain_policy_master_seed,
        hero_index=args.hero_index,
        opponent_style=args.opponent,
        equity_simulations=args.simulations,
        rollout_simulations=args.rollouts,
        risk_aversion=args.risk_aversion,
        audit_path=args.audit,
    ).as_dict()
    if args.compact:
        result["decision_count"] = len(result.pop("decision_records"))
        result.pop("pokerkit_hand_history")
    _print_json(result)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_practice_match(
        hands=args.hands,
        environment_master_seed=args.environment_master_seed,
        hero_policy_master_seed=args.hero_policy_master_seed,
        villain_policy_master_seed=args.villain_policy_master_seed,
        opponent_style=args.opponent,
        equity_simulations=args.simulations,
        rollout_simulations=args.rollouts,
        risk_aversion=args.risk_aversion,
        audit_path=args.audit,
    )
    _print_json(result.as_dict(include_hands=args.include_hands))
    return 0


def _run_train(args: argparse.Namespace) -> int:
    _print_json(train_kuhn_cfr(args.iterations))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "equity": _run_equity,
        "advise": _run_advise,
        "play": _run_play,
        "evaluate": _run_evaluate,
        "train-kuhn": _run_train,
    }
    try:
        return handlers[args.command](args)
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
