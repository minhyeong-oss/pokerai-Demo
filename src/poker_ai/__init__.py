"""Heads-up no-limit Hold'em research AI.

The public boundary is deliberately small: an :class:`Observation` containing
only information visible to a player is transformed into a logged decision.
PokerKit remains behind the environment adapter and is never exposed to the
policy as a source of hidden cards.
"""

__version__ = "0.1.0"
