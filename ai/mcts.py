"""
Monte Carlo Tree Search
"""
from __future__ import annotations

import math
import random
from copy import deepcopy
from engine.game import Game
from engine.move import Move
from engine.game import Outcome
from engine.piece import PieceColor


class MCTSNode:
    """
    Node in the tree
    """
    def __init__(self, game_state: Game, player: PieceColor, parent: MCTSNode | None = None, action: Move | None = None) -> None:
        self.game_state: Game = deepcopy(game_state)
        self.parent: MCTSNode | None = parent
        self.action: Move | None = action
        self.player: PieceColor = player
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = self.game_state.generate_potential_moves()

    def is_terminal(self) -> bool:
        """
        Check if node is terminal
        """
        return not self.game_state.is_in_progress()

    def is_fully_expanded(self) -> bool:
        """
        Check if all actions (possible moves) are explored
        """
        return len(self.untried_actions) == 0

    def expand(self) -> MCTSNode:
        """
        Expand node
        """
        action: Move = self.untried_actions.pop()
        new_game_state = deepcopy(self.game_state)

        new_game_state.make_move(action)

        child = MCTSNode(new_game_state, self.player, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c: float = 1.4) -> MCTSNode:
        """
        Select best child using UCB (Upper Confidence Bounds)
        """
        for child in self.children:
            if child.visits == 0:
                return child

        def ucb(child):
            exploit = child.wins / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            return exploit + explore

        return max(self.children, key=ucb)

    def rollout(self) -> Outcome:
        """
        Plays random moves until the game finishes
        """
        state = deepcopy(self.game_state)

        while True:
            outcome: Outcome = state.final_outcome()
            if outcome != Outcome.NOT_FINISHED:
                return outcome

            actions = state.generate_potential_moves()
            if not actions:
                return state.final_outcome()

            move = random.choice(actions)
            state.make_move(move)

    def backpropagate(self, outcome: Outcome) -> None:
        """
        Updates the wins and visits from simulation results.
        """
        self.visits += 1

        if outcome == Outcome.DRAW:
            self.wins += 0.5
        elif (self.player == PieceColor.LIGHT and outcome == Outcome.LIGHT) or (
            self.player == PieceColor.DARK and outcome == Outcome.DARK
        ):
            self.wins += 1.0

        if self.parent:
            self.parent.backpropagate(outcome)


def mcts_search(game_state: Game, player: PieceColor, iterations: int = 1000) -> Move:
    """
    Runs the complete MCTS process:
    - Selection
    - Expansion
    - Simulation
    - Backpropagation
    """
    root = MCTSNode(game_state, player)

    for _ in range(iterations):
        node = root

        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        outcome = node.rollout()
        node.backpropagate(outcome)


    best = max(root.children, key=lambda c: c.visits)
    return best.action
