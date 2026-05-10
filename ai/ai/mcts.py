"""
Monte Carlo Tree Search
"""
from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from ai.engine.game import Game
from ai.engine.move import Move
from ai.engine.game import Outcome
from ai.engine.piece import PieceColor


class MCTSNode:
    """
    Node in the MCTS (Monte Carlo Tree Search).
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
        Check if node is terminal.
        """
        return not self.game_state.is_in_progress()

    def is_fully_expanded(self) -> bool:
        """
        Check if all actions (possible moves) that can be taken from this node are explored.
        """
        return len(self.untried_actions) == 0

    def expand(self) -> MCTSNode:
        """
        Makes expansion step in the MCTS process.
        It means that new child will be added to this node
        and it is returned so that simulation step can be
        performed on it.
        """
        action: Move = self.untried_actions.pop()
        new_game_state = deepcopy(self.game_state)

        new_game_state.make_move(action)

        child = MCTSNode(new_game_state, self.player, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c: float = 1.4) -> MCTSNode:
        """
        Makes selection step in the MCTS process.
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
        Makes simulation step in the MCTS process.
        Plays random moves until the game finishes.
        Can be extended to use policy network that will decide
        which move to explore because it looks promising to the
        pretrained neural network.
        """
        state = deepcopy(self.game_state)

        while state.is_in_progress():

            actions = state.generate_potential_moves()

            move = random.choice(actions)
            state.make_move(move)

        return state.final_outcome()

    def backpropagate(self, outcome: Outcome) -> None:
        """
        Makes backproapgation step in the MCTS process.
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


def mcts_search(game_state: Game, player: PieceColor, max_time: float) -> Move:
    """
    Runs the complete MCTS process:
    - Selection
    - Expansion
    - Simulation
    - Backpropagation
    In every iteration we start from the root node.
    If current node has all possible children (is expanded) then best child is selected using UCB.
    If current node can have another child (there is action that can be taken) then this
    new child is added to this node and new child is simulated.
    Important is that during backpropagation expanded nodes have their stats updated.
    UCB during selection takes into account wins and visits. But at the final decision at
    the root only child with the most visits is returned as best because if it has the most visits
    it means that it was generally most frequently selected as the best child by UCB.

    Parameters:
        game_state: the state from which we want to start search.
        player: the player piece color for whom we want to calculate the best action (move).
        iterations: the number of times we want to traverse full MCTS process described above.
    """
    possible_moves = game_state.generate_potential_moves()
    if len(possible_moves) == 1:
        return possible_moves[0]
    root = MCTSNode(game_state, player)

    start_time = time.time()
    while True:
        node = root

        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        outcome = node.rollout()
        node.backpropagate(outcome)
        if time.time() - start_time >= max_time:
            break

    if not root.children:
        return random.choice(possible_moves)


    best = max(root.children, key=lambda c: c.visits)
    return best.action
