"""
Monte Carlo Tree Search enforced by the Policy NN and Value NN.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch
import random
from copy import deepcopy
from ai.engine.game import Game
from ai.engine.move import Move
from ai.engine.game import Outcome
from ai.engine.piece import PieceColor
from ai.ai.policy_network import PolicyNetwork
from ai.ai.value_network import ValueNetwork


class MCTSNodeNN1:
    """
    Node in the MCTS (Monte Carlo Tree Search) that uses policy network during simulation step.
    """
    def __init__(self, game_state: Game, policy_model: PolicyNetwork, value_model: ValueNetwork, player: PieceColor, parent: MCTSNodeNN1 | None = None, action: Move | None = None) -> None:
        """
        action is the move that created this node
        """
        self.game_state: Game = deepcopy(game_state)
        self.parent: MCTSNodeNN1 | None = parent
        self.action: Move | None = action
        self.player: PieceColor = player
        self.children: list[MCTSNodeNN1] = []
        self.visits = 0
        self.wins = 0.0
        self.c = 1.4
        self.untried_actions = self.game_state.generate_potential_moves()
        self.policy_model = policy_model
        self.value_model = value_model


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

    def expand(self) -> MCTSNodeNN1:
        """
        Makes expansion step in the MCTS process.
        It means that new child will be added to this node
        and it is returned so that simulation step can be
        performed on it.
        """
        action: Move = self.untried_actions.pop()
        new_game_state = deepcopy(self.game_state)

        new_game_state.make_move(action)

        child = MCTSNodeNN1(new_game_state, self.policy_model, self.value_model, self.player, parent=self, action=action)
        self.children.append(child)
        return child

    def ucb(self, child):
        """
        Upper Bound Confidence
        """
        exploit = child.wins / child.visits # win rate
        explore = self.c * math.sqrt(math.log(self.visits) / child.visits)
        return exploit + explore

    def best_child(self) -> MCTSNodeNN1:
        """
        Makes selection step in the MCTS process.
        Select best child using UCB (Upper Confidence Bounds)
        """
        for child in self.children:
            # children that were never visited have potential to be good
            if child.visits == 0:
                return child

        return max(self.children, key=self.ucb)

    def rollout(self) -> Outcome:
        """
        Makes simulation step in the MCTS process.
        Plays moves selected by policy network until the game finishes.
        Policy network decides which move to explore because it looks 
        promising to the pretrained neural network.
        """
        state = deepcopy(self.game_state)

        while state.is_in_progress():
            # policy network selects promising moves
            mask = np.array(state.possible_moves_mask(), dtype=np.float32)
            
            with torch.inference_mode():
                curr_state = state.get_state_list()
                action_probability_dist = self.policy_model(torch.from_numpy(np.array(curr_state)).unsqueeze(0).float())
                action_probability_dist = action_probability_dist.squeeze(0).numpy()
                action_probability_dist = mask * action_probability_dist
                total_probability = action_probability_dist.sum()
                action_probability_dist = action_probability_dist / total_probability
                
                rng = np.random.default_rng()
                moves_space = state.possible_moves_space()
                action_idx = rng.choice(np.arange(len(moves_space)), p=action_probability_dist)
                start_end = moves_space[action_idx]
                moves = state.moves_from(start_end[0], start_end[1])
                state.make_move(moves[0])
        
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


def mcts_search_nn1(game_state: Game, policy_model: PolicyNetwork, value_model: ValueNetwork, player: PieceColor, max_time: float) -> Move:
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
    the root only child with the most visits is returned as best.
    It means that it was generally most frequently selected as the best child by UCB.

    Parameters:
        game_state: the state from which we want to start search.
        player: the player piece color for whom we want to calculate the best action (move).
        iterations: the number of times we want to traverse full MCTS process described above.
    """
    possible_moves = game_state.generate_potential_moves()
    if len(possible_moves) == 1:
        return possible_moves[0]
    root = MCTSNodeNN1(game_state, policy_model, value_model, player)

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

class MCTSNodeNN2:
    """
    Node in the MCTS (Monte Carlo Tree Search) that uses probabilities of the policy network and estimates of the value network.
    """
    def __init__(self, game_state: Game, policy_model: PolicyNetwork, value_model: ValueNetwork, parent: MCTSNodeNN2 | None = None, action: Move | None = None, prior: float = 0.0) -> None:
        self.game_state = deepcopy(game_state)
        self.parent = parent
        self.action = action

        self.policy_model = policy_model
        self.value_model = value_model

        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.c_puct = 1.4

    def is_terminal(self) -> bool:
        """
        Check if node is terminal - the state is final.
        """
        return not self.game_state.is_in_progress()

    def q_value(self):
        """
        How on average this node is estimated by the value network
        """
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    def puct(self, child):
        """
        Predictor + Upper Confidence bounds applied to Trees
        Extends UCT/UCB to use P - prior probability / predictor / policy network
        """
        q = child.q_value()
        u = self.c_puct * child.prior * math.sqrt(self.visits + 1) / (1 + child.visits)
        return q + u
    
    def best_child(self):
        """
        Returns the best child of the current node that has the highest PUCT.
        """
        return max(self.children, key=self.puct)

    def expand(self, dirichlet: bool = False) -> list[MCTSNodeNN2]:
        """
        Expands all children at once because we have immediate feedback from value network
        Noise from Dirichlet distribution is added to the probabilities of the children of the root node.
        It helps explore strange looking openings.
        If child gets high dirichlet it will have higher exploration factor.
        """

        with torch.inference_mode():
            state_tensor = torch.tensor(self.game_state.get_state_list(), dtype=torch.float32).unsqueeze(0)
            probs = self.policy_model(state_tensor)[0].numpy()

        moves_space = self.game_state.possible_moves_space()
        mask  = np.array(self.game_state.possible_moves_mask())
        if dirichlet:
            epsilon = 0.25
            # small alpha means values like [0.98,0.01,0.01] "spikes"
            # big alpha means values like [0.33,0.33,0.34] "smooth"
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(moves_space))
            probs = (1 - epsilon) * probs + epsilon * noise

        probs = probs * mask
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = mask / mask.sum()

        children = []

        for (prob, start_end) in zip(probs, moves_space):
            start, end = start_end
            if prob > 0:
                move = self.game_state.moves_from(start, end)[0]
                new_game_state = deepcopy(self.game_state)
                new_game_state.make_move(move)
                child = MCTSNodeNN2(new_game_state, self.policy_model, self.value_model, parent=self, action=move, prior=prob)
                self.children.append(child)
                children.append(child)

        return children
    
    def evaluate(self) -> float:
        """
        Evaluates state of the game after expansion.
        If this state is final then returns value based on the player passed from the root.
        If not then value network estimates the state of the game.
        """
        if not self.game_state.is_in_progress():

            outcome = self.game_state.final_outcome()

            if outcome == Outcome.DRAW:
                return 0.0

            if (outcome == Outcome.LIGHT and self.game_state.whose_turn == PieceColor.LIGHT):
                return 1.0

            if (outcome == Outcome.DARK and self.game_state.whose_turn == PieceColor.DARK):
                return 1.0

            return -1.0

        with torch.inference_mode():
            state_tensor = torch.tensor(self.game_state.get_state_list(), dtype=torch.float32).unsqueeze(0)
            value = self.value_model(state_tensor)[0]

        value = float(value[0] - value[1])
        if self.game_state.whose_turn == PieceColor.DARK:
            return -value
        return value

    def backpropagate(self, value):
        """
        Q value gathers positive estimates of the value network but what
        is positive for the one player is negative for the other.
        """
        self.visits += 1

        self.value_sum += value

        if self.parent:
            self.parent.backpropagate(-value)


def mcts_search_nn2(game_state: Game, policy_model: PolicyNetwork, value_model: ValueNetwork, max_time: float) -> Move:
    """
    Runs the modified MCTS process:
    - Selection
    - Expansion
    - Backpropagation
    In every iteration we start from the root node.
    If node has no children then all are expanded at once.
    Best child is selected using PUCT.
    There is no simluation step - it is replaced by the value network.
    Important is that during backpropagation expanded nodes have their stats updated.
    PUCT during selection step takes into account sum of estimated values and visits. But at the final decision at
    the root only child with the most visits is returned as best.
    It means that it was generally most frequently selected as the best child by PUCT.

    Parameters:
        game_state: the state from which we want to start search.
        player: the player piece color for whom we want to calculate the best action (move).
        iterations: the number of times we want to traverse full MCTS process described above.
    """
    possible_moves = game_state.generate_potential_moves()

    if len(possible_moves) == 1:
        return possible_moves[0]

    root = MCTSNodeNN2(game_state, policy_model, value_model)

    start_time = time.time()
    while True:

        node = root

        # selection
        while node.children and not node.is_terminal():
            node = node.best_child()

        # expansion
        if not node.is_terminal():

            if not node.children:
                node.expand()

            node = node.best_child()

        # evaluation
        value = node.evaluate()

        node.backpropagate(value)

        if time.time() - start_time >= max_time:
            break

    if not root.children:
        return random.choice(possible_moves)

    best = max(root.children, key=lambda c: c.visits)

    return best.action
