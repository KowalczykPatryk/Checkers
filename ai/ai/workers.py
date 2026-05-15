"""
Contains functions that will be run in parallel.
"""
import numpy as np
import torch
from ai.ai.policy_network import PolicyNetwork
from ai.ai.value_network import ValueNetwork
from ai.ai.neural_mcts import MCTSNodeNN2
from ai.engine.game import Outcome


def self_play_worker(args):
    """
    Function that runs self-play process and returns history of the match.
    """
    N_SIMULATIONS = 150

    game, policy_model_state, value_model_state, display, display_queue = args

    moves_space = game.possible_moves_space()
    policy_model = PolicyNetwork(5, len(moves_space))
    value_model = ValueNetwork(5)

    policy_model.load_state_dict(policy_model_state)
    value_model.load_state_dict(value_model_state)
    policy_model.eval()
    value_model.eval()
    torch.set_grad_enabled(False)
    history = []
    move_count = 0

    while game.is_in_progress():
        move_count += 1
        root = MCTSNodeNN2(game, policy_model, value_model)
        # initial expansion with dirichlet noise
        root.expand(dirichlet=True)
        if display:
            display_queue.put({
                "title": "Self-play mode active: model is playing against itself (Ctrl+C to interrupt training).\nGames are played in parallel using multiprocessing (one of them is displayed below).",
                "board": game.board.to_string("SELF-PLAY", "SELF-PLAY"),
                "move": f"Move: {move_count}"
            })
        # simulations
        for _ in range(N_SIMULATIONS):
            node = root
            # selection
            while node.children and not node.is_terminal():
                node = node.best_child()
            # evaluation / expansion
            if node.is_terminal():
                value = node.evaluate()
            else:
                node.expand()
                # choose one newly expanded child
                node = node.best_child()
                value = node.evaluate()

            # backpropagation
            node.backpropagate(value)

        moves_space = game.possible_moves_space()
        # building pi from visit counts
        visits = np.zeros(len(moves_space), dtype=np.float64)

        for child in root.children:
            for i, (start, end) in enumerate(moves_space):
                moves = game.moves_from(start, end)

                if moves and child.action == moves[0]:
                    visits[i] = child.visits
                    break

        # temperature schedule
        if move_count < 50:
            tau = max(0.15, 1.0 * (0.98 ** move_count))
        elif move_count < 70:
            tau = 0.1
        else:
            tau = 0.0

        # applying temperature safely against big exponentials
        if tau <= 0.05:
            pi = np.zeros_like(visits)
            best_idx = np.argmax(visits)
            pi[best_idx] = 1.0

        else:
            visits = visits ** (1.0 / tau)
            total = visits.sum()

            if total == 0 or np.isnan(total) or np.isinf(total):

                mask = np.array(game.possible_moves_mask(),dtype=np.float32)
                pi = mask / mask.sum()

            else:
                pi = visits / total

        history.append((game.get_state_list(), game.possible_moves_mask(), pi))

        # sample move from pi
        action_idx = np.random.choice(np.arange(len(moves_space)), p=pi)

        start_end = moves_space[action_idx]
        moves = game.moves_from(start_end[0], start_end[1])
        move = moves[0]
        game.make_move(move)

    outcome = game.final_outcome()

    if outcome == Outcome.LIGHT:
        z = 1.0
    elif outcome == Outcome.DARK:
        z = -1.0
    else:
        z = 0.0

    return (history, z)
