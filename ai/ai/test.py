
import time
import torch
from tqdm import tqdm
import numpy as np
from ai.ai.value_network import ValueNetwork
from ai.ai.policy_network import PolicyNetwork
from ai.engine.game import Game
from ai.engine.piece import PieceColor
from ai.engine.game import Outcome
from ai.ai.minimax import minimax
from ai.ai.neural_mcts import mcts_search_nn2
from ai.ai.mcts import mcts_search

def test(policy_model: PolicyNetwork, value_model: ValueNetwork) -> float:
    """
    Returns win rate of two algorithms against each other.
    """
    game = Game()
    policy_model.eval()
    value_model.eval()

    TIME_PER_MOVE = 5

    score = 0
    N_GAMES = 10
    for g_idx in tqdm(range(N_GAMES), desc="Testing"):
        game = Game()
        while game.is_in_progress():
            if game.whose_turn == PieceColor.LIGHT:
                # move = minimax(game, True, PieceColor.LIGHT, time.time(), TIME_PER_MOVE)
                # game.make_move(move[1])
                move = mcts_search(game, PieceColor.LIGHT, TIME_PER_MOVE)
                game.make_move(move)
            else:
                move = mcts_search_nn2(game, policy_model, value_model, TIME_PER_MOVE)
                game.make_move(move)
            tqdm.write("\033[H\033[J", end="")
            tqdm.write("Testing mode active: model is playing against CLASSIC MCTS (Ctrl+C to interrupt training/testing).")
            tqdm.write(f"Game {g_idx+1} / {N_GAMES}")
            tqdm.write("Current Score: "+str(score))
            tqdm.write("Pieces Evaluation:"+str(game.evaluate(PieceColor.LIGHT)))
            curr_state = game.get_state_list()
            evalu = value_model(torch.from_numpy(np.array(curr_state)).unsqueeze(0).float())[0]
            tqdm.write("Value Network Evaluation:"+str(evalu))
            tqdm.write(game.board.to_string("CLASSIC MCTS", "NEURAL MCTS"))

        if game.final_outcome() == Outcome.DRAW:
            score += 0.5
        elif game.final_outcome() == Outcome.DARK:
            score += 1.0
    policy_model.train()
    value_model.train()
    tqdm.write("\033[H\033[J", end="")
    return score
