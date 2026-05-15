"""
This file shows:
 - how to use engine
 - how to load pretrained weights into the models
 - how to use minimax algorithm
 - how to use MCTS (Monte Carlo Tree Search) algorithm
 - how to use network enhanced MCTS (with Policy Network and Value Network)
"""

import time
from pathlib import Path
import torch
import numpy as np
from ai.engine.game import Game
from ai.engine.piece import PieceColor
from ai.engine.game import Outcome
from ai.ai.minimax import minimax
from ai.ai.mcts import mcts_search
from ai.ai.neural_mcts import mcts_search_nn2, mcts_search_nn1
from ai.ai.policy_network import PolicyNetwork
from ai.ai.value_network import ValueNetwork
from ai.ai.download import download_model


if __name__ == "__main__":

    UPPER_PLAYER_NAME = "CLASSIC MCTS" # the name for the player that plays light pieces
    LOWER_PLAYER_NAME = "NEURAL MCTS" # the name for the player that plays dark pieces

    game = Game()
    game.board.print(UPPER_PLAYER_NAME, LOWER_PLAYER_NAME)

    moves_space = game.possible_moves_space()
    policy_model = PolicyNetwork(5, len(moves_space))
    value_model = ValueNetwork(5)

    # uncomment to load 1.0 models
    # # retrieving 1.0 policy network
    # POLICY_MODEL_URL = "https://drive.google.com/file/d/1RSN43tz27N9WY8mZAJqioTh2WA1fbslM/view?usp=sharing"
    # BASE_DIR = Path(__file__).resolve().parent
    # POLICY_PATH = BASE_DIR / "models" / "1.0"
    # POLICY_PATH.mkdir(parents=True, exist_ok=True)
    # POLICY_MODEL_NAME = "PolicyNetwork.pt"
    # if not (POLICY_PATH / POLICY_MODEL_NAME).exists():
    #     download_model(POLICY_MODEL_URL, str(POLICY_PATH / POLICY_MODEL_NAME))
    # checkpoint = torch.load(POLICY_PATH / POLICY_MODEL_NAME)
    # policy_model.load_state_dict(checkpoint["model"])

    # # retrieving 1.0 value network
    # VALUE_MODEL_URL = "https://drive.google.com/file/d/1nRVQ64_sWxXUj6QQLYEVKfFZOtucvdkm/view?usp=sharing"
    # VALUE_PATH = BASE_DIR / "models" / "1.0"
    # VALUE_PATH.mkdir(parents=True, exist_ok=True)
    # VALUE_MODEL_NAME = "ValueNetwork.pt"
    # if not (VALUE_PATH / VALUE_MODEL_NAME).exists():
    #     download_model(VALUE_MODEL_URL, str(VALUE_PATH / VALUE_MODEL_NAME))
    # checkpoint = torch.load(VALUE_PATH / VALUE_MODEL_NAME)
    # value_model.load_state_dict(checkpoint["model"])

    # uncomment to load 2.0 models
    # retrieving 2.0 policy network
    POLICY_MODEL_URL = "https://drive.google.com/file/d/1BHl0o2WcBZ_NWZf2LHr0ZAmzTzUoENRP/view?usp=sharing"
    BASE_DIR = Path(__file__).resolve().parent
    POLICY_PATH = BASE_DIR / "models" / "2.0"
    POLICY_PATH.mkdir(parents=True, exist_ok=True)
    POLICY_MODEL_NAME = "PolicyNetwork.pt"
    if not (POLICY_PATH / POLICY_MODEL_NAME).exists():
        download_model(POLICY_MODEL_URL, str(POLICY_PATH / POLICY_MODEL_NAME))
    checkpoint = torch.load(POLICY_PATH / POLICY_MODEL_NAME)
    policy_model.load_state_dict(checkpoint["model"])

    # retrieving 2.0 value network
    VALUE_MODEL_URL = "https://drive.google.com/file/d/1_d-Ra5BSBj5PXtlrqyB_Bqlexv1gPCRm/view?usp=sharing"
    VALUE_PATH = BASE_DIR / "models" / "2.0"
    VALUE_PATH.mkdir(parents=True, exist_ok=True)
    VALUE_MODEL_NAME = "ValueNetwork.pt"
    if not (VALUE_PATH / VALUE_MODEL_NAME).exists():
        download_model(VALUE_MODEL_URL, str(VALUE_PATH / VALUE_MODEL_NAME))
    checkpoint = torch.load(VALUE_PATH / VALUE_MODEL_NAME)
    value_model.load_state_dict(checkpoint["model"])

    policy_model.eval()
    value_model.eval()

    MAX_TIME_PER_MOVE = 10 # in seconds

    while game.is_in_progress():
        if game.whose_turn == PieceColor.LIGHT:
            move = mcts_search(game, PieceColor.LIGHT, MAX_TIME_PER_MOVE)
            game.make_move(move)
            # move = minimax(game, True, PieceColor.LIGHT, time.time(), MAX_TIME_PER_MOVE)
            # game.make_move(move[1])
        else:
            # move = mcts_search(game, PieceColor.DARK)
            # game.make_move(move)
            # move = mcts_search_nn1(game, policy_model, value_model, PieceColor.DARK, MAX_TIME_PER_MOVE)
            # game.make_move(move)
            move = mcts_search_nn2(game, policy_model, value_model, MAX_TIME_PER_MOVE)
            game.make_move(move)

        print("\033[H\033[J", end="")
        print("Match (Ctrl+C to interrupt match).")
        print("Pieces Evaluation: "+str(game.evaluate(PieceColor.LIGHT)))
        curr_state = game.get_state_list()
        evalu = value_model(torch.from_numpy(np.array(curr_state)).unsqueeze(0).float())[0]
        print("Value Network Evaluation: "+str(evalu))
        print(game.board.to_string(UPPER_PLAYER_NAME, LOWER_PLAYER_NAME))

    print("\033[H\033[J", end="")
    if game.final_outcome() != Outcome.DRAW:
        print("The winner is:", UPPER_PLAYER_NAME if game.final_outcome() == Outcome.LIGHT else LOWER_PLAYER_NAME, ".")
    else:
        print("The match ended in draw.")
