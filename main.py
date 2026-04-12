
import time
from engine.game import Game
from engine.piece import PieceColor
from engine.game import Outcome
from ai.minimax import minimax
from ai.mcts import mcts_search


if __name__ == "__main__":
    DEPTH = 5

    game = Game()
    game.board.print()
    time.sleep(1)
    while game.is_in_progress():
        if game.whose_turn == PieceColor.LIGHT:
            possible_moves = game.generate_potential_moves()
            move = minimax(game, DEPTH, True, PieceColor.LIGHT)
            print("Light moved")
            print("Evaluation:", game.evaluate(PieceColor.LIGHT))
            game.make_move(move[1])
        else:
            move = mcts_search(game, PieceColor.DARK)
            print("Dark moved")
            game.make_move(move)
        game.board.print()
        time.sleep(1)

    if game.final_outcome() != Outcome.DRAW:
        print("The winner is:", "Light" if game.final_outcome() == Outcome.LIGHT else "Dark")
    else:
        print("This is the draw")
    game.board.print()
