
import time
import random
from engine.game import Game
from engine.piece import PieceColor
from engine.game import Outcome
from ai.minimax import minimax


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
            possible_moves = game.generate_potential_moves()
            print("Dark moved")
            game.make_move(random.choice(possible_moves))
        game.board.print()
        time.sleep(1)

    if game.final_outcome() != Outcome.DRAW:
        print("The winner is:", "Light" if game.final_outcome() == Outcome.LIGHT else "Dark")
    else:
        print("This is the draw")
    game.board.print()
