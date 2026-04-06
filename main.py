from engine.game import Game
from engine.piece import PieceColor
from engine.game import Outcome
import time
import random

if __name__ == "__main__":
    game = Game()
    game.board.print()
    time.sleep(1)
    while game.is_in_progress():
        if game.whose_turn == PieceColor.LIGHT:
            possible_moves = game.generate_potential_moves()
            print("Light:", random.choice(possible_moves))
            game.make_move(random.choice(possible_moves))
        else:
            possible_moves = game.generate_potential_moves()
            print("Dark:", random.choice(possible_moves))
            game.make_move(random.choice(possible_moves))
        game.board.print()
        time.sleep(1)

    if game.final_outcome() != Outcome.DRAW:
        print("The winner is:", "Light" if game.final_outcome() == Outcome.LIGHT else "Dark")
    else:
        print("This is the draw")
    game.board.print()