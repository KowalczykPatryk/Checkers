
from copy import deepcopy
import time
from ai.engine.game import Game
from ai.engine.move import Move
from ai.engine.piece import PieceColor

def minimax(game_state: Game, maximizing: bool, global_maximizer: PieceColor, start_time: float, max_time: float, alpha=float('-inf'), beta=float('inf'), depth: int|None = None) -> tuple[float, Move | None]:
    """
    start_time - time passed to the first minimax node as a reference to the global start time
    max_time - max time per move in seconds 
    global_maximizer - passed deep into the recursion to pass the information for whom positive score has to be calculated and for whom negative.
    Alpha is the best value that the maximizer currently can guarantee at that level or above.
    Beta is the best value (most negative) that the minimizer currently can guarantee at that level or below.
    Pruning condition is: if at any point beta <= alpha, we can prune the remaining branches.
    Alpha and beta are passed from parent to children in the tree. And in this tree MAX and MIN take turns when another recursion call. So for example MIN parent
    found so far pretty negative beta value that were returned from other children but these children were maximizing so if maximizing child found so far sth bigger than minimazing parent passed down then
    maximizing child can stop further searching. It is because maximizing child cannot take sth less than he found already and what he found is already bigger than other maximizing child so it won't
    be selected by minimizing parent.
    """

    if (depth is not None and depth==0) or (not game_state.is_in_progress()) or (time.time() - start_time >= max_time):
        return game_state.evaluate(global_maximizer), None

    if maximizing:
        value = float('-inf')
        best_movement = None
        potential_moves = game_state.generate_potential_moves()

        # there is no need to continue recursion when there is only one possible move
        if len(potential_moves) == 1:
            game = deepcopy(game_state)
            game.make_move(potential_moves[0])
            return game.evaluate(global_maximizer), potential_moves[0]
        for move in potential_moves:
            game = deepcopy(game_state)
            game.make_move(move)

            if depth is not None:
                tmp = minimax(game, False, global_maximizer, start_time, max_time, alpha, beta, depth-1)[0]
            else:
                tmp = minimax(game, False, global_maximizer, start_time, max_time, alpha, beta)[0]
            if tmp > value:
                value = tmp
                best_movement = move

            # alpha beta pruning
            alpha = max(alpha, value)
            if alpha >= beta:
                break

    else:
        value = float('inf')
        best_movement = None
        potential_moves = game_state.generate_potential_moves()

        # there is no need to continue recursion when there is only one possible move
        if len(potential_moves) == 1:
            game = deepcopy(game_state)
            game.make_move(potential_moves[0])
            return game.evaluate(global_maximizer), potential_moves[0]
        for move in potential_moves:
            game = deepcopy(game_state)
            game.make_move(move)

            if depth is not None:
                tmp = minimax(game, True, global_maximizer, start_time, max_time, alpha, beta, depth-1)[0]
            else:
                tmp = minimax(game, True, global_maximizer, start_time, max_time, alpha, beta)[0]
            if tmp < value:
                value = tmp
                best_movement = move

            # alpha beta pruning
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, best_movement
