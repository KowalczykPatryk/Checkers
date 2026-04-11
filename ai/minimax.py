
from copy import deepcopy
from engine.game import Game
from engine.move import Move
from engine.piece import PieceColor

def minimax(game_state: Game, depth: int, maximizing: bool, global_maximizer: PieceColor, alpha=float('-inf'), beta=float('inf')) -> tuple[int, Move | None]:
    """
    global_maximizer - passed deep into the recursion to pass the information for whom positive score has to be calculated and for whom negative.
    Alpha is the best value that the maximizer currently can guarantee at that level or above.
    Beta is the best value (most negative) that the minimizer currently can guarantee at that level or below.
    Pruning condition is: if at any point beta <= alpha, we can prune the remaining branches.
    Alpha and beta are passed from parent to children in the tree. And in this tree MAX and MIN take turns when another recursion call. So for example MIN parent
    found so far pretty negative beta value that were returned from other children but these children were maximizing so if maximizing child found so far sth bigger than minimazing parent passed down then
    maximizing child can stop further searching. It is because maximizing child cannot take sth less than he found already and what he found is already bigger than other maximizing child so it won't
    be selected by minimizing parent.
    """

    if (depth==0) or (not game_state.is_in_progress()):
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

            tmp = minimax(game, depth-1, False, global_maximizer, alpha, beta)[0]
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

            tmp = minimax(game, depth-1, True, global_maximizer, alpha, beta)[0]
            if tmp < value:
                value = tmp
                best_movement = move

            # alpha beta pruning
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, best_movement
