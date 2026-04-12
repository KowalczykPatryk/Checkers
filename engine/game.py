import copy
from engine.board import Board
from engine.move import Move
from engine.position import Position
from engine.piece import PieceColor, PieceType
from engine.zobrist import Zobrist
from enum import Enum

class Outcome(Enum):
    DARK = PieceColor.DARK
    LIGHT = PieceColor.LIGHT
    DRAW = 2
    NOT_FINISHED = 3

class OnlyKingsType(Enum):
    UNRELEVANT = 0
    T1VS1 = 1
    T2VS1 = 2
    TNVS1 = 3

class Game:
    """
    By default board is placed in a way that white pieces are at the bottom.
    Currently all methods assume board is in the default mode as described up above.
    As in the rules piece is promoted only after making full move.
    """
    def __init__(self, variant: str = "international") -> None:
        if variant == "international":
            self.board = Board(10)
            self.board.place_pieces(4)
        self.moves_history: list[Move] = []
        self.whose_turn: PieceColor = PieceColor.LIGHT
        self.only_kings_move_counter = -1
        self.only_kings_type = OnlyKingsType.UNRELEVANT
        self.no_progress_counter = 0
        self.outcome: Outcome = Outcome.NOT_FINISHED
        self.zobrist: Zobrist = Zobrist()
        self.zobrist.init_hash(self.board)
        self.position_counts = {self.get_position_key(): 1}

    def _one_field_move(self, position, new_position) -> Move | None:
        if self.board.is_in_board(new_position) and self.board.is_field_free(new_position):
            move = Move()
            move.add_position(position)
            move.add_position(new_position)
            return move
        return None

    def _is_opponent_at_field(self, position: Position, my_color: PieceColor) -> bool:
        if not self.board.is_in_board(position):
            return False
        piece = self.board.get_field_piece(position)
        if not piece:
            return False
        return piece.color != my_color

    def _was_piece_already_taken(self, position: Position, move: Move) -> bool:
        if move.empty():
            return False
        for i in range(len(move.positions) - 1):
            start = move.positions[i]
            end = move.positions[i+1]

            dx = 1 if end.x > start.x else - 1
            dy = 1 if end.y > start.y else - 1

            x, y = start.x + dx, start.y + dy

            while (x,y) != (end.x, end.y):
                if position.x == x and position.y == y:
                    return True
                x += dx
                y += dy
        return False

    # in python lists are passed by reference but reasignment doesn't change original
    def _add_one_capture(self, potential_moves: list[Move], current_position: Position, current_move: Move, flying: bool = False) -> None:
        if flying:
            there_is_next = False
            for dx, dy in zip((1, -1, -1, 1), (1, 1, -1, -1)):
                x, y = current_position.x, current_position.y
                x += dx
                y += dy
                while self.board.is_in_board(Position(x,y)) and self.board.is_field_free(Position(x,y)):
                    x += dx
                    y += dy
                if not self.board.is_in_board(Position(x,y)):
                    continue
                if not self._is_opponent_at_field(Position(x,y), self.whose_turn) or self._was_piece_already_taken(Position(x,y), copy.deepcopy(current_move)):
                    continue
                x += dx
                y += dy
                there_is_next = True
                while self.board.is_in_board(Position(x,y)) and self.board.is_field_free(Position(x,y)):
                    move = copy.deepcopy(current_move)
                    next_position = Position(x,y)
                    move.add_position(next_position)
                    self._add_one_capture(potential_moves, next_position, move, flying=True)
                    x += dx
                    y += dy

            if not there_is_next and not current_move.empty():
                potential_moves.append(current_move)

        else:
            there_is_next = False
            for dx, dy in zip((1, -1, -1, 1), (1, 1, -1, -1)):
                neighbour_position = Position(current_position.x+dx, current_position.y+dy)
                next_neighbour_position = Position(neighbour_position.x+dx, neighbour_position.y+dy)
                if self.board.is_in_board(neighbour_position) and self.board.is_in_board(next_neighbour_position) and self._is_opponent_at_field(neighbour_position, self.whose_turn) and self.board.is_field_free(next_neighbour_position) and not self._was_piece_already_taken(neighbour_position, copy.deepcopy(current_move)):
                    there_is_next = True
                    move = copy.deepcopy(current_move)
                    move.add_position(next_neighbour_position)
                    self._add_one_capture(potential_moves, next_neighbour_position, move)

            if not there_is_next and not current_move.empty():
                potential_moves.append(current_move)

    def evaluate(self, global_maximizer: PieceColor) -> float:
        if not self.is_in_progress():
            if global_maximizer == PieceColor.LIGHT:
                if self.final_outcome() == Outcome.LIGHT:
                    return 1000
                if self.final_outcome() == Outcome.DARK:
                    return -1000
                return -0.2
            if self.final_outcome() == Outcome.DARK:
                return 1000
            if self.final_outcome() == Outcome.LIGHT:
                return -1000
            return -0.2
        
        score = 0
        for pos in self._all_pieces_positions(global_maximizer):
            if self.board.fields[pos.y][pos.x].piece.type == PieceType.MAN:
                score += 3
            else:
                score += 5
        minimizer = (PieceColor.DARK if global_maximizer == PieceColor.LIGHT else PieceColor.LIGHT)

        for pos in self._all_pieces_positions(minimizer):
            if self.board.fields[pos.y][pos.x].piece.type == PieceType.MAN:
                score += -3
            else:
                score += -5
        return score

    def _all_pieces_positions(self, color: PieceColor | None = None) -> list[Position]:
        if not color:
            color = self.whose_turn
        positions: list[Position] = []
        for y in range(self.board.size):
            for x in range(self.board.size):
                piece = self.board.fields[y][x].piece
                if piece is not None and piece.color == color:
                    positions.append(Position(x,y))
        return positions
    
    def is_in_progress(self) -> bool:
        if not self.generate_potential_moves():
            if self.whose_turn == PieceColor.LIGHT:
                self.outcome = Outcome.DARK
            else:
                self.outcome = Outcome.LIGHT
            return False
        
        if self.no_progress_counter > 25:
            self.outcome = Outcome.DRAW
            return False
        
        if any(value >= 3 for value in self.position_counts.values()):
            self.outcome = Outcome.DRAW
            return False

        only_kings = True
        n_light_kings = 0
        n_dark_kings = 0
        for position in self._all_pieces_positions(PieceColor.LIGHT) + self._all_pieces_positions(PieceColor.DARK):
            if self.board.fields[position.y][position.x].piece.type == PieceType.MAN:
                only_kings = False
                break
            else:
                if self.board.fields[position.y][position.x].piece.color == PieceColor.LIGHT:
                    n_light_kings += 1
                else:
                    n_dark_kings += 1
        current_only_kings_type = self.only_kings_type
        if only_kings:
            if n_dark_kings == 1 and n_light_kings == 1:
                self.outcome = Outcome.DRAW
                return False
            self.only_kings_move_counter += 1

            # alternativelly sorted([n_light_kings, n_dark_kings]) == [1,2]
            if set([n_light_kings, n_dark_kings]) == set([2,1]):
                self.only_kings_type = OnlyKingsType.T2VS1
                if self.only_kings_move_counter > 16:
                    self.outcome = Outcome.DRAW
                    return False
            elif n_dark_kings == 1 or n_light_kings == 1:
                self.only_kings_type = OnlyKingsType.TNVS1
                if self.only_kings_move_counter > 32:
                    self.outcome = Outcome.DRAW
                    return False
                
            if current_only_kings_type not in [OnlyKingsType.UNRELEVANT, self.only_kings_type]:
                self.only_kings_move_counter = 0
                
        return True

    def final_outcome(self) -> Outcome:
        """
        Loss/Win:
            You lose if you have no legal moves or you don't have any pieces left but it simplifies to the first condition.
        Draw:
            Draw is when the same position (board arrangement + whose turn) appears 3 times.
            No progress is made: during 25 consecutive moves there was no capture and man-type piece hasn't moved.
            There is only 1 King vs 1 King.
            If there is 2 Kings vs 1 King then max 16 moves.
            If there is 3+ Kings vs 1 King then max 32 moves.
            Moving from one position of type only_kings to the other resets counter and updates upper limit.

        All this is managed in the is_in_progress() method.
        """
        return self.outcome
            
    def get_position_key(self) -> int:
        return self.zobrist.hash

    def generate_potential_moves(self) -> list[Move]:
        """
        Returns all moves that maximize capture rate for all pieces of the current player.
        If there is no possible moves [] is returned.
        """
        potential_moves: list[Move] = []

        for position in self._all_pieces_positions():
            moves = self._generate_potential_moves_for_position(position)
            if moves:
                potential_moves.extend(moves)
        if len(potential_moves) == 0:
            return []
        return self._remove_not_maximum_moves(potential_moves)


    def _generate_potential_moves_for_position(self, position: Position) -> list[Move]:
        """
        Moves that are returned are only the ones that maximize the number of captures because it is stated in the rules of checkers.
        There is also need to generate potential moves for each piece and then keep only the one that maximize the number of captures.
        If passed position is occupied not by the piece that belongs to the player whose turn is then None is returned.
        If there are no possible moves from passed position then empty list is returned.
        """
        if self.whose_turn != self.board.fields[position.y][position.x].piece.color:
            return []
        potential_moves: list[Move] = []
        if self.board.fields[position.y][position.x].piece.type == PieceType.MAN:
            if self.whose_turn == PieceColor.LIGHT:
                # one field plain moves
                for dx in [-1, 1]:
                    new_position = Position(position.x+dx, position.y+1)
                    if move := self._one_field_move(position, new_position):
                        potential_moves.append(move)

                # capture moves found recurrently
                move = Move()
                move.add_position(position)
                self._add_one_capture(potential_moves, position, move)
            else:
                # one field plain moves
                for dx in [-1, 1]:
                    new_position = Position(position.x+dx, position.y-1)
                    if move := self._one_field_move(position, new_position):
                        potential_moves.append(move)

                # capture moves found recurrently
                move = Move()
                move.add_position(position)
                self._add_one_capture(potential_moves, position, move)

        elif self.board.fields[position.y][position.x].piece.type == PieceType.KING:

            # plain moves
            for dx, dy in zip((1, -1, -1, 1), (1, 1, -1, -1)):
                x, y = position.x, position.y
                x += dx
                y += dy
                while self.board.is_in_board(Position(x,y)) and self.board.is_field_free(Position(x,y)):
                    move = Move()
                    move.add_position(position)
                    move.add_position(Position(x,y))
                    potential_moves.append(move)
                    x += dx
                    y += dy
            # capture moves
            move = Move()
            move.add_position(position)
            self._add_one_capture(potential_moves, position, move, flying=True)

        potential_moves = self._remove_not_maximum_moves(potential_moves)
        return potential_moves
    
    def _are_positions_neighbour(self, position0: Position, position1: Position) -> bool:
        if abs(position0.x - position1.x) == 1 and abs(position0.y - position1.y) == 1:
            return True
        return False
    
    def _count_n_captures(self, move: Move) -> int:
        if self._are_positions_neighbour(move.positions[0], move.positions[0]):
            return 0
        captures = 0
        for i in range(len(move.positions) - 1):
            start = move.positions[i]
            end = move.positions[i+1]

            dx = 1 if end.x > start.x else - 1
            dy = 1 if end.y > start.y else - 1

            x, y = start.x + dx, start.y + dy

            while (x,y) != (end.x, end.y):
                if not self.board.is_field_free(Position(x,y)):
                    captures += 1
                    break
                x += dx
                y += dy
        return captures

    def _remove_not_maximum_moves(self, potential_moves: list[Move]) -> list[Move]:
        max_idxs = []
        max_captures = 0
        for i, move in enumerate(potential_moves):
            n_captures = self._count_n_captures(move)
            if n_captures > max_captures:
                max_idxs = []
                max_idxs.append(i)
                max_captures = n_captures
            elif n_captures == max_captures:
                max_idxs.append(i)
        max_moves: list[Move] = []
        for i in max_idxs:
            max_moves.append(potential_moves[i])
        return max_moves

    def make_move(self, move: Move) -> None:
        """
        If piece got to the back rank of the oponent it is promoted.
        Pieces captured during move are taken from the board.
        Piece is taken to move's final position.
        """
        start_position = move.positions[0]
        end_position = move.positions[-1]
        piece = self.board.get_field_piece(start_position)
        # no progress counter reset
        if piece.type == PieceType.MAN or self._count_n_captures(move) > 0:
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1

        # promote MAN to KING
        for pos in move.positions[1:]:
            if pos.y == self.board.size - 1 and piece.color == PieceColor.LIGHT and piece.type == PieceType.MAN:
                piece.type = PieceType.KING
                break
            if pos.y == 0 and piece.color == PieceColor.DARK and piece.type == PieceType.MAN:
                piece.type = PieceType.KING
                break
        # remove all pieces that are taken between positions
        for i in range(len(move.positions) - 1):
            start = move.positions[i]
            end = move.positions[i+1]

            dx = 1 if end.x > start.x else - 1
            dy = 1 if end.y > start.y else - 1

            x, y = start.x + dx, start.y + dy

            while (x,y) != (end.x, end.y):
                position = Position(x, y)
                piece = self.board.get_field_piece(position)
                if piece is not None and piece.color != self.whose_turn:
                    self.board.change_field_piece(position)
                    self.zobrist.update_hash(position, piece.color, piece.type)
                x += dx
                y += dy
        # move piece on the board
        piece = self.board.get_field_piece(start_position)
        self.board.change_field_piece(start_position)
        self.zobrist.update_hash(start_position, piece.color, piece.type)
        self.board.change_field_piece(end_position, piece)
        piece = self.board.get_field_piece(end_position)
        self.zobrist.update_hash(end_position, piece.color, piece.type)
        self.moves_history.append(move)
        self.whose_turn = PieceColor.LIGHT if self.whose_turn == PieceColor.DARK else PieceColor.DARK

        # update zobrist hash - whose turn part
        self.zobrist.apply_side_hash()

        # same position (board arrangement + whose turn) counter increment
        key = self.get_position_key()
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
