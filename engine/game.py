import copy
from engine.board import Board
from engine.move import Move
from engine.position import Position
from engine.piece import PieceColor, PieceType

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

    def _one_field_move(self, position, new_position) -> Move | None:
        if self.board.is_field_free(new_position) and self.board.is_in_board(new_position):
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
                if self._is_opponent_at_field(neighbour_position, self.whose_turn) and self.board.is_field_free(next_neighbour_position) and not self._was_piece_already_taken(neighbour_position, copy.deepcopy(current_move)):
                    there_is_next = True
                    move = copy.deepcopy(current_move)
                    move.add_position(next_neighbour_position)
                    self._add_one_capture(potential_moves, next_neighbour_position, move)

            if not there_is_next and not current_move.empty():
                potential_moves.append(current_move)

    def _all_pieces_positions(self) -> list[Position]:
        positions: list[Position] = []
        for y in range(self.board.size):
            for x in range(self.board.size):
                if self.board.fields[y][x].color == self.whose_turn:
                    positions.append(Position(x,y))
        return positions

    def generate_potential_moves(self) -> list[Move] | None:
        """
        Returns all moves that maximize capture rate for all pieces of the current player.
        If there is no possible moves None is returned.
        """
        potential_moves: list[Move] = []

        for position in self._all_pieces_positions():
            moves = self._generate_potential_moves_for_position(position)
            if moves:
                potential_moves.extend(moves)
        if len(potential_moves) == 0:
            return None
        return self._remove_not_maximum_moves(potential_moves)


    def _generate_potential_moves_for_position(self, position: Position) -> list[Move] | None:
        """
        Moves that are returned are only the ones that maximize the number of captures because it is stated in the rules of checkers.
        There is also need to generate potential moves for each piece and then keep only the one that maximize the number of captures.
        If passed position is occupied not by the piece that belongs to the player whose turn is then None is returned.
        If there are no possible moves from passed position then empty list is returned.
        """
        if self.whose_turn != self.board.fields[position.y][position.x].piece.color:
            return None
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
        # promote MAN to KING
        for mov in move.positions[1:]:
            if mov.y == self.board.size - 1 and piece.color == PieceColor.LIGHT and piece.type == PieceType.MAN:
                piece.type = PieceType.KING
                break
            if mov.y == 0 and piece.color == PieceColor.DARK and piece.type == PieceType.MAN:
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
                self.board.change_field_piece(Position(x,y))
                x += dx
                y += dy
        # move piece on the board
        self.board.change_field_piece(start_position)
        self.board.change_field_piece(end_position, piece)
        self.moves_history.append(move)
        self.whose_turn = PieceColor.LIGHT if self.whose_turn == PieceColor.DARK else PieceColor.DARK
