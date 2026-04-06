

from engine.field import Field, FieldColor
from engine.piece import Piece, PieceColor, PieceType
from engine.position import Position

class Board:
    def __init__(self, size: int) -> None:
        self.size = size
        self.fields: list[list[Field]] = []
        self._prepare_board()
    def _prepare_board(self) -> None:
        for row in range(self.size):
            current_row = []
            for col in range(self.size):
                if (row + col) % 2 == 0:
                    color = FieldColor.DARK
                else:
                    color = FieldColor.LIGHT
                current_row.append(Field(color))
            self.fields.append(current_row)

    def place_pieces(self, n_rows: int, light_bottom: bool = True) -> None:
        # bottom rows
        for row in range(n_rows):
            for col in range(self.size):
                if self.fields[row][col].color == FieldColor.DARK:
                    self.fields[row][col].piece = Piece(
                        PieceColor.LIGHT if light_bottom else PieceColor.DARK,
                        PieceType.MAN
                        )

        # top rows
        for row in range(self.size - n_rows, self.size):
            for col in range(self.size):
                if self.fields[row][col].color == FieldColor.DARK:
                    self.fields[row][col].piece = Piece(
                        PieceColor.DARK if light_bottom else PieceColor.LIGHT,
                        PieceType.MAN
                        )
    def is_in_board(self, position: Position) -> bool:
        return 0 <= position.x < self.size and 0 <= position.y < self.size

    def is_field_free(self, position: Position) -> bool:
        return not bool(self.fields[position.y][position.x].piece)

    def get_field_piece(self, position: Position) -> Piece | None:
        return self.fields[position.y][position.x].piece

    def change_field_piece(self, position: Position, piece: Piece | None = None) -> None:
        self.fields[position.y][position.x].piece = piece

    def print(self) -> None:
        for row in reversed(self.fields):
            print("|", end="")
            for field in row:
                if not field.piece:
                    print("_|", end="")
                else:
                    if field.piece.color == PieceColor.LIGHT:
                        if field.piece.type == PieceType.KING:
                            print("L|", end="")
                        else:
                            print("l|", end="")
                    else:
                        if field.piece.type == PieceType.KING:
                            print("D|", end="")
                        else:
                            print("d|", end="")
            print("")
