

from engine.field import Field, FieldColor
from engine.piece import Piece, PieceColor, PieceType
from engine.position import Position

class Board:
    def __init__(self, size: int) -> None:
        self.size = size
        self.fields: list[list[Field]] = []
        self._prepare_board()
    def _prepare_board(self) -> None:
        row0 = []
        row1 = []
        for col in range(self.size):
            if col % 2 == 0:
                row0.append(Field(FieldColor.DARK))
            else:
                row1.append(Field(FieldColor.LIGHT))

        for row in range(self.size):
            if row % 2 == 0:
                self.fields.append(row0)
            else:
                self.fields.append(row1)

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
