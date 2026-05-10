

from ai.engine.field import Field, FieldColor
from ai.engine.piece import Piece, PieceColor, PieceType
from ai.engine.position import Position

class Board:
    """
    Left bottom field MUST be dark as it is stated in the rules.
    However on some online boards it is the opposite.
    """
    def __init__(self, size: int) -> None:
        self.size = size
        self.fields: list[list[Field]] = []
        self._prepare_board()
    def _prepare_board(self) -> None:
        for row in range(self.size):
            current_row = []
            for col in range(self.size):
                if (row + col) % 2 == 0:
                    color = FieldColor.LIGHT
                else:
                    color = FieldColor.DARK
                current_row.append(Field(color))
            self.fields.append(current_row)

    def place_pieces(self, n_rows: int, light_bottom: bool = True) -> None:
        """
        n_rows - states in how many rows pieces will be placed
        example: in 10x10 board in international checkers there are 4 rows of pieces
        for each player
        light_bottom - whether light pieces should be placed at the bottom of the board or
        at the top. It doesn't change logic but might help with visualisation.
        """
        # top rows
        for row in range(n_rows):
            for col in range(self.size):
                if self.fields[row][col].color == FieldColor.DARK:
                    self.fields[row][col].piece = Piece(
                        PieceColor.LIGHT if light_bottom else PieceColor.DARK,
                        PieceType.MAN
                        )

        # bottom rows
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
        """
        Checks whether there is already a piece in the provided position
        on the board.
        """
        return not bool(self.fields[position.y][position.x].piece)

    def get_field_piece(self, position: Position) -> Piece | None:
        return self.fields[position.y][position.x].piece

    def change_field_piece(self, position: Position, piece: Piece | None = None) -> None:
        self.fields[position.y][position.x].piece = piece

    def print(self, upper_player_name: str, lower_player_name: str) -> None:
        """
        Prints to the terminal what is currently on the board.
        """
        for i, row in enumerate(self.fields):
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
            
            if i == 0:
                print(f" {upper_player_name}", end="")
            if i == len(self.fields) - 1:
                print(f" {lower_player_name}", end="")
            print("")

    def to_string(self, upper_player_name: str, lower_player_name: str) -> str:
        string = ""
        for i, row in enumerate(self.fields):
            string += "|"
            for field in row:
                if not field.piece:
                    string += "_|"
                else:
                    if field.piece.color == PieceColor.LIGHT:
                        if field.piece.type == PieceType.KING:
                            string += "L|"
                        else:
                            string += "l|"
                    else:
                        if field.piece.type == PieceType.KING:
                            string += "D|"
                        else:
                            string += "d|"

            if i == 0:
                string += f" {upper_player_name}"
            if i == len(self.fields) - 1:
                string += f" {lower_player_name}"
            string += "\n"
        return string
