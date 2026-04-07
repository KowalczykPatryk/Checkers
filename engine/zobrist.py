
import random
from engine.piece import PieceColor, PieceType
from engine.board import Board
from engine.position import Position

class Zobrist:
    def __init__(self, size: tuple[int, int, int] = (10, 10, 4), min_v: int = 0, max_v: int = pow(2, 64)) -> None:
        """
        size:
            0 - x size of the board
            1 - y size of the board
            2 - the number of possible pieces type (color of the piece, type of the piece)
        min_v:
            Minimal value of the random number inside zobrist table
        max_v:
            Maximal value of the random number inside zobrist table
        """
        self.min_v: int = min_v
        self.max_v: int = max_v
        self.size: tuple[int, int, int] = size
        self.table: list[list[list[int]]]
        self.hash: int
        self.side_hash: int

        self.init_table()

    def random_int(self, min_v: int, max_v: int) -> int:
        return random.randint(min_v, max_v)

    def index_of(self, piece_color: PieceColor, piece_type: PieceType) -> int:
        match piece_color:
            case PieceColor.LIGHT:
                if piece_type == PieceType.MAN:
                    return 0
                return 1
            case PieceColor.DARK:
                if piece_type == PieceType.MAN:
                    return 2
                return 3

    def init_table(self) -> None:
        self.table =  [[[self.random_int(self.min_v, self.max_v) for k in range(self.size[2])] for j in range(self.size[1])] for i in range(self.size[0])]

    def init_hash(self, board: Board) -> None:
        h = 0
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if board.fields[i][j].piece is not None:
                    h ^= self.table[i][j][self.index_of(board.fields[i][j].piece.color, board.fields[i][j].piece.type)]

        self.hash = h
        self.side_hash = self.random_int(self.min_v, self.max_v)

    def update_hash(self, position: Position, piece_color: PieceColor, piece_type: PieceType) -> None:
        """
        To remove piece from hash it has to be added again because XOR is used under the hood
        """
        self.hash ^= self.table[position.x][position.y][self.index_of(piece_color, piece_type)]
    
    def apply_side_hash(self) -> None:
        self.hash ^= self.side_hash

