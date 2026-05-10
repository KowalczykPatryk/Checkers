
from enum import Enum

class PieceColor(Enum):
    DARK = 0
    LIGHT = 1

class PieceType(Enum):
    MAN = 0
    KING = 1

class Piece:
    def __init__(self, color: PieceColor, piece_type: PieceType):
        self.color: PieceColor = color
        self.type: PieceType = piece_type
