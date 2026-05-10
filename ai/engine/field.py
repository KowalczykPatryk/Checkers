from enum import Enum
from ai.engine.piece import Piece

class FieldColor(Enum):
    DARK = 0
    LIGHT = 1

class Field:
    def __init__(self, color: FieldColor = FieldColor.LIGHT, piece: Piece | None = None):
        self.color: FieldColor = color
        self.piece: Piece | None = piece
