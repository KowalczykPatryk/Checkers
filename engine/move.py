
from engine.position import Position

class Move:
    def __init__(self) -> None:
        self.positions: list[Position] = []
        # move should also store positions of captured pieces with types of captured pieces and also whether after move there was promotion
        # previous state?


    def add_position(self, position: Position) -> None:
        self.positions.append(position)

    def empty(self) -> bool:
        return len(self.positions) == 0 or len(self.positions) == 1
