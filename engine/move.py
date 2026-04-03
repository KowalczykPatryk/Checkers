
from engine.position import Position

class Move:
    def __init__(self) -> None:
        self.positions: list[Position] = []


    def add_position(self, position: Position) -> None:
        self.positions.append(position)

    def empty(self) -> bool:
        return len(self.positions) == 0 or len(self.positions) == 1
