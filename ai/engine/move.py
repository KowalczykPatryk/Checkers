
from ai.engine.position import Position

class Move:
    def __init__(self) -> None:
        self.positions: list[Position] = []

    def __eq__(self, other) -> bool:
        for p1, p2 in zip(self.positions, other.positions):
            if p1 != p2:
                return False
        return True

    def add_position(self, position: Position) -> None:
        self.positions.append(position)

    def empty(self) -> bool:
        return len(self.positions) == 0 or len(self.positions) == 1
