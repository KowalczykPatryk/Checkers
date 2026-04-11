# Checkers

## Abstract objects:
### Board - represents checkers board and contains Fields
### Piece - represents all pieces with different colors and types
### Field - can contain Piece or None if there is no piece
### Position - has x and y value that represents position on the board
### Move - contains all positions that are taken during move (more than 2 in multi-capture move)
### Game - contains all logic behind the game:
- calculates all possible moves in current state
- calculates moves that are maximal in the sense of number of captures
- returns draw if no progress is made: during 25 consecutive moves there was no capture and man-type piece hasn't moved
- only kings ending counter management
- keeps history of the moves
- tracks whose turn is now
- checks whether game has already ended
- keeps zobrist hash of the current state of the board
- counts occurences of the same board state (must be less than 3) using zobrist algorith
- ensured after making move everything is updated
- returns evaluation (score) of current state of the board for desired side
- returns final outcome of the game (who is winner or draw)

## Algorithms:
### Zobrist:  
During initialization selects 3D tensor of random ints (for all fields every possible piece). Ints are from range $[0, 2^{64}]$ so that possibility of same value crash is low.
If each combination of position and piece yields different number (in binary representation at least one bit is different) then taking XOR of this numbers outputs unique hash for each state of the board. On binary level XOR works like addition with modulo 2. So XOR of the same number return 0. This property enables us to take piece from the position in constant time by adding it to the board the second time. Unique state also includes whose turn is now. This is managed by having unique number for this property and using XOR as described above.

### Minimax:
By using recursion and depth decrement tree of possible game states in explored. Above game engine is used so that only possible moves are calculated. As depth decreses maximizer and minimizer take turns (maximizer selects move that comes with the most positive score whereas minimizer does the opposite). When depth has decresed to 0 or game has ended then evaluation of board is returned. During unwinding the call stack function execution is resumed, best score from call is saved and another branch is explored.

### Alpha-Beta Pruning:
- Alpha is the best value that the maximizer currently can guarantee at that level or above.
- Beta is the best value (most negative) that the minimizer currently can guarantee at that level or below.

Pruning condition is: if at any point beta <= alpha, we can prune the remaining branches.  
Alpha and beta are passed from parent to children in the tree. And in this tree MAX and MIN take turns when another recursion call. So for example MIN parent
found so far pretty negative beta value that were returned from other children but these children were maximizing so if maximizing child found so far sth bigger than minimazing parent passed down in beta variable then maximizing child can stop further searching. It is because maximizing child cannot take sth less than he found already and what he found is already bigger than other maximizing child so it won't be selected by minimizing parent.
