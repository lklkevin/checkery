from __future__ import annotations

import copy

cache = {}
POS_INF = 99999
NEG_INF = -99999
LEFT = -1
RIGHT = 1
UP = -1
DOWN = 1
CUTOFF = 5
RED = ('r', 'R')
BLACK = ('b', 'B')
KINGS = ('R', 'B')
WEIGHT_MATRIX = [
    [1, 1, 1, 2, 2, 1, 1, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 3, 4, 5, 5, 4, 3, 1],
    [2, 3, 5, 6, 6, 5, 3, 2],
    [2, 3, 5, 6, 6, 5, 3, 2],
    [1, 3, 4, 5, 5, 4, 3, 1],
    [1, 2, 3, 3, 3, 3, 2, 1],
    [1, 1, 1, 2, 2, 1, 1, 1]
]

RED_WM = [[i] * 8 for i in range(7, -1, -1)]
BLK_WM = [[i] * 8 for i in range(0, 8)]

class State:
    def __hash__(self):
        """Class representing the state of the game."""
        return hash(str(self.board))

    def __eq__(self, other):
        """Checks equality based on the board configuration."""
        return self.r == other.r and self.b == other.b

    def __init__(self, board):
        """Initializes the state with a given board configuration."""
        self.board = board
        self.r = set()
        self.b = set()

        for i in range(8):
            for j in range(8):
                if self.board[i][j] in RED:
                    self.r.add((i, j))
                elif self.board[i][j] in BLACK:
                    self.b.add((i, j))

        self.width = 8
        self.height = 8

    def pieces(self) -> int:
        return len(self.b) + len(self.r)

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")

    def num_moves(self, curr_t: str) -> int:
        total = 0
        if curr_t == 'r':
            search = self.r
            direction = UP
        else:
            search = self.b
            direction = DOWN

        for coord in search:
            y = coord[0]
            x = coord[1]

            if self.board[y][x] in KINGS:
                if self.left_move(-direction, x, y):
                    total += 1
                if self.right_move(-direction, x, y):
                    total += 1
                if self.left_jump(-direction, x, y):
                    total += 1
                if self.right_jump(-direction, x, y):
                    total += 1
            if self.left_move(direction, x, y):
                total += 1
            if self.right_move(direction, x, y):
                total += 1
            if self.left_jump(direction, x, y):
                total += 1
            if self.right_jump(direction, x, y):
                total += 1

        return total

    def diff(self, other) -> list[(int, int)]:
        """Finds the differences between the current state and another state.

        Args:
            other: Another state to compare against.

        Returns:
            A list of coordinates where the states differ.
        """
        diffs = []
        a = self.board
        b = other.board
        for r in range(8):
            for c in range(8):
                if a[r][c] != b[r][c]:
                    diffs.append((r, c))

        return diffs

    def left_move(self, direction: int, x: int, y: int) -> bool:
        """Checks if a left move is possible.

        Args:
            direction: Y-direction of the move.
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            True if the left move is possible, False otherwise.
        """
        if 0 <= (y + direction) < 8 and x - 1 >= 0 and self.board[y + direction][x - 1] == '.':
            return True

        return False

    def right_move(self, direction: int, x: int, y: int) -> bool:
        """Checks if a right move is possible.

        Args:
            direction: Y-direction of the move.
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            True if the right move is possible, False otherwise.
        """
        if 0 <= (y + direction) < 8 and x + 1 < 8 and self.board[y + direction][x + 1] == '.':
            return True

        return False

    def left_jump(self, direction: int, x: int, y: int) -> bool:
        """Checks if a left jump is possible.

        Args:
            direction: Y-direction of the jump.
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            True if the left jump is possible, False otherwise.
        """
        if 0 <= (y + direction * 2) < 8:
            if x - 2 >= 0 and self.board[y + direction * 2][x - 2] == '.':
                if self.board[y + direction][x - 1] in get_opp_char(self.board[y][x]):
                    return True

        return False

    def right_jump(self, direction: int, x: int, y: int) -> bool:
        """Checks if a right jump is possible.

        Args:
            direction: Y-direction of the jump.
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            True if the right jump is possible, False otherwise.
        """
        if 0 <= (y + direction * 2) < 8:
            if x + 2 < 8 and self.board[y + direction * 2][x + 2] == '.':
                if self.board[y + direction][x + 1] in get_opp_char(self.board[y][x]):
                    return True

        return False

    def jump_available(self, c_turn: str) -> bool:
        """Checks if a jump is available for the current turn.

        Args:
            c_turn: Current turn ('r' or 'b').

        Returns:
            True if a jump is available, False otherwise.
        """
        if c_turn == 'r':
            search = self.r
            direction = UP
        else:
            search = self.b
            direction = DOWN

        for coord in search:
            y = coord[0]
            x = coord[1]
            if self.board[y][x] in KINGS and (
                    self.left_jump(-direction, x, y) or self.right_jump(-direction, x, y)):
                return True

            if self.left_jump(direction, x, y) or self.right_jump(direction, x, y):
                return True

        return False

    def edit(self, remove: set[(int, int)], add: tuple[int, int]) -> None:
        for r in remove:
            if r in self.r:
                self.r.remove(r)
            else:
                self.b.remove(r)

        if self.board[add[0]][add[1]] in RED:
            self.r.add(add)
        else:
            self.b.add(add)

    def terminal(self, curr_t: str) -> bool:
        if len(self.r) == 0 or len(self.b) == 0:
            return True
        elif self.move_possible(curr_t):
            return False

        return True

    def utility(self, player: str, depth: int) -> float:
        if player == 'r':
            ply = self.r
            opp = self.b
        else:
            ply = self.b
            opp = self.r

        if len(opp) == 0:
            return POS_INF - depth
        elif len(ply) == 0:
            return NEG_INF + depth

        if self.move_possible(player):
            return POS_INF - depth
        else:
            return NEG_INF + depth

    def move_possible(self, player: str) -> bool:
        if self.jump_available(player):
            return True

        if player == 'r':
            search = self.r
            direction = UP
        else:
            search = self.b
            direction = DOWN

        for coord in search:
            y = coord[0]
            x = coord[1]

            if self.board[y][x] in KINGS:
                if self.left_move(-direction, x, y):
                    return True
                if self.right_move(-direction, x, y):
                    return True
            if self.left_move(direction, x, y):
                return True
            if self.right_move(direction, x, y):
                return True

    def eval_func(self, player: str) -> float:
        opps = 0
        plys = 0
        local_opp = 0
        local_ply = 0
        moves_diff = self.num_moves(player) - self.num_moves(get_next_turn(player))

        side_opp = 0
        side_ply = 0

        if player == 'r':
            ply = self.r
            opp = self.b
            p, o = RED, BLACK
            p_wm, o_wm = RED_WM, BLK_WM
        else:
            ply = self.b
            opp = self.r
            p, o = BLACK, RED
            p_wm, o_wm = BLK_WM, RED_WM

        for coord in ply:
            x = coord[1]
            y = coord[0]
            if self.board[y][x] == p[1]:
                plys += 3
            else:
                plys += 1
                side_ply += p_wm[y][x]
            local_ply += WEIGHT_MATRIX[y][x]

        for coord in opp:
            x = coord[1]
            y = coord[0]
            if self.board[y][x] == o[1]:
                opps += 3
            else:
                opps += 1
                side_opp += o_wm[y][x]
            local_opp += WEIGHT_MATRIX[y][x]

        val = 15 * (plys - opps) + 2 * (local_ply - local_opp) + 4 * moves_diff + 3 * (side_ply - side_opp)

        return val

    def promote(self, x: int, y: int) -> bool:
        """Promotes a piece to a king if it reaches the opposite end.

        Args:
            x: X-coordinate.
            y: Y-coordinate.

        Returns:
            True if the piece is promoted, False otherwise.
        """
        if self.board[y][x] == 'r' and y == 0:
            self.board[y][x] = 'R'
            return True

        if self.board[y][x] == 'b' and y == 7:
            self.board[y][x] = 'B'
            return True

        return False


def get_successors(s: State, curr_turn: str) -> list[State]:
    """Generates all possible successor states for the current turn.

    Args:
        s: The current state.
        curr_turn: Current turn ('r' or 'b').

    Returns:
        A list of successor states.
    """
    states = []

    if curr_turn == 'r':
        search = s.r
        direction = UP
    else:
        search = s.b
        direction = DOWN

    if s.jump_available(curr_turn):
        for coord in search:
            states.extend(get_jumps(s, coord[1], coord[0], direction))

        return states

    for coord in search:
        y = coord[0]
        x = coord[1]

        if s.board[y][x] in KINGS:
            if s.left_move(-direction, x, y):
                states.append(move(s, x, y, LEFT, -direction))
            if s.right_move(-direction, x, y):
                states.append(move(s, x, y, RIGHT, -direction))

        if s.left_move(direction, x, y):
            states.append(move(s, x, y, LEFT, direction))
        if s.right_move(direction, x, y):
            states.append(move(s, x, y, RIGHT, direction))

    return states


def move(s: State, x: int, y: int, hor: int, ver: int) -> State:
    """Makes a move and returns the resulting state.

    Args:
        s: The current state.
        x: X-coordinate.
        y: Y-coordinate.
        hor: Horizontal direction of the move.
        ver: Vertical direction of the move.

    Returns:
        The resulting state after the move.
    """
    new = copy.deepcopy(s)
    new.board[y][x], new.board[y + ver][x + hor] = new.board[y + ver][x + hor], new.board[y][x]
    new.promote(x + hor, y + ver)

    new.edit({(y, x)}, (y + ver, x + hor))

    return new


def jump(s: State, x: int, y: int, hor: int, ver: int) -> State:
    """Makes a jump and returns the resulting state.

    Args:
        s: The current state.
        x: X-coordinate.
        y: Y-coordinate.
        hor: Horizontal direction of the jump.
        ver: Vertical direction of the jump.

    Returns:
        The resulting state after the jump.
    """
    new = copy.deepcopy(s)
    new.board[y][x], new.board[y + ver * 2][x + hor * 2] = new.board[y + ver * 2][x + hor * 2], new.board[y][x]
    new.board[y + ver][x + hor] = '.'

    new.edit({(y, x), (y + ver, x + hor)}, (y + ver * 2, x + hor * 2))

    return new


def get_jumps(s: State, x: int, y: int, direction: int) -> list[State]:
    """Generates all possible jump states for a given piece.

    Args:
        s: The current state.
        x: X-coordinate.
        y: Y-coordinate.
        direction: Direction of the jump.

    Returns:
        A list of states resulting from all possible jumps.
    """
    curr = s.board[y][x]
    temps = []
    final = []
    if curr == 'R' or curr == 'B':
        if s.left_jump(-direction, x, y):
            new = jump(s, x, y, LEFT, -direction)
            temps.append((new, x + LEFT * 2, y - direction * 2))

        if s.right_jump(-direction, x, y):
            new = jump(s, x, y, RIGHT, -direction)
            temps.append((new, x + RIGHT * 2, y - direction * 2))

    if s.left_jump(direction, x, y):
        new = jump(s, x, y, LEFT, direction)
        temps.append((new, x + LEFT * 2, y + direction * 2))
    if s.right_jump(direction, x, y):
        new = jump(s, x, y, RIGHT, direction)
        temps.append((new, x + RIGHT * 2, y + direction * 2))

    for temp in temps:
        if temp[0].promote(temp[1], temp[2]):
            final.append(temp[0])
        else:
            further = get_jumps(temp[0], temp[1], temp[2], direction)
            if further:
                final.extend(further)
            else:
                final.append(temp[0])

    return final


def get_opp_char(player: str) -> list[str]:
    """Gets the opponent's characters.

    Args:
        player: The player ('r' or 'b').

    Returns:
        A list of the opponent's characters.
    """
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']


def get_next_turn(curr_turn: str) -> str:
    """Gets the next turn.

    Args:
        curr_turn: Current turn ('r' or 'b').

    Returns:
        The next turn ('r' or 'b').
    """
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'


def min_move(
        s: State, alpha: float, beta: float, depth: int, curr: str, n: str, cutoff: int) -> tuple[float, State | None]:
    """Minimizing player's move in the minimax algorithm with alpha-beta pruning.

    Args:
        s: The current state.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
        depth: Current depth in the tree.
        curr: Current turn ('r' or 'b').
        n: Minimizing player.
        cutoff: Max tree depth.

    Returns:
        The evaluation value and the best move.
    """
    if s in cache and cache[s]['d'] <= depth and cache[s]['ply'] == curr:
        return cache[s]['v'], cache[s]['move']

    if s.terminal(curr):
        return s.utility(n, depth), None

    if depth >= cutoff:
        return s.eval_func(n), None

    v = POS_INF * 10
    best_move = None
    moves = get_successors(s, curr)
    # moves = sorted(moves, key=lambda m: s.eval_func(n), reverse=True)
    for successor in moves:
        ev, _ = max_move(successor, alpha, beta, depth + 1, get_next_turn(curr), n, cutoff)
        if ev < v:
            v, best_move = ev, successor
        beta = min(beta, ev)
        if beta <= alpha:
            return v, best_move

    cache[s] = {'d': depth, 'ply': curr, 'v': v, 'move': best_move}
    return v, best_move


def max_move(
        s: State, alpha: float, beta: float, depth: int, curr: str, n: str, cutoff: int) -> tuple[float, State | None]:
    """Maximizing player's move in the minimax algorithm with alpha-beta pruning.

    Args:
        s: The current state.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
        depth: Current depth in the tree.
        curr: Current turn ('r' or 'b').
        n: Maximizing player.
        cutoff: Max tree depth.

    Returns:
        The evaluation value and the best move.
    """
    if s in cache and cache[s]['d'] <= depth and cache[s]['ply'] == curr:
        return cache[s]['v'], cache[s]['move']

    if s.terminal(curr):
        return s.utility(n, depth), None

    if depth >= cutoff:
        return s.eval_func(n), None

    v = NEG_INF * 10
    best_move = None
    moves = get_successors(s, curr)
    # moves = sorted(moves, key=lambda m: s.eval_func(n))
    for successor in moves:
        ev, _ = min_move(successor, alpha, beta, depth + 1, get_next_turn(curr), n, cutoff)
        if ev > v:
            v, best_move = ev, successor
        alpha = max(alpha, ev)
        if beta <= alpha:
            return v, best_move

    cache[s] = {'d': depth, 'ply': curr, 'v': v, 'move': best_move}
    return v, best_move


def minimax(s: State, maximizing_player: str, cutoff: int) -> State:
    """Runs the minimax algorithm to find the best move for the maximizing player.

    Args:
        s: The current state.
        maximizing_player: The maximizing player ('r' or 'b').
        cutoff: Max depth cutoff.

    Returns:
        The best move state.
    """
    return max_move(s, NEG_INF, POS_INF, 0, maximizing_player, maximizing_player, cutoff)[1]
