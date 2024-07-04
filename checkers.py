import copy

cache = {}
POS_INF = 99999
NEG_INF = -99999
LEFT = -1
RIGHT = 1
UP = -1
DOWN = 1
CUTOFF = 6
WEIGHT_MATRIX = [
    [5, 4, 3, 3, 3, 3, 4, 5],
    [4, 3, 2, 2, 2, 2, 3, 4],
    [3, 2, 6, 6, 6, 6, 2, 3],
    [3, 2, 6, 8, 8, 6, 2, 3],
    [3, 2, 6, 8, 8, 6, 2, 3],
    [3, 2, 6, 6, 6, 6, 2, 3],
    [4, 3, 2, 2, 2, 2, 3, 4],
    [5, 4, 3, 3, 3, 3, 4, 5]
]


class State:
    def __hash__(self):
        """Class representing the state of the game."""
        return hash(str(self.board))

    def __eq__(self, other):
        """Checks equality based on the board configuration."""
        return self.board == other.board

    def __init__(self, board):
        """Initializes the state with a given board configuration."""
        self.board = board

        self.width = 8
        self.height = 8

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
            search = ('r', 'R')
            direction = UP
        else:
            search = ('b', 'B')
            direction = DOWN

        for i in range(8):
            for j in range(8):
                curr = self.board[i][j]
                if curr in search:
                    if curr == search[1]:
                        if self.left_jump(-direction, j, i) or self.right_jump(-direction, j, i):
                            return True

                    if self.left_jump(direction, j, i) or self.right_jump(direction, j, i):
                        return True

        return False


def terminal(s: State, curr_turn: str) -> bool:
    """Checks if the game is in a terminal state.

    Args:
        s: The current state.
        curr_turn: Current turn ('r' or 'b').

    Returns:
        True if the game is in a terminal state, False otherwise.
    """
    r = 0
    b = 0
    for i in range(8):
        for j in range(8):
            if s.board[i][j] == 'r' or s.board[i][j] == 'R':
                r += 1
            elif s.board[i][j] == 'b' or s.board[i][j] == 'B':
                b += 1

    if r == 0 or b == 0:
        return True

    if get_successors(s, curr_turn):
        return False

    return True


def utility(s: State, player: str, depth: int) -> float:
    """Calculates the utility of a terminal state.

    Args:
        s: The current state.
        player: The player ('r' or 'b').
        depth: The current depth.

    Returns:
        The utility value of the state.
    """
    opp = 0
    ply = 0
    for i in range(8):
        for j in range(8):
            if s.board[i][j] in get_opp_char(player):
                opp += 1

            if s.board[i][j] in get_opp_char(get_next_turn(player)):
                ply += 1

    if opp == 0:
        return POS_INF - depth
    if ply == 0:
        return NEG_INF + depth

    if get_successors(s, player):
        return POS_INF - depth
    else:
        return NEG_INF + depth


def eval_func(s: State, player: str) -> float:
    """Evaluates a state using a heuristic function.

    Args:
        s: The current state.
        player: The player ('r' or 'b').

    Returns:
        The heuristic value of the state.
    """
    opp = 0
    ply = 0
    loc_opp = 0
    loc_ply = 0
    for i in range(8):
        for j in range(8):
            if s.board[i][j] == get_opp_char(player)[0]:
                opp += 1
                loc_opp += WEIGHT_MATRIX[i][j]

            if s.board[i][j] == get_opp_char(player)[1]:
                opp += 3
                loc_opp += WEIGHT_MATRIX[i][j] * 2

            if s.board[i][j] == get_opp_char(get_next_turn(player))[0]:
                ply += 1
                loc_ply += WEIGHT_MATRIX[i][j]

            if s.board[i][j] == get_opp_char(get_next_turn(player))[1]:
                ply += 3
                loc_ply += WEIGHT_MATRIX[i][j] * 2

    return ply - opp + (loc_opp - loc_ply) * 0.1


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
        search = ('r', 'R')
        direction = UP
    else:
        search = ('b', 'B')
        direction = DOWN

    if s.jump_available(curr_turn):
        for i in range(8):
            for j in range(8):
                if s.board[i][j] in search:
                    states.extend(get_jumps(s, j, i, direction))

        return states

    for i in range(8):
        for j in range(8):
            if s.board[i][j] not in search:
                continue

            if s.board[i][j] == search[1]:
                if s.left_move(-direction, j, i):
                    states.append(move(s, j, i, LEFT, -direction))
                if s.right_move(-direction, j, i):
                    states.append(move(s, j, i, RIGHT, -direction))

            if s.left_move(direction, j, i):
                states.append(move(s, j, i, LEFT, direction))

            if s.right_move(direction, j, i):
                states.append(move(s, j, i, RIGHT, direction))

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
    promote(new, x + hor, y + ver)
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
    return new


def promote(s: State, x: int, y: int) -> bool:
    """Promotes a piece to a king if it reaches the opposite end.

    Args:
        s: The current state.
        x: X-coordinate.
        y: Y-coordinate.

    Returns:
        True if the piece is promoted, False otherwise.
    """
    if s.board[y][x] == 'r' and y == 0:
        s.board[y][x] = 'R'
        return True

    if s.board[y][x] == 'b' and y == 7:
        s.board[y][x] = 'B'
        return True

    return False


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
        if promote(temp[0], temp[1], temp[2]):
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


def min_move(s: State, alpha: float, beta: float, depth: int, curr: str, n: str) -> (float, State):
    """Minimizing player's move in the minimax algorithm with alpha-beta pruning.

    Args:
        s: The current state.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
        depth: Current depth in the tree.
        curr: Current turn ('r' or 'b').
        n: Minimizing player.

    Returns:
        The evaluation value and the best move.
    """
    if s in cache and cache[s]['d'] <= depth and cache[s]['ply'] == curr:
        return cache[s]['v'], cache[s]['move']

    if terminal(s, curr):
        return utility(s, n, depth), None

    if depth >= CUTOFF:
        return eval_func(s, n), None

    v = POS_INF * 10
    best_move = None
    moves = get_successors(s, curr)
    moves = sorted(moves, key=lambda m: eval_func(s, n))
    for successor in moves:
        ev, _ = max_move(successor, alpha, beta, depth + 1, get_next_turn(curr), n)
        if ev < v:
            v, best_move = ev, successor
        beta = min(beta, ev)
        if beta <= alpha:
            return v, best_move

    cache[s] = {'d': depth, 'ply': curr, 'v': v, 'move': best_move}
    return v, best_move


def max_move(s: State, alpha: float, beta: float, depth: int, curr: str, n: str) -> (float, State):
    """Maximizing player's move in the minimax algorithm with alpha-beta pruning.

    Args:
        s: The current state.
        alpha: Alpha value for pruning.
        beta: Beta value for pruning.
        depth: Current depth in the tree.
        curr: Current turn ('r' or 'b').
        n: Maximizing player.

    Returns:
        The evaluation value and the best move.
    """
    if s in cache and cache[s]['d'] <= depth and cache[s]['ply'] == curr:
        return cache[s]['v'], cache[s]['move']

    if terminal(s, curr):
        return utility(s, n, depth), None

    if depth >= CUTOFF:
        return eval_func(s, n), None

    v = NEG_INF * 10
    best_move = None
    moves = get_successors(s, curr)
    moves = sorted(moves, key=lambda m: eval_func(s, n), reverse=True)
    for successor in moves:
        ev, _ = min_move(successor, alpha, beta, depth + 1, get_next_turn(curr), n)
        if ev > v:
            v, best_move = ev, successor
        alpha = max(alpha, ev)
        if beta <= alpha:
            return v, best_move

    cache[s] = {'d': depth, 'ply': curr, 'v': v, 'move': best_move}
    return v, best_move


def minimax(s: State, maximizing_player: str) -> State:
    """Runs the minimax algorithm to find the best move for the maximizing player.

    Args:
        s: The current state.
        maximizing_player: The maximizing player ('r' or 'b').

    Returns:
        The best move state.
    """
    return max_move(s, NEG_INF, POS_INF, 0, maximizing_player, maximizing_player)[1]
