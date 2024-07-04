import pygame
import sys
import checkers

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
RED = (225, 55, 55)
GRAY = (65, 65, 65)
LIGHT_B = (173, 216, 230)
LIGHT_G = (159, 226, 191)
YELLOW = (255, 253, 141)

IDLE = 0
SELECTED = 1
R_TURN = ('r', 'R')
B_TURN = ('b', 'B')

INITIAL_BOARD = [
    ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
    ['b', '.', 'b', '.', 'b', '.', 'b', '.'],
    ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.'],
    ['r', '.', 'r', '.', 'r', '.', 'r', '.'],
    ['.', 'r', '.', 'r', '.', 'r', '.', 'r'],
    ['r', '.', 'r', '.', 'r', '.', 'r', '.']
]


def get_turn_pieces(t: str) -> (str, str):
    """
    Get the pieces corresponding to the current turn.

    Args:
        t (str): The current turn ('r' or 'b').

    Returns:
        tuple: A tuple containing the piece and king piece for the current turn.
    """
    if t == 'r':
        return 'r', 'R'
    else:
        return 'b', 'B'


def draw_grid(s: checkers.State, highlights: list[(int, int)], select: (int, int), dif: list[(int, int)]) -> None:
    """
    Draw the checkers grid on the screen.

    Args:
        s (checkers.State): The current game state.
        highlights (list[(int, int)]): List of positions to highlight.
        select ((int, int)): The currently selected position.
        dif (list[(int, int)]): Changes from the previous move if the move was made by the bot.
    """
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            if (row, col) in highlights:
                pygame.draw.rect(screen, LIGHT_B, rect)
            else:
                pygame.draw.rect(screen, WHITE if (row + col) % 2 == 0 else GRAY, rect)

            if select and (row, col) == select:
                pygame.draw.rect(screen, LIGHT_B, rect)

            if dif and (row, col) in dif:
                pygame.draw.rect(screen, LIGHT_G, rect)

            center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

            if s.board[row][col] == 'R' or s.board[row][col] == 'B':
                pygame.draw.circle(screen, YELLOW, (center_x, center_y), SQUARE_SIZE // 3 + 2)
                pygame.draw.circle(screen, RED if s.board[row][col] == 'R' else BLACK,
                                   (center_x, center_y), SQUARE_SIZE // 3 - 2)

            if s.board[row][col] == 'r':
                pygame.draw.circle(screen, RED, (center_x, center_y), SQUARE_SIZE // 3)
            elif s.board[row][col] == 'b':
                pygame.draw.circle(screen, BLACK, (center_x, center_y), SQUARE_SIZE // 3)


def handle_click(pos: (float, float)) -> (int, int):
    """
    Handle a mouse click and return the corresponding board position.

    Args:
        pos ((float, float)): The position of the mouse click.

    Returns:
        (int, int): The board position corresponding to the click.
    """
    x, y = pos
    row, col = y // SQUARE_SIZE, x // SQUARE_SIZE
    s = (row, col)
    return s


def determine_moves(pos: (int, int), s: checkers.State, t: str) -> list[(int, int)]:
    """
    Determine the possible moves for a given piece.

    Args:
        pos ((int, int)): The current position of the piece.
        s (checkers.State): The current game state.
        t (str): The current turn ('r' or 'b').

    Returns:
        list[(int, int)]: A list of possible moves for the piece.
    """
    moves = []
    if t == 'r':
        direction = checkers.UP
    else:
        direction = checkers.DOWN

    y = pos[0]
    x = pos[1]

    if s.jump_available(t):
        if s.board[y][x] in ('R', 'B'):
            if s.right_jump(-direction, x, y):
                moves.append((y - direction * 2, x + 2))
            if s.left_jump(-direction, x, y):
                moves.append((y - direction * 2, x - 2))

        if s.right_jump(direction, x, y):
            moves.append((y + direction * 2, x + 2))
        if s.left_jump(direction, x, y):
            moves.append((y + direction * 2, x - 2))

        return moves

    if s.board[y][x] in ('R', 'B'):
        if s.right_move(-direction, x, y):
            moves.append((y - direction, x + 1))
        if s.left_move(-direction, x, y):
            moves.append((y - direction, x - 1))

    if s.right_move(direction, x, y):
        moves.append((y + direction, x + 1))
    if s.left_move(direction, x, y):
        moves.append((y + direction, x - 1))

    return moves


def jump_possible(pos: (int, int), s: checkers.State, t: str) -> bool:
    """
    Determine if a piece can perform a jump.

    Args:
        pos ((int, int)): The current position of the piece.
        s (checkers.State): The current game state.
        t (str): The current turn ('r' or 'b').

    Returns:
        bool: True if the piece can a jump, False otherwise.
    """
    if t == 'r':
        direction = checkers.UP
    else:
        direction = checkers.DOWN

    y = pos[0]
    x = pos[1]

    return ((s.board[y][x] in ('R', 'B') and (s.right_jump(-direction, x, y) or s.left_jump(-direction, x, y))) or
            s.right_jump(direction, x, y) or s.left_jump(direction, x, y))


def draw_menu() -> None:
    """
    Draw the game menu.
    """
    screen.fill(BLACK)
    font = pygame.font.Font(None, 74)
    text = font.render("Checkers", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 4))

    font = pygame.font.Font(None, 50)
    text = font.render("Play as:", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 50))

    font = pygame.font.Font(None, 40)
    text = font.render("1. Red", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

    text = font.render("2. Black", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 + 50))

    text = font.render("3. Both", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 + 100))

    pygame.display.flip()


def draw_finish(s: checkers.State) -> None:
    """
    Draw the game finish screen.

    Args:
        s (checkers.State): The final game state.
    """
    screen.fill(BLACK)
    if checkers.utility(s, 'r', 0) == checkers.POS_INF:
        txt = "Red Wins"
    else:
        txt = "Black Wins"
    font = pygame.font.Font(None, 74)
    text = font.render(txt, True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 4))

    font = pygame.font.Font(None, 40)
    text = font.render("1. Restart", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

    text = font.render("2. Quit", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 + 50))\

    pygame.display.flip()


def finish(s: checkers.State) -> bool:
    """
    Handle the game finish state.

    Args:
        s (checkers.State): The final game state.

    Returns:
        bool: True if the player chooses to quit, False if the player chooses to restart.
    """
    while True:
        draw_finish(s)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1:
                    return False
                elif ev.key == pygame.K_2:
                    return True


def menu() -> (bool, bool):
    """
    Display the game menu and get the player's choice.

    Returns:
        (bool, bool): A tuple indicating if the player wants a bot to play for red and black respectively.
    """
    while True:
        draw_menu()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1:
                    return False, True
                elif ev.key == pygame.K_2:
                    return True, False
                elif ev.key == pygame.K_3:
                    return False, False


def run() -> checkers.State:
    """
    Run the main game loop.

    Returns:
        checkers.State: The final game state.
    """
    state = checkers.State(INITIAL_BOARD)
    action = IDLE
    selection = None
    turn = 'r'
    possible_moves = []
    bot_r, bot_b = menu()
    diffs = []

    running = True
    while running and not checkers.terminal(state, turn):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (turn == 'r' and not bot_r) or (turn == 'b' and not bot_b):
                    diffs = []
                    prev_selection = selection
                    selection = handle_click(event.pos)
                    if not selection:
                        continue

                    if action == IDLE and state.board[selection[0]][selection[1]] in get_turn_pieces(turn):
                        action = SELECTED
                        possible_moves = determine_moves(selection, state, turn)
                    elif state.board[selection[0]][selection[1]] in get_turn_pieces(turn):
                        possible_moves = determine_moves(selection, state, turn)
                    elif selection not in possible_moves or selection == prev_selection:
                        selection = None
                        action = IDLE
                        possible_moves = []

                    if action == SELECTED and selection in possible_moves:
                        y_diff = selection[0] - prev_selection[0]
                        x_diff = selection[1] - prev_selection[1]
                        if abs(y_diff) == 2:
                            state = checkers.jump(state, prev_selection[1], prev_selection[0], x_diff // 2, y_diff // 2)
                            status = checkers.promote(state, selection[1], selection[0])
                            if (not jump_possible(selection, state, turn)) or status:
                                turn = checkers.get_next_turn(turn)
                        else:
                            state = checkers.move(state, prev_selection[1], prev_selection[0], x_diff, y_diff)
                            turn = checkers.get_next_turn(turn)

                        action = IDLE
                        selection = None
                        possible_moves = []
                else:
                    prev_state = state
                    state = checkers.minimax(state, turn)
                    diffs = prev_state.diff(state)
                    turn = checkers.get_next_turn(turn)

        draw_grid(state, possible_moves, selection, diffs)
        pygame.display.flip()
    return state


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Checkers')

    curr_state = run()
    while not finish(curr_state):
        curr_state = run()

    pygame.quit()
    sys.exit()
