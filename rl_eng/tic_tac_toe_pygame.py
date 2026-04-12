import pygame
import sys
import os
from rl_eng.tic_tac_toe import Environment, Agent, CROSS, CIRCLE, EMPTY

# Constants
WIDTH, HEIGHT = 400, 500 # Extra height for buttons/status
LINE_WIDTH = 10
SQUARE_SIZE = WIDTH // 3
CIRCLE_RADIUS, CIRCLE_WIDTH = SQUARE_SIZE // 3, 15
CROSS_WIDTH, SPACE = 25, SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
BUTTON_COLOR = (255, 255, 255)

class TicTacToePygame:
    def __init__(self, run_dir):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('RL-Eng Tic-Tac-Toe')
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.run_dir = run_dir
        
        self.env = Environment()
        self.game_started = False
        self.game_over = False
        self.human_player = None # 'X' or 'O'
        self.agent = None
        
    def draw_start_screen(self):
        self.screen.fill(BG_COLOR)
        title = self.font.render("Choose Your Side", True, (255, 255, 255))
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))
        
        # Draw Buttons
        self.btn_x = pygame.Rect(50, 200, 120, 50)
        self.btn_o = pygame.Rect(230, 200, 120, 50)
        
        pygame.draw.rect(self.screen, CROSS_COLOR, self.btn_x, border_radius=10)
        pygame.draw.rect(self.screen, CIRCLE_COLOR, self.btn_o, border_radius=10)
        
        txt_x = self.font.render("Play X", True, (255, 255, 255))
        txt_o = self.font.render("Play O", True, (66, 66, 66))
        
        self.screen.blit(txt_x, (self.btn_x.centerx - txt_x.get_width() // 2, self.btn_x.centery - txt_x.get_height() // 2))
        self.screen.blit(txt_o, (self.btn_o.centerx - txt_o.get_width() // 2, self.btn_o.centery - txt_o.get_height() // 2))

    def setup_game(self, choice):
        self.human_player = choice
        self.game_started = True
        
        # Mirroring human_agent_compete() logic
        if self.human_player == 'X':
            self.agent = Agent(player='O', epsilon=0.0)
            self.agent.load_state_value_table(self.run_dir)
            self.turn = 'human'
        else:
            self.agent = Agent(player='X', epsilon=0.0)
            self.agent.load_state_value_table(self.run_dir)
            self.turn = 'agent'
            
        self.screen.fill(BG_COLOR)
        self.draw_lines()

    def draw_lines(self):
        for i in range(1, 3):
            pygame.draw.line(self.screen, LINE_COLOR, (SQUARE_SIZE * i, 0), (SQUARE_SIZE * i, WIDTH), LINE_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, (0, SQUARE_SIZE * i), (WIDTH, SQUARE_SIZE * i), LINE_WIDTH)

    def draw_figures(self):
        for row in range(3):
            for col in range(3):
                val = self.env.board[row][col]
                center = (int(col * SQUARE_SIZE + SQUARE_SIZE // 2), int(row * SQUARE_SIZE + SQUARE_SIZE // 2))
                if val == CIRCLE:
                    pygame.draw.circle(self.screen, CIRCLE_COLOR, center, CIRCLE_RADIUS, CIRCLE_WIDTH)
                elif val == CROSS:
                    pygame.draw.line(self.screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
                    pygame.draw.line(self.screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)

    def show_status(self, text):
        pygame.draw.rect(self.screen, BG_COLOR, (0, WIDTH, WIDTH, HEIGHT - WIDTH))
        msg = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, WIDTH + 20))

    def run(self):
        while True:
            if not self.game_started:
                self.draw_start_screen()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.game_started:
                        if self.btn_x.collidepoint(event.pos): self.setup_game('X')
                        if self.btn_o.collidepoint(event.pos): self.setup_game('O')
                    elif not self.game_over and self.turn == 'human':
                        # Human move logic
                        row, col = int(event.pos[1] // SQUARE_SIZE), int(event.pos[0] // SQUARE_SIZE)
                        if event.pos[1] < WIDTH and self.env.board[row][col] == EMPTY:
                            symbol = CROSS if self.human_player == 'X' else CIRCLE
                            self.env = self.env.step(row, col, symbol)
                            self.turn = 'agent'

            # Agent Turn Logic (outside event loop for auto-play)
            if self.game_started and not self.game_over and self.turn == 'agent':
                self.draw_figures(); self.show_status("Robot is thinking...")
                pygame.display.update()
                pygame.time.wait(600)
                
                r, c, _ = self.agent.select_position(self.env)
                self.env = self.env.step(r, c, self.agent.symbol)
                self.turn = 'human'

            if self.game_started:
                self.draw_figures()
                if self.env.is_done():
                    self.game_over = True
                    winner_map = {CROSS: "X Wins!", CIRCLE: "O Wins!", EMPTY: "It's a Tie!"}
                    self.show_status(winner_map[self.env.winner])
                else:
                    self.show_status(f"Turn: {self.turn.capitalize()}")

            pygame.display.update()

if __name__ == "__main__":
    """Use:
        # To play against the agent:
        python3 -m rl_eng.tic_tac_toe_pygame --run_id <run_id>
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
    TicTacToePygame(os.path.join("runs", args.run_id)).run()