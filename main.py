import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

# Colors
RED = (255, 0, 0) # Closed
GREEN = (0, 255, 0) # Open
BLUE = (0, 0, 255) # Barrier
YELLOW = (255, 255, 0) # Start
WHITE = (255, 255, 255) # Empty
BLACK = (0, 0, 0) # Border
PURPLE = (128, 0, 128) # Path
ORANGE = (255, 165, 0) # End
GREY = (128, 128, 128) # Grid Lines
TURQUOISE = (64, 224, 208) # Border

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    # Get position of spot
    def get_pos(self):
        return self.row, self.col
    
    # Check if spot is closed
    def is_closed(self):
        return self.color == RED
    
    # Check if spot is open
    def is_open(self):
        return self.color == GREEN
    
    # Check if spot is barrier
    def is_barrier(self):
        return self.color == BLACK
    
    # Check if spot is start
    def is_start(self):
        return self.color == ORANGE
    
    # Check if spot is end
    def is_end(self):
        return self.color == TURQUOISE
    
    # Reset spot
    def reset(self):
        self.color = WHITE

    # Make spot closed
    def make_closed(self):
        self.color = RED

    # Make spot open
    def make_open(self):
        self.color = GREEN

    # Make spot barrier
    def make_barrier(self):
        self.color = BLACK

    # Make spot start
    def make_start(self):
        self.color = ORANGE

    # Make spot end
    def make_end(self):
        self.color = TURQUOISE

    # Make spot path
    def make_path(self):
        self.color = PURPLE

    # Draw spot
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # Check if spot below is not barrier
        if self.row < self.total_rows -1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # Check if spot above is not barrier
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # Check if spot right is not barrier
        if self.col < self.total_rows -1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # Check if spot left is not barrier
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False
    

# Heuristic function
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Manhattan distance
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    # Draw path
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def algorithm(draw, grid, start, end):
    count = 0
    # Priority queue
    open_set = PriorityQueue()
    # Add start to queue
    open_set.put((0, count, start))
    # Keep track of where we came from
    came_from = {}
    # Keep track of g score
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    # Keep track of f score
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    # Keep track of what's in the queue
    open_set_hash = {start}

    while not open_set.empty():
        # Quit if user closes window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current: Spot = open_set.get()[2]
        open_set_hash.remove(current)

        # If we found the end
        if current == end:
            # Draw path
            reconstruct_path(came_from, end, draw)
            # Make end
            end.make_end()
            # Make start
            start.make_start()
            return True
        
        # Check neighbors
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                # Update came from
                came_from[neighbor] = current
                # Update g score
                g_score[neighbor] = temp_g_score
                # Update f score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                # Add neighbor to queue
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    # Make neighbor open
                    neighbor.make_open()
        
        draw()

        if current != start:
            # Make current closed
            current.make_closed()
    return False


# Make grid
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot) 

    return grid

# Draw grid lines
def draw_grid_lines(win, rows, width):
    gap = width // rows
    for i in range(rows):
        # Draw horizontal lines
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        # Draw vertical lines
        pygame.draw.line(win, GREY, (i * gap, 0), (i * gap, width))

# DRAW EVERYTHING
def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)
    
    draw_grid_lines(win, rows, width)
    pygame.display.update()

# Get get mouse position
def get_mouse_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

# Main algorithm
def main(win, width):
    ROWS = 50 # Make this 6 for sci oly
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    started = False
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            # Exit
            if event.type == pygame.QUIT:
                run = False

            # Ignore other user input if algorithm is running
            if started:
                continue
            
            # Left mouse button
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                spot: Spot = grid[row][col]

                # Set start
                if not start and spot is not end:
                    start = spot
                    start.make_start()
                # Set end
                elif not end and spot is not start:
                    end = spot
                    end.make_end()
                # Set barrier
                elif spot is not start and spot is not end:
                    spot.make_barrier()

            # Right mouse button
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                spot: Spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started and start and end:
                    started = True
                    # Set neighbors
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    # Run algorithm
                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    started = False
                if event.key == pygame.K_r:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)
