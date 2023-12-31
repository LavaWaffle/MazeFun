import pygame
import math
from queue import PriorityQueue
from typing import List
from time import sleep
pygame.init()

WIDTH = 500
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

font = pygame.font.Font('ComicNeue-Regular.ttf', 32)

# Colors
RED = (255, 0, 0)  # Closed
GREEN = (0, 255, 0)  # Open
BLUE = (0, 0, 255)  # Barrier
YELLOW = (255, 255, 0)  # Start
WHITE = (255, 255, 255)  # Empty
BLACK = (0, 0, 0)  # Border
PURPLE = (128, 0, 128)  # Path
ORANGE = (255, 165, 0)  # End
GREY = (128, 128, 128)  # Grid Lines
TURQUOISE = (64, 224, 208)  # Border
BROWN = (222, 184, 135)  # Wall


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
        self.temp_color = None
        self.text = None
        self.text_rect = None
        self.border = None

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
        self.text = None
        self.text_rect = None
        self.border = None

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

    # Temp change color
    def temp_change(self, color):
        self.temp_color = color

    def make_gate(self):
        self.border = GREEN
    
    def remove_temp(self):
        self.temp_color = None

    def set_text(self, text):
        self.text = font.render(text, True, BLACK)
        self.text_rect = self.text.get_rect()
        self.text_rect.center = (self.x + self.width / 2, self.y + self.width / 2)
    
    # Draw spot
    def draw(self, win):
        if self.temp_color is None:
            pygame.draw.rect(win, self.color,
                             (self.x, self.y, self.width, self.width))
        else:
            pygame.draw.rect(win, self.temp_color,
                             (self.x, self.y, self.width, self.width))
        if self.text != None:
            win.blit(self.text, self.text_rect)
        if self.border != None:
            pygame.draw.rect(win, self.border,
                             (self.x+5, self.y+5, self.width-8, self.width-8), 10)

    def update_neighbors(self, grid, walls):
        other_spots = []
        for wall in walls:
            if wall.spot1 == self:
                if wall.spot2 not in other_spots:
                    other_spots.append(wall.spot2)
            elif wall.spot2 == self:
                if wall.spot1 not in other_spots:
                    other_spots.append(wall.spot1)
                
        self.neighbors = []
        # Check if spot below is not barrier
        if self.row < self.total_rows - 1 and not grid[self.row + 1][
                self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # Check if spot above is not barrier
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # Check if spot right is not barrier
        if self.col < self.total_rows - 1 and not grid[self.row][
                self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # Check if spot left is not barrier
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

        # check if neighbors include any other_spots. If so remove that neighbor
        for spot in other_spots:
            if spot in self.neighbors:
                self.neighbors.remove(spot)
        
    def __lt__(self, other):
        return False

    def __equa__(self, other):
        return self.col == other.col and self.row == other.row
    
    def __repr__(self):
        return f"({self.row}, {self.col})"
    
    def clone(self):
        temp = Spot(self.row, self.col, self.width, self.total_rows)
        if self.temp_color is not None:
            temp.temp_change(self.temp_color)
        if self.color is not None:
            temp.color = self.color
        if self.text is not None:
            temp.text = self.text
        if self.text_rect is not None:
            temp.text_rect = self.text_rect
        if self.border is not None:
            temp.border = self.border
        return temp


class Wall:

    def __init__(self, spot1: Spot, spot2: Spot):
        self.spot1 = spot1
        self.spot2 = spot2
        self.direction = "Y"
        # spot 1 and spot 2 are two adjacent spots, a wall should be drawn between them on the border line
        # check if spot 1 is above spot 2
        if spot1.col > spot2.col:
            # Spot 1 is below spot 2
            # Define x and y of wall
            self.direction = "H"
            self.y = spot2.y + spot2.width
            self.x = spot1.x
            self.width = spot2.width
        elif spot1.col < spot2.col:
            # Spot 1 is above spot 2
            # Define x and y of wall
            self.direction = "H"
            self.y = spot1.y + spot1.width
            self.x = spot2.x
            self.width = spot1.width
        elif spot1.row > spot2.row:
            # Spot 1 is to the right of spot 2
            # Define x and y of wall
            self.direction = "V"
            self.y = spot2.y
            self.x = spot1.x
            self.length = spot2.width
        elif spot1.row < spot2.row:
            # Spot 1 is to the left of spot 2
            # Define x and y of wall
            self.direction = "V"
            self.y = spot1.y
            self.x = spot2.x
            self.length = spot1.width

    def draw(self, win):
        if self.direction == "H":
            pygame.draw.line(win, BROWN, (self.x, self.y),
                             (self.x + self.width, self.y), 10)
        elif self.direction == "V":
            pygame.draw.line(win, BROWN, (self.x, self.y),
                             (self.x, self.y + self.length), 10)


# Heuristic function
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # Manhattan distance
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw) -> List[Spot]:
    retval = []
    # Draw path
    while current in came_from:
        current = came_from[current]
        current.make_path()
        retval.append(current)
        draw()
    return retval


def algorithm(draw, grid, start, end) -> dict[bool, List[Spot]]:
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
            val = reconstruct_path(came_from, end, draw)
            return [True, val]

        # Check neighbors
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                # Update came from
                came_from[neighbor] = current
                # Update g score
                g_score[neighbor] = temp_g_score
                # Update f score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(),
                                                     end.get_pos())
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
    return [False, []]

def set_grid_to_snapshot(grid, snapshot, walls):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].color = snapshot[i][j].color
            grid[i][j].text = snapshot[i][j].text
            grid[i][j].text_rect = snapshot[i][j].text_rect
            grid[i][j].border = snapshot[i][j].border
            grid[i][j].temp_color = snapshot[i][j].temp_color
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].update_neighbors(grid, walls)


from time import sleep

from enum import Enum
class Instruction(Enum):
    Forward = 1
    Backward = 2
    TurnLeft = 3
    TurnRight = 4

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
class Instructions:
    def __init__(self, direction: Direction, lots_of_spots: List[List[Spot]]):
        self.instructions: List[Instruction] = []
        self.direction = direction
        self.turns = 0
        self.moves = 0
        previous_spot = None

        for spots in lots_of_spots:
            for spot in spots:
                if previous_spot is None:
                    previous_spot = spot
                    continue
                
                is_special = False
                if spots == lots_of_spots[-1] and spot == spots[-1]:
                    is_special = True
                elif spot == spots[0]:
                    is_special = True

                x_diff = spot.row - previous_spot.row
                y_diff = spot.col - previous_spot.col

                previous_spot = spot

                if x_diff == 1:
                    # Move right one
                    if(self.direction == Direction.LEFT) and not is_special:
                        self.backward()
                    else:
                        self.face_direction(Direction.RIGHT)
                        self.forward()
                elif x_diff == -1:
                    # Move left one
                    if(self.direction == Direction.RIGHT) and not is_special:
                        self.backward()
                    else:
                        self.face_direction(Direction.LEFT)
                        self.forward()
                elif y_diff == 1:
                    # Move down one
                    if(self.direction == Direction.UP) and not is_special:
                        self.backward()
                    else:
                        self.face_direction(Direction.DOWN)
                        self.forward()
                elif y_diff == -1:
                    # Move up one
                    if(self.direction == Direction.DOWN) and not is_special:
                        self.backward()
                    else:
                        self.face_direction(Direction.UP)
                        self.forward()
    
    def __repr__(self):
        return f"Instructions: {self.instructions}, turns: {self.turns}, moves: {self.moves}"
    
    def turn(self, left_or_right):
        if left_or_right == "left":
            self.instructions.append(Instruction.TurnLeft)
        elif left_or_right == "right":
            self.instructions.append(Instruction.TurnRight)
        self.turns += 1
    
    def face_direction(self, desired_direction: Direction):
        if(self.direction != desired_direction):
            if (self.direction == Direction.UP and desired_direction == Direction.RIGHT):
                self.turn("right")
            elif (self.direction == Direction.UP and desired_direction == Direction.LEFT):
                self.turn("left")
            elif (self.direction == Direction.UP and desired_direction == Direction.DOWN):
                self.turn("right")
                self.turn("right")
            elif (self.direction == Direction.DOWN and desired_direction == Direction.RIGHT):
                self.turn("left")
            elif (self.direction == Direction.DOWN and desired_direction == Direction.LEFT):
                self.turn("right")
            elif (self.direction == Direction.DOWN and desired_direction == Direction.UP):
                self.turn("right")
                self.turn("right")
            elif (self.direction == Direction.LEFT and desired_direction == Direction.RIGHT):
                self.turn("right")
                self.turn("right")
            elif (self.direction == Direction.LEFT and desired_direction == Direction.DOWN):
                self.turn("left")
            elif (self.direction == Direction.LEFT and desired_direction == Direction.UP):
                self.turn("right")
            elif (self.direction == Direction.RIGHT and desired_direction == Direction.LEFT):
                self.turn("right")
                self.turn("right")
            elif (self.direction == Direction.RIGHT and desired_direction == Direction.DOWN):
                self.turn("right")
            elif (self.direction == Direction.RIGHT and desired_direction == Direction.UP):
                self.turn("left")
            self.direction = desired_direction

    def forward(self):
        self.instructions.append(Instruction.Forward)
        self.moves += 1

    def backward(self):
        self.instructions.append(Instruction.Backward)
        self.moves += 1
    
        


def gate_algorithm(draw, grid: List[List[Spot]], start: Spot, gate_a: Spot, gate_b: Spot, gate_c: Spot, end: Spot, walls: List[Wall]):
    grid_snapshot = [[spot.clone() for spot in row] for row in grid]
    
    mini_paths = {}
    
    mini_paths['start'] = {}
    mini_paths['start']['A'] = algorithm(draw, grid, start, gate_a)[1][::-1]
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    sleep(.1)
    mini_paths['start']['B'] = algorithm(draw, grid, start, gate_b)[1][::-1]
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    sleep(.1)
    mini_paths['start']['C'] = algorithm(draw, grid, start, gate_c)[1][::-1]
    sleep(.1)

    mini_paths['A'] = {}
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['A']['B'] = algorithm(draw, grid, gate_a, gate_b)[1][::-1]
    sleep(.1)
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['A']['C'] = algorithm(draw, grid, gate_a, gate_c)[1][::-1]
    sleep(.1)
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['A']['END'] = algorithm(draw, grid, gate_a, end)[1][::-1]
    sleep(.1)

    mini_paths['B'] = {}
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['B']['A'] = algorithm(draw, grid, gate_b, gate_a)[1][::-1]
    sleep(.1)
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['B']['C'] = algorithm(draw, grid, gate_b, gate_c)[1][::-1]
    sleep(.1)
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['B']['END'] = algorithm(draw, grid, gate_b, end)[1][::-1]
    sleep(.1)

    mini_paths['C'] = {}
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['C']['A'] = algorithm(draw, grid, gate_c, gate_a)[1][::-1]
    sleep(.1)
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['C']['B'] = algorithm(draw, grid, gate_c, gate_b)[1][::-1]
    sleep(.1)
    set_grid_to_snapshot(grid, grid_snapshot, walls)
    mini_paths['C']['END'] = algorithm(draw, grid, gate_c, end)[1][::-1]
    sleep(.1)

    set_grid_to_snapshot(grid, grid_snapshot, walls)
    print("DONE")
    paths: dict[str, List[List[Spot]]] = {
        'start-a-b-c-end': [mini_paths['start']['A'], mini_paths['A']['B'], mini_paths['B']['C'], mini_paths['C']['END'] + [end]],
        'start-a-c-b-end': [mini_paths['start']['A'], mini_paths['A']['C'], mini_paths['C']['B'], mini_paths['B']['END'] + [end]],
        'start-b-a-c-end': [mini_paths['start']['B'], mini_paths['B']['A'], mini_paths['A']['C'], mini_paths['C']['END'] + [end]],
        'start-b-c-a-end': [mini_paths['start']['B'], mini_paths['B']['C'], mini_paths['C']['A'], mini_paths['A']['END'] + [end]],
        'start-c-a-b-end': [mini_paths['start']['C'], mini_paths['C']['A'], mini_paths['A']['B'], mini_paths['B']['END'] + [end]],
        'start-c-b-a-end': [mini_paths['start']['C'], mini_paths['C']['B'], mini_paths['B']['A'], mini_paths['A']['END'] + [end]]
    }

    # # draw path start-c-b-a-end
    # for spot in paths['start-c-b-a-end']:
    #     spot.make_path()
    #     draw()
    #     sleep(0.5)

    # loop over paths and print each key and value
    for key, value in paths.items():
        print(key, ' : ', len(value), ' : ', value)
    
    # find path with least amount of turns
    least_turns = 100000
    all_instructions = []
    initial_direction = Direction.UP
    # get user input for direction
    user_direction = input("Enter direction (up, down, left, right): ")
    if user_direction == "down":
        initial_direction = Direction.DOWN
    elif user_direction == "left":
        initial_direction = Direction.LEFT
    elif user_direction == "right":
        initial_direction = Direction.RIGHT
    for key, value in paths.items():
        instructions = Instructions(Direction.UP, value)
        all_instructions.append(instructions)
        if instructions.turns < least_turns:
            least_turns = instructions.turns
            instructions_with_least_turns = instructions

    return instructions_with_least_turns

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


def make_wall(spot1, spot2, walls):
    print(f"Making wall spot1: {spot1}, spot2: {spot2}")
    spot1.remove_temp()
    if spot1 == spot2: 
        print("Wall failed, same spot")
        return
    
    # check if wall already exists
    for wall in walls:
        if wall.spot1 == spot1 and wall.spot2 == spot2:
            print("Wall already exists")
            return
        elif wall.spot1 == spot2 and wall.spot2 == spot1:
            print("Wall already exists")
            return
    
    walls.append(Wall(spot1, spot2))


# Draw grid lines
def draw_grid_lines(win, rows, width):
    gap = width // rows
    for i in range(rows):
        # Draw horizontal lines
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap), 10)
        # Draw vertical lines
        pygame.draw.line(win, GREY, (i * gap, 0), (i * gap, width), 10)
    # Missing lines at right edge and bottom edge
    pygame.draw.line(win, GREY, (width, 0), (width, width), 10)
    pygame.draw.line(win, GREY, (0, width), (width, width), 10)


# DRAW EVERYTHING
def draw(win, grid, rows, width, walls):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid_lines(win, rows, width)

    for wall in walls:
        wall.draw(win)

    pygame.display.update()


# Get get mouse position
def get_mouse_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def is_spot_special(spot, *args):
    for arg in args:
        if arg == spot:
            return True
    return False

# Main algorithm
def main(win, width):
    ROWS = 6  # Make this 6 for sci oly
    grid = make_grid(ROWS, width)

    start = None
    gate_a = None
    gate_b = None
    gate_c = None
    end = None

    run = True
    started = False

    walls = []

    f_key_pressed = False
    spot1 = None

    while run:
        draw(win, grid, ROWS, width, walls)
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
                if not start and not is_spot_special(spot, gate_a, gate_b, gate_c, end) and not f_key_pressed:
                    start = spot
                    start.make_start()
                    start.set_text("start")
                # Set gate a
                if not gate_a and not is_spot_special(spot, start, gate_b, gate_c, end) and not f_key_pressed:
                    gate_a = spot
                    gate_a.make_gate()
                    gate_a.set_text("A")
                # Set gate b
                elif not gate_b and not is_spot_special(spot, start, gate_a, gate_c, end) and not f_key_pressed:
                    gate_b = spot
                    gate_b.make_gate()
                    gate_b.set_text("B")
                # Set gate c
                elif not gate_c and not is_spot_special(spot, start, gate_a, gate_b, end) and not f_key_pressed:
                    gate_c = spot
                    gate_c.make_gate()
                    gate_c.set_text("C")
                # Set end
                elif not end and not is_spot_special(spot, start, gate_a, gate_b, gate_c) and not f_key_pressed:
                    end = spot
                    end.make_end()
                    end.set_text("end")
                # Set barrier
                elif not is_spot_special(spot, start, gate_a, gate_b, gate_c, end) and not f_key_pressed:
                    spot.make_barrier()
                # Set start wall
                elif f_key_pressed and spot1 is None:
                    spot1 = spot
                    spot1.temp_change(BROWN)
                    print(spot1)
                # Set end wall
                elif f_key_pressed and spot1 is not None:
                    spot2 = spot
                    make_wall(spot1, spot2, walls)
                    spot1 = None

            # Right mouse button
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                spot: Spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == gate_a:
                    gate_a = None
                elif spot == gate_b:
                    gate_b = None
                elif spot == gate_c:
                    gate_c = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started and start and end:
                    started = True
                    # Set neighbors
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid, walls)

                    # Run algorithm
                    owo = gate_algorithm(lambda: draw(win, grid, ROWS, width, walls), grid,
                              start, gate_a, gate_b, gate_c, end, walls)
                    print(owo)
                    started = False
                if event.key == pygame.K_r:
                    start = None
                    gate_a = None
                    gate_b = None
                    gate_c = None
                    end = None
                    walls = []
                    grid = make_grid(ROWS, width)

                if event.key == pygame.K_f:
                    f_key_pressed = True

                if event.key == pygame.K_z:
                    # remove last wall from array
                    if walls:
                        walls.pop()
                if event.key == pygame.K_w:
                    walls = []

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_f:
                    f_key_pressed = False
    pygame.quit()


main(WIN, WIDTH)
