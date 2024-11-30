import pygame

# Screen settings
WIDTH, HEIGHT = 800, 600
# WIDTH, HEIGHT = 100, 100
# Screen settings
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FRAME_RATE = 60
FRAME_DURATION = 1.0/60.0

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Simulation settings
GRID_SAMPLING = 3.5
NUM_PARTICLES = 1000
NUM_DRONES = 3
PARTICLE_RADIUS = 8
PERSONAL_SPACE = 10
TARGET = (WIDTH // 2, 0)
MAX_SPEED = 2
DRONE_RADIUS = 70  # Drone's sensing radius
DRONE_REPULSION_FORCE = .1