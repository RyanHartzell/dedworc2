import pygame

# Screen settings
WIDTH, HEIGHT = 800, 600
# Screen settings
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Simulation settings
NUM_PARTICLES = 500
NUM_DRONES = 4
PARTICLE_RADIUS = 8
PERSONAL_SPACE = 20
TARGET = (WIDTH // 2, 0)
MAX_SPEED = 2
DRONE_RADIUS = 70  # Drone's sensing radius