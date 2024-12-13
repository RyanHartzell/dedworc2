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

# Simulation Settings
GRID_SAMPLING = 3.5
TARGET = (WIDTH // 2, 0)

# Particle Settings
NUM_PARTICLES = 1000
PARTICLE_RADIUS = 8
PARTICLE_MAX_SPEED = 3
PERSONAL_SPACE = 10
TARGET_BARRIER = 200

# Drone Settings
NUM_DRONES = 1
MAX_SPEED = max(WIDTH, HEIGHT) / 30 # We should probably base this off the number of frames so the drone can traverse the full width of the environment
DRONE_RADIUS = 70  # Drone's sensing radius
DRONE_REPULSION_FORCE = .1 * 500