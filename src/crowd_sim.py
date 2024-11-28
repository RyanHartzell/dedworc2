import pygame
import random
import math
from crowd_sim_cons import *
from drone import Drone
from particle import Particle

# Initialize Pygame
pygame.init()

pygame.display.set_caption("Crowd Simulator: Pseudorandom Walk to Target")

# Create particles
def create_particles():
    particles = []
    for _ in range(NUM_PARTICLES):
        x = random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
        y = random.randint(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS)
        particles.append(Particle(x, y))
    return particles


# Create particles
def create_drones():
    drones = []
    for _ in range(NUM_DRONES):
        x = random.randint(DRONE_RADIUS, WIDTH - DRONE_RADIUS)
        y = random.randint(DRONE_RADIUS, HEIGHT - DRONE_RADIUS)
        drones.append(Drone(x, y))
    return drones


# Main simulation loop
def main():
    clock = pygame.time.Clock()
    particles = create_particles()

    drones = create_drones()
    # drone = Drone(WIDTH // 4, HEIGHT // 4)

    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update particles
        for particle in particles:
            particle.avoid_others(particles)
            particle.detect_border_collision()
            # particle.random_attractor()
            particle.move()
            particle.draw()

        for drone in drones:  
            # Update and draw the drone
            drone.avoid_other_drones(drones)
            drone.patrol()
            drone.detect_border_collision()
            detected_particles = drone.sense(particles)
            drone.draw(detected_particles)

        # Draw the target
        pygame.draw.circle(screen, WHITE, TARGET, 10)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
