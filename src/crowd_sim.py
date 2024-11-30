import pygame
import random
import numpy as np
from crowd_sim_cons import *
from drone import Drone
from particle import Particle
from map import Map

class Simulation():
    def __init__(self, training_mode = False) -> None:
        self.particles = self.create_particles()
        self.drones = self.create_drones()
        self.global_map = Map((HEIGHT, WIDTH))
        self.advance = True
        self.training_mode = training_mode
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Crowd Simulator: Pseudorandom Walk to Target")
        clock = pygame.time.Clock()



    # Create particles
    def create_particles(self):
        particles = []
        for id in range(NUM_PARTICLES):
            x = random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
            y = random.randint(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS)
            particles.append(Particle(x, y, id))
        return particles


    # Create particles
    def create_drones(self):
        drones = []
        for id in range(NUM_DRONES):
            x = random.randint(DRONE_RADIUS, WIDTH - DRONE_RADIUS)
            y = random.randint(DRONE_RADIUS, HEIGHT - DRONE_RADIUS)
            drones.append(Drone(x, y, id, (HEIGHT, WIDTH)))
        return drones


    # Main simulation loop
    def main(self):
        clock = pygame.time.Clock()

        running = True
        while running:
            if(self.training_mode == True):
                self.advance = False
            
            screen.fill(BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update particles
            for particle in self.particles:
                particle.avoid_others(self.particles)
                particle.detect_border_collision()
                # particle.random_attractor()
                particle.move()
                particle.draw()

            for drone in self.drones:  
                # Update and draw the drone
                drone.avoid_other_drones(self.drones)
                drone.patrol()
                drone.detect_border_collision()
                detected_particles = drone.measure_particles(self.particles)
                drone.draw(detected_particles)

            self.global_map.update_instantaneous_occupancy_map(self.particles)
            # Draw the target
            pygame.draw.circle(screen, WHITE, TARGET, 10)

            pygame.display.flip()
            clock.tick(FRAME_RATE)

            while(not self.advance):
                continue
        
        # Main simulation loop
    def step(self):

        running = True
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update particles
        for particle in self.particles:
            particle.avoid_others(self.particles)
            particle.detect_border_collision()
            # particle.random_attractor()
            particle.move()
            particle.draw()

        for drone in self.drones:  
            # Update and draw the drone
            drone.avoid_other_drones(self.drones)
            drone.patrol()
            drone.detect_border_collision()
            detected_particles = drone.measure_particles(self.particles)
            drone.draw(detected_particles)

        self.global_map.update_instantaneous_occupancy_map(self.particles)
        # Draw the target
        pygame.draw.circle(screen, WHITE, TARGET, 10)

        pygame.display.flip()
            

            
            

        # pygame.quit()

if __name__ == "__main__":
    Training_mode = True
    crowd_sim = Simulation()
    crowd_sim.main()