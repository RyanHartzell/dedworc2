import pygame
import random
from crowd_sim_cons import *
from numpy import random


# Particle class
class Particle:
    def __init__(self, x, y, id):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * PARTICLE_MAX_SPEED
        self.id = id
        self.current_target = pygame.math.Vector2(TARGET)
        # self.personal_space = random.lognormal(0,1)*PERSONAL_SPACE
        self.personal_space = PERSONAL_SPACE

    def get_position(self):
        return (self.position[0], self.position[1])
    
    def move(self):
        # Add a pseudorandom component to the velocity
        random_walk = pygame.math.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        
        # Compute direction towards the target
        direction_to_target = pygame.math.Vector2(TARGET) - self.position
        if direction_to_target.length() > 0:
            direction_to_target.normalize_ip()

        # Update velocity with a mix of random walk and target attraction
        self.velocity += random_walk + direction_to_target * 0.1
        if self.velocity.length() > PARTICLE_MAX_SPEED:
            self.velocity.scale_to_length(PARTICLE_MAX_SPEED)

        self.position += self.velocity
    
    def detect_border_collision(self):
        # Check for collisions with screen borders
        if self.position.x - PARTICLE_RADIUS < 0 or self.position.x + PARTICLE_RADIUS > WIDTH:
            self.velocity.x *= -1  # Reverse horizontal velocity
            self.position.x = max(PARTICLE_RADIUS, min(WIDTH - PARTICLE_RADIUS, self.position.x))
            
        if self.position.y - PARTICLE_RADIUS < 0 or self.position.y + PARTICLE_RADIUS > HEIGHT:
            self.velocity.y *= -1  # Reverse vertical velocity
            self.position.y = max(PARTICLE_RADIUS, min(HEIGHT - PARTICLE_RADIUS, self.position.y))

    def detect_stage_collision(self):
        # Check for collisions with screen borders
        TARGET_BARRIER
        if self.position.distance_to(self.current_target) <= TARGET_BARRIER:
            self.velocity.x *= -1  # Reverse horizontal velocity
            self.position.x = max(PARTICLE_RADIUS, min(WIDTH - PARTICLE_RADIUS, self.position.x))
            self.velocity.y *= -1  # Reverse vertical velocity
            self.position.y = max(PARTICLE_RADIUS, min(HEIGHT - PARTICLE_RADIUS, self.position.y))
            


    def avoid_others(self, particles):
        for other in particles:
            if other == self:
                continue
            distance = self.position.distance_to(other.position)
            if distance < self.personal_space:
                # Apply a repulsion force
                direction_away = self.position - other.position
                if direction_away.length() > 0:
                    direction_away.normalize_ip()
                self.velocity += direction_away * 0.5
    
    # This function picks a random point on the map Every particle within a radius of interest will make that their new target
    def random_attractor(self):
        attractor_location = pygame.math.Vector2(random.randint(PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS), random.randint(PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS))
        # Check for collisions with screen borders
        distance = self.position.distance_to(attractor_location)
        print(distance)
        if distance <= 100:
            self.velocity.x *= -50  # Reverse horizontal velocity
            self.position.x = max(PARTICLE_RADIUS, min(WIDTH - PARTICLE_RADIUS, self.position.x))
            self.velocity.y *= -50  # Reverse horizontal velocity
            self.position.y = max(PARTICLE_RADIUS, min(WIDTH - PARTICLE_RADIUS, self.position.y))
            


    #  This function picks a random point on the map, repulsing any particle within a certain distance.
    def random_repulsor(self):
        pass

    def draw(self):
        pygame.draw.circle(screen, BLUE, (int(self.position.x), int(self.position.y)), PARTICLE_RADIUS)
