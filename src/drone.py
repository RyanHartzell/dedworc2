import pygame
from crowd_sim_cons import *
from map import Map

class Drone:
    def __init__(self, x, y, id, map_shape):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(1, 1).normalize() * 2
        self.id = id
        self.map = Map(map_shape)

    def patrol(self, target=TARGET):
        # Move towards the target (similar to particles)
        direction_to_target = pygame.math.Vector2(target) - self.position
        if direction_to_target.length() > 0:
            direction_to_target.normalize_ip()
        self.velocity += direction_to_target * 0.1

        # Limit speed
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)

        self.position += self.velocity

    def avoid_other_drones(self, drones):
        for other in drones:
            if other == self:
                continue
            distance = self.position.distance_to(other.position)
            if distance < DRONE_RADIUS * 2:  # Avoid overlap of sensing radii
                direction_away = self.position - other.position
                if direction_away.length() > 0:
                    direction_away.normalize_ip()
                self.velocity += direction_away * DRONE_REPULSION_FORCE

    def measure_particles(self, particles):
        detected_particles = []
        for particle in particles:
            distance = self.position.distance_to(particle.position)
            if distance < DRONE_RADIUS:
                detected_particles.append(particle)
        self.map.update_instantaneous_occupancy_map(detected_particles)
        return detected_particles
    
    def detect_border_collision(self):
        # Check for collisions with screen borders
        if self.position.x - DRONE_RADIUS < 0 or self.position.x + DRONE_RADIUS > WIDTH:
            self.velocity.x *= -1  # Reverse horizontal velocity
            self.position.x = max(DRONE_RADIUS, min(WIDTH - DRONE_RADIUS, self.position.x))
            
        if self.position.y - DRONE_RADIUS < 0 or self.position.y + DRONE_RADIUS > HEIGHT:
            self.velocity.y *= -1  # Reverse vertical velocity
            self.position.y = max(DRONE_RADIUS, min(HEIGHT - DRONE_RADIUS, self.position.y))



    def draw(self, detected_particles):
        # Draw drone and sensing radius
        pygame.draw.circle(screen, RED, (int(self.position.x), int(self.position.y)), 10)
        pygame.draw.circle(screen, WHITE, (int(self.position.x), int(self.position.y)), DRONE_RADIUS, 1)

        # Highlight detected particles
        for particle in detected_particles:

            pygame.draw.circle(screen, GREEN, (int(particle.position.x), int(particle.position.y)), PARTICLE_RADIUS)
