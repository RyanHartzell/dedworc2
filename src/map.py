import numpy as np
import particle
from crowd_sim_cons import *
from scipy.signal import convolve2d

def circular_pass_filter(shape, radius=0.1):
    x, y = np.meshgrid(np.linspace(-radius, radius, int(shape[0])), np.linspace(-radius, radius, int(shape[1])))
    circle = np.sqrt(x*x + y*y)

    return np.where(circle > radius, 0, 1)


class Map:
    def __init__(self, map_shape) -> None:
        self.map_shape = map_shape
        self.mesh_arrays = (np.linspace(0, 600, num=int(600.0 / (DRONE_RADIUS / GRID_SAMPLING))), np.linspace(0, 800, num=int(800.0 / (DRONE_RADIUS / GRID_SAMPLING))))
        self.mesh_shape = (int(600.0 / (DRONE_RADIUS / GRID_SAMPLING)), int(800.0 / (DRONE_RADIUS / GRID_SAMPLING)))

        print(np.linspace(0, 600, num=int(600.0 / (DRONE_RADIUS / GRID_SAMPLING))))
        print(np.linspace(0, 800, num=int(800.0 / (DRONE_RADIUS / GRID_SAMPLING))))

        self.instantaneous_occupancy_map = np.zeros(self.mesh_shape)
        self.density = np.zeros_like(self.instantaneous_occupancy_map)
        self.coordinate_mesh = np.meshgrid(self.mesh_arrays[1], self.mesh_arrays[0])
        

    def update_instantaneous_occupancy_map(self, particles):
        self.instantaneous_occupancy_map = np.zeros(self.mesh_shape)
        for particle in particles:
            position = particle.get_position()
            
            self.instantaneous_occupancy_map[int((position[1] / (DRONE_RADIUS / GRID_SAMPLING) )//1), int(position[0] / (DRONE_RADIUS / GRID_SAMPLING) )//1] += 1

    def get_density_map(self):
        # tophat = np.array([[0,1,0],[1,1,1],[0,1,0]])
        tophat = circular_pass_filter((GRID_SAMPLING*2, GRID_SAMPLING*2), GRID_SAMPLING)
        if self.instantaneous_occupancy_map.sum() == 0:
            self.density = np.ones_like(self.density)
            self.density /= self.density.sum()
        else:
            self.density[...] = convolve2d(self.instantaneous_occupancy_map, tophat, mode='same')
            self.density = self.density/self.density.sum()
        return self.density


if __name__=="__main__":
    class Dummy:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def get_position(self):
            return (self.x, self.y)
    
    locs = np.random.uniform(0, 100, size=(2,400))
    particles = [Dummy(locs[0,i], locs[1,i]) for i in range(locs.shape[-1])]
    m = Map((HEIGHT, WIDTH))
    m.update_instantaneous_occupancy_map(particles)
    dm = m.get_density_map()

    import matplotlib.pyplot as plt
    plt.imshow(dm)
    plt.show()
