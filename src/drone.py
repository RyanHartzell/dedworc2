"""
Drone module containing Agent definition:

REQUIREMENTS:
=============
* Must be able to move within boundary of environment
* Must be able to observe local crowd state
* Must be able to build internal map of global environment
* Must own actor/critic modules
* Must be able to visualize map
    * Traces
    * Current Crowd Belief state
* Must have Random Walk mode
* Must be able to flush  history to disk

"""

__authors__=["Ryan Hartzell", "Matt Desaulniers"]

class Drone:
    def __init__(self, env, pos_x=0.0, pos_y=0.0, fov=5.0) -> None:
        # Basic characteristics
        self._pos = (pos_x, pos_y)
        self._fov = fov

        # Packet Loss Rate (DEFAULT: Full efficiency)
        self.plr = 0.0 # probability that messages to/from this drone are dropped/lost on global update

        # Belief map (over a fixed grid)
        self.env = env # Reference to our simulation environment
        self.belief = self.env.raster # Should be a masked array or fixed grid with -1 values? Check with Matt on his implementation

        # Set horizon
        self.horizon = 2.0 * self._fov

        # Records and history
        self.state_history = []
        self.obs_history = []

    @property
    def position(self):
        return self._pos

    @property
    def fov(self):
        return self._fov # This is basically just a radius about the Center of Mass
    
    def _merge_maps(self):
        return

    def step(self):
        self.state_history.append(new_state)

    def observe(self):
        self.obs_history.append()

    def render(self):
        return


if __name__=="__main__":
    print("Initializing Drone Test....")

    drones = [Drone() for _ in range(3)]