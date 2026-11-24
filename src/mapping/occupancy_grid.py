import numpy as np

class OccupancyGrid3D:
    """
    Onboard 3D Occupancy Voxel Grid.
    Owner: Bhargav Kumar
    
    Represents local volumetric map in memory, generates frontier deltas to transmit
    over low-bandwidth mesh radios, and merges peer maps.
    """
    def __init__(self, bounds=(200.0, 200.0, 50.0), resolution=5.0):
        self.bounds = np.array(bounds, dtype=np.float32)
        self.resolution = float(resolution)
        self.dims = np.ceil(self.bounds / self.resolution).astype(np.int32)
        # 0 = Unknown, 1 = Free, 2 = Occupied
        self.grid = np.zeros(tuple(self.dims), dtype=np.int8)
        self.new_deltas = set()

    def integrate_scan(self, origin, point_cloud):
        """Updates voxel occupancy from 3D sensor point cloud."""
        for pt in point_cloud:
            pt = np.asarray(pt, dtype=np.float32)
            idx = np.floor(pt / self.resolution).astype(np.int32)
            if np.all(idx >= 0) and np.all(idx < self.dims):
                t_idx = (int(idx[0]), int(idx[1]), int(idx[2]))
                if self.grid[t_idx] != 2:
                    self.grid[t_idx] = 2  # Occupied
                    self.new_deltas.add((t_idx, 2))

    def extract_deltas(self):
        """Returns compressed coordinates of newly discovered voxels for P2P broadcast."""
        deltas = list(self.new_deltas)
        self.new_deltas.clear()
        return deltas

    def merge_deltas(self, deltas):
        """Merges received peer frontier deltas into local occupancy grid."""
        for idx, val in deltas:
            if 0 <= idx[0] < self.dims[0] and 0 <= idx[1] < self.dims[1] and 0 <= idx[2] < self.dims[2]:
                self.grid[idx] = max(self.grid[idx], int(val))

