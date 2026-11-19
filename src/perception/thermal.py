"""Vectorized, two-stage thermal contacts with obstacle line-of-sight checks."""
import numpy as np

_OBSTACLE_BROADPHASE = {}

FAINT_SIGNAL_RADIUS = 50.0
CONFIRM_RADIUS = 10.0
FIRE_ALERT_RADIUS = 25.0


def _inside_building(pos, building, expansion=2.0):
    if building is None or building.shape != "box":
        return False
    half = building.size_params[:2] / 2.0 + expansion
    delta = np.asarray(pos)[:2] - building.pos[:2]
    angle = np.deg2rad(building.heading_deg)
    local = np.array([np.cos(angle) * delta[0] + np.sin(angle) * delta[1],
                      -np.sin(angle) * delta[0] + np.cos(angle) * delta[1]])
    return bool(np.all(np.abs(local) <= half))


def _segment_hits_obstacle(start, end, obstacle):
    """Fast exact segment test for the supported obstacle primitives."""
    a, b = np.asarray(start, dtype=float).copy(), np.asarray(end, dtype=float).copy()
    if obstacle.shape == "box":
        angle = np.deg2rad(obstacle.heading_deg)
        c, s = np.cos(angle), np.sin(angle)
        for point in (a, b):
            dx, dy = point[0] - obstacle.pos[0], point[1] - obstacle.pos[1]
            point[0] = c * dx + s * dy
            point[1] = -s * dx + c * dy
            point[2] -= obstacle.pos[2]
        half = obstacle.size_params[:3] / 2.0
        lo, hi = 0.0, 1.0
        delta = b - a
        for axis in range(3):
            if abs(delta[axis]) < 1e-10:
                if abs(a[axis]) > half[axis]:
                    return False
                continue
            t1, t2 = (-half[axis] - a[axis]) / delta[axis], (half[axis] - a[axis]) / delta[axis]
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
            if lo > hi:
                return False
        return True
    if obstacle.shape in ("sphere", "tree"):
        centers = [(obstacle.pos, obstacle.size_params[0])] if obstacle.shape == "sphere" else [
            (np.array([obstacle.pos[0], obstacle.pos[1], obstacle.size_params[3]]), obstacle.size_params[2])]
        for center, radius in centers:
            v = b - a
            t = np.clip(np.dot(center - a, v) / max(np.dot(v, v), 1e-12), 0.0, 1.0)
            if np.linalg.norm(a + t * v - center) <= radius:
                return True
        if obstacle.shape == "tree":
            return _segment_hits_cylinder(a, b, obstacle.pos, obstacle.size_params[0], obstacle.size_params[1])
        return False
    if obstacle.shape == "cylinder":
        return _segment_hits_cylinder(a, b, obstacle.pos, obstacle.size_params[0], obstacle.size_params[1])
    return False


def _segment_hits_cylinder(a, b, center, radius, height):
    """Vertical finite cylinder intersection, accounting for both radial and Z bounds."""
    a, b = np.asarray(a), np.asarray(b)
    c = np.asarray(center)
    v = b - a
    q = a[:2] - c[:2]
    radial_v = v[:2]
    A = float(np.dot(radial_v, radial_v))
    B = 2.0 * float(np.dot(q, radial_v))
    C = float(np.dot(q, q) - radius * radius)
    if A < 1e-12:
        if C > 0:
            return False
        radial_lo, radial_hi = 0.0, 1.0
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return False
        root = np.sqrt(disc)
        radial_lo, radial_hi = max(0.0, (-B-root)/(2*A)), min(1.0, (-B+root)/(2*A))
    zlo, zhi = c[2] - height / 2.0, c[2] + height / 2.0
    if abs(v[2]) < 1e-12:
        return zlo <= a[2] <= zhi and radial_lo <= radial_hi
    t1, t2 = (zlo-a[2])/v[2], (zhi-a[2])/v[2]
    return max(radial_lo, min(t1, t2)) <= min(radial_hi, max(t1, t2))


def _los_clear(start, end, obstacles):
    # Sample each candidate segment and use the existing exact point-to-solid geometry.
    delta = end - start
    length = float(np.linalg.norm(delta))
    midpoint = (start + end) * 0.5
    # Vectorized broadphase, consistent with the obstacle-position prefilter used
    # elsewhere in the simulator. Exact geometry is tested only for nearby solids.
    if not obstacles:
        return True
    key = id(obstacles)
    cached = _OBSTACLE_BROADPHASE.get(key)
    if cached is None or cached[0] is not obstacles:
        centers = np.asarray([o.pos for o in obstacles], dtype=np.float64)
        radii = []
        for o in obstacles:
            if o.shape == "box":
                radii.append(float(np.linalg.norm(o.size_params[:3] / 2.0)))
            elif o.shape == "cylinder":
                radii.append(float(np.hypot(o.size_params[0], o.size_params[1] / 2.0)))
            elif o.shape == "sphere":
                radii.append(float(o.size_params[0]))
            elif o.shape == "tree":
                radii.append(float(o.size_params[2] + o.size_params[1] / 2.0))
            else:
                radii.append(float(np.max(o.size_params)))
        cached = (obstacles, centers, np.asarray(radii, dtype=np.float64))
        _OBSTACLE_BROADPHASE[key] = cached
    _, centers, radii = cached
    delta = end - start
    denom = max(float(np.dot(delta, delta)), 1e-12)
    along = np.clip(((centers - start) @ delta) / denom, 0.0, 1.0)
    closest = start + along[:, None] * delta
    lateral = np.sum((centers - closest) ** 2, axis=1)
    candidates = np.flatnonzero(lateral <= (radii + 1.0) ** 2)
    for idx in candidates:
        obstacle = obstacles[int(idx)]
        if _segment_hits_obstacle(start, end, obstacle):
            return False
    return True


def get_thermal_contacts(drone, humans, fires, obstacles, local_ambient_fn):
    """Return faint human/hot-blob contacts, confirmed victims, and nearby fire alerts."""
    result = {"faint": [], "confirmed": [], "fire_alerts": []}
    dpos = np.asarray(drone.pos, dtype=np.float64)
    if humans:
        positions = np.asarray([h.pos for h in humans], dtype=np.float64)
        distances = np.linalg.norm(positions - dpos, axis=1)
        for idx in np.flatnonzero(distances <= FAINT_SIGNAL_RADIUS):
            human, distance = humans[idx], float(distances[idx])
            building = next((o for o in obstacles if o.name == human.parent_building), None) if human.indoor else None
            if human.indoor:
                if not _inside_building(dpos, building):
                    continue
            elif not _los_clear(dpos, np.asarray(human.pos), obstacles):
                continue
            if local_ambient_fn(human.pos) > human.surface_temp_c and distance > CONFIRM_RADIUS:
                continue
            contact = {"id": human.id, "distance": distance,
                       "relative_position": tuple(np.asarray(human.pos) - dpos)}
            human.detected_faint = True
            if distance <= CONFIRM_RADIUS:
                human.confirmed = True
                result["confirmed"].append(contact)
            else:
                result["faint"].append(contact)
    if fires:
        positions = np.asarray([f.pos for f in fires], dtype=np.float64)
        distances = np.linalg.norm(positions - dpos, axis=1)
        for idx in np.flatnonzero(distances <= FAINT_SIGNAL_RADIUS):
            fire, distance = fires[idx], float(distances[idx])
            building = next((o for o in obstacles if o.name == fire.parent_building), None) if fire.indoor else None
            if fire.indoor:
                if not _inside_building(dpos, building):
                    continue
            elif not _los_clear(dpos, np.asarray(fire.pos), obstacles):
                continue
            contact = {"id": fire.id, "distance": distance,
                       "relative_position": tuple(np.asarray(fire.pos) - dpos), "type": "fire"}
            result["faint"].append(contact)
            if distance <= FIRE_ALERT_RADIUS:
                result["fire_alerts"].append(contact)
    result["faint"].sort(key=lambda c: c["distance"])
    result["confirmed"].sort(key=lambda c: c["distance"])
    result["fire_alerts"].sort(key=lambda c: c["distance"])
    return result
