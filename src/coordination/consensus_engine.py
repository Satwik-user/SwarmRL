"""
consensus_engine.py  --  Owner: Somnath Saha
Distributed Bayesian survivor-confirmation voting.

IDEA (beginner version):
  Each drone's sensor says "I'm X% sure a survivor is here". One drone can be
  wrong (false alarm). So drones SHARE their reports and COMBINE them with
  Bayes' rule. Reports that are close together (within `merge_radius` metres)
  are treated as being about the SAME place. When the combined confidence of a
  place reaches 85% (and >= 2 different drones contributed), the survivor
  location is "locked" and becomes a priority target.

Needs: Python 3.9+ and numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


EPS = 1e-6  # keeps probabilities away from exactly 0 or 1 (log(0) would crash)
Vec3 = Tuple[float, float, float]


def logit(p: float) -> float:
    """Probability -> log-odds. 0.5 -> 0, >0.5 -> positive, <0.5 -> negative.
    In log-odds space, independent evidence is combined by simple ADDITION."""
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Log-odds -> probability (inverse of logit), numerically stable."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Observation:
    """One sensor report from one drone about one place."""
    drone_id: int
    sensor: str                 # "thermal", "audio", "vision" ...
    position: Vec3              # suspected survivor position (x, y, z) in metres
    # >>> NEEDS FROM SATWIK/BHARGAV: someone must write the simulated thermal/audio
    # >>> sensors that turn "what the drone senses" into this 0..1 confidence.
    confidence: float           # sensor's own P(survivor), in [0, 1]
    timestamp: int              # simulation step of the report


def _newer_key(o: Observation):
    """Total order used to decide which of two reports wins (deterministic on every drone)."""
    return (o.timestamp, o.confidence, o.position)


class ConsensusEngine:
    """
    One instance lives ON EACH DRONE. Drones exchange `export_state()` lists over
    the mesh and call `merge()`. Merging the same data twice changes nothing
    (idempotent), so duplicated packets are harmless.
    """

    # >>> YOU (SOMNATH) MUST: tune prior, merge_radius, max_llr by experiment
    # >>> (benchmark_fault_tolerance.py has a prior sweep) and write the chosen
    # >>> values + reasons in your progress log.
    def __init__(
        self,
        lock_threshold: float = 0.85,  # confidence needed to lock a survivor (project spec)
        prior: float = 0.5,            # belief BEFORE any sensor (0.5 = no opinion)
        merge_radius: float = 3.0,     # metres: reports closer than this are "the same place"
        min_voters: int = 2,           # need >= this many DIFFERENT drones before locking
        max_llr: float = 3.0,          # cap on one report's evidence (limits a faulty sensor)
        cell_size: float = 5.0,        # metres; ONLY used for stable task ids (cell_of)
        time_window: int = 200,        # steps: reports older than this (vs the newest report of
                                       # the same place) are ignored, so scattered false alarms
                                       # from different times cannot add up to a fake survivor
    ):
        assert 0.5 <= lock_threshold < 1.0 and 0.0 < prior < 1.0
        assert merge_radius > 0 and min_voters >= 1 and max_llr > 0 and cell_size > 0
        self.lock_threshold = lock_threshold
        self.prior = prior
        self.merge_radius = merge_radius
        self.min_voters = min_voters
        self.max_llr = max_llr
        self.cell_size = cell_size
        self.time_window = time_window
        self.obs: List[Observation] = []        # newest report per (drone, sensor, place)
        self._locked_pts: List[Vec3] = []       # positions frozen at the moment of locking
        self._cache: Optional[list] = None      # cached clusters (cleared on any change)

    # ----------------------------------------------------------- input -----
    def add_observation(self, ob: Observation) -> None:
        """Record a local sensor report (or one received from a peer)."""
        if not (0.0 <= ob.confidence <= 1.0):
            raise ValueError("confidence must be within [0, 1]")
        for k, old in enumerate(self.obs):
            same_source = old.drone_id == ob.drone_id and old.sensor == ob.sensor
            if same_source and math.dist(old.position, ob.position) <= self.merge_radius:
                # Same drone + same sensor + same place: keep ONLY the newest, so one
                # drone cannot "vote many times" and inflate the confidence.
                if _newer_key(ob) > _newer_key(old):
                    self.obs[k] = ob
                    self._cache = None
                return
        self.obs.append(ob)
        self._cache = None

    # ----------------------------------------------------- the maths -------
    def _evidence(self, o: Observation, base: float) -> float:
        """How far one report moves belief away from the prior (log-odds), clipped."""
        e = logit(o.confidence) - base
        return max(-self.max_llr, min(self.max_llr, e))

    def _build_clusters(self) -> list:
        """Group nearby reports (anchor clustering, see below) and score each group.
        Posterior log-odds = logit(prior) + sum of clipped evidence.
        ASSUMPTION: different sensors/drones err independently (naive Bayes)."""
        if self._cache is not None:
            return self._cache
        # ANCHOR clustering: strongest reports first; each report joins the nearest
        # existing anchor within merge_radius, else becomes a new anchor. (Plain
        # "chain" clustering lets a stray false alarm link two groups and drag the
        # centre away, which produced false locks in testing.) The result depends only
        # on the SET of reports, not on the order they arrived in.
        order = sorted(self.obs, key=lambda o: (-o.confidence, o.timestamp, o.drone_id, o.sensor, o.position))
        anchors: List[Vec3] = []
        groups_l: List[List[Observation]] = []
        for o in order:
            best, bd = -1, self.merge_radius
            for gi, ap in enumerate(anchors):
                d = math.dist(ap, o.position)
                if d <= bd:
                    best, bd = gi, d
            if best < 0:
                anchors.append(o.position)
                groups_l.append([o])
            else:
                groups_l[best].append(o)
        groups = dict(enumerate(groups_l))
        base = logit(self.prior)
        clusters = []
        for members in groups.values():
            latest: Dict[Tuple[int, str], Observation] = {}      # newest per (drone, sensor)
            for o in members:
                k = (o.drone_id, o.sensor)
                if k not in latest or _newer_key(o) > _newer_key(latest[k]):
                    latest[k] = o
            tmax = max(o.timestamp for o in latest.values())
            used = sorted((o for o in latest.values() if o.timestamp >= tmax - self.time_window),
                          key=_newer_key)                      # drop stale reports (see time_window)
            conf = sigmoid(base + sum(self._evidence(o, base) for o in used))
            w = [max(o.confidence, EPS) for o in used]
            cen = tuple(sum(wi * o.position[k] for wi, o in zip(w, used)) / sum(w) for k in range(3))
            voters = len({o.drone_id for o in used})
            if voters >= self.min_voters and conf >= self.lock_threshold:
                if not any(math.dist(cen, p) <= self.merge_radius for p in self._locked_pts):
                    self._locked_pts.append(cen)                  # freeze: once locked, stay locked
            frozen = next((p for p in self._locked_pts if math.dist(cen, p) <= self.merge_radius), None)
            clusters.append({"centroid": cen, "conf": conf, "voters": voters, "locked_pos": frozen})
        clusters.sort(key=lambda c: c["centroid"])                # deterministic order
        self._cache = clusters
        return clusters

    # ------------------------------------------------------- queries -------
    def confidence_at(self, position: Vec3) -> float:
        """Combined confidence of the place nearest to `position` (within merge_radius), else the prior."""
        best, bd = self.prior, self.merge_radius
        for c in self._build_clusters():
            d = math.dist(c["centroid"], position)
            if d <= bd:
                best, bd = c["conf"], d
        return best

    def locked_survivors(self) -> List[Tuple[Vec3, float]]:
        """[(position, confidence)] of confirmed survivors. The position is frozen at the
        moment of locking so that task ids built from it never change later."""
        return [(c["locked_pos"], c["conf"]) for c in self._build_clusters() if c["locked_pos"] is not None]

    def unlocked_candidates(self, min_confidence: float = 0.55) -> List[Tuple[Vec3, float]]:
        """[(position, confidence)] of suspected-but-unconfirmed places with confidence >=
        min_confidence. The task allocator turns these into 'go and verify' tasks."""
        return [(c["centroid"], c["conf"]) for c in self._build_clusters()
                if c["locked_pos"] is None and c["conf"] >= min_confidence]

    def cell_of(self, pos) -> Tuple[int, int, int]:
        """Coarse grid cell of a position; used ONLY to make stable task ids."""
        return tuple(int(math.floor(c / self.cell_size)) for c in pos)

    # --------------------------------------------------- mesh exchange -----
    # >>> NEEDS FROM CHANDAN: his comm_mesh.py must carry this list to nearby drones
    # >>> (with packet loss) and hand received lists to merge().
    # >>> YOU (SOMNATH) MUST: call export_state() every few steps from the drone loop.
    def export_state(self) -> List[Observation]:
        """Everything this drone knows -> a list of observations to broadcast."""
        return list(self.obs)

    def merge(self, observations: List[Observation]) -> None:
        """Absorb a peer's broadcast. Safe to call repeatedly with the same data."""
        for ob in observations:
            self.add_observation(ob)