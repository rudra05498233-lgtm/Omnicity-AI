"""
╔══════════════════════════════════════════════════════════════════════╗
║          OmniCity AI  —  Digital Twin Simulation Engine             ║
║          v2.0.0  —  Autonomous Urban Operating System               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Architecture:                                                       ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │  SpatialSkeleton ←→ PowerMesh ←→ WaterVeinNetwork           │   ║
║  │        ↑                                                      │   ║
║  │  AgentPopulation → IncidentEngine → OperationalAPI           │   ║
║  │        ↓                                                      │   ║
║  │  CV-4/5/6 Sentinel ──► ML-1 RouteOptimizer ──► GreenWave    │   ║
║  │  CV-7/8 ANPR+Density ──► TrafficDensityField                 │   ║
║  │           SimulationLoop (heartbeat, shared state)            │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ──────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONSTANTS & LOGGING
# ──────────────────────────────────────────────────────────────────────────────

GRID_ROWS             = 20
GRID_COLS             = 20
NUM_CITIZENS          = 500
TICKS_PER_DAY         = 1_440
WORK_START_TICK       = 480
WORK_END_TICK         = 1_020
INCIDENT_ROLL_INTERVAL = 100
NUM_POWER_PLANTS      = 3
NUM_SUBSTATIONS       = 12
CASCADE_PROB          = 0.40
LOG_FILE              = Path("omnicityai_log.jsonl")

# CV/ML thresholds
CV4_WEAPON_CONFIDENCE  = 0.85   # minimum score to generate an incident packet
CV5_FIRE_CONFIDENCE    = 0.80
CV6_ACCIDENT_CONFIDENCE = 0.75
CONGESTION_THRESHOLD   = 0.70   # density > this → ML-1 reroutes
ML1_SCAN_INTERVAL      = 5      # ticks between route-optimizer cycles

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
log = logging.getLogger("omnicityai")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SPATIAL SKELETON
# ──────────────────────────────────────────────────────────────────────────────

class SignalState(Enum):
    RED    = auto()
    YELLOW = auto()
    GREEN  = auto()


class SmartSignal:
    """Timed state-machine traffic signal at an intersection."""

    CYCLE = {
        SignalState.GREEN:  (SignalState.YELLOW, 45),
        SignalState.YELLOW: (SignalState.RED,    10),
        SignalState.RED:    (SignalState.GREEN,  45),
    }

    def __init__(self, initial_state: Optional[SignalState] = None) -> None:
        self.state = initial_state or random.choice(list(SignalState))
        _, duration = self.CYCLE[self.state]
        self._ticks_remaining = random.randint(1, duration)
        self.forced_green    = False
        self._force_countdown = 0

    def tick(self) -> None:
        if self.forced_green:
            # Count down the force override; release when expired
            self._force_countdown -= 1
            if self._force_countdown <= 0:
                self.forced_green = False
            self.state = SignalState.GREEN
            return
        self._ticks_remaining -= 1
        if self._ticks_remaining <= 0:
            next_state, duration = self.CYCLE[self.state]
            self.state            = next_state
            self._ticks_remaining = duration

    def force_green(self, duration_ticks: int = 120) -> None:
        self.forced_green     = True
        self._force_countdown = duration_ticks
        self.state            = SignalState.GREEN

    def release_force(self) -> None:
        self.forced_green     = False
        self._force_countdown = 0

    @property
    def passable(self) -> bool:
        return self.state in (SignalState.GREEN, SignalState.YELLOW)

    def __repr__(self) -> str:
        return f"Signal({self.state.name}{'★' if self.forced_green else ''})"


class RoadStatus(Enum):
    OPEN    = "Open"
    BLOCKED = "Blocked"


class SpatialSkeleton:
    """
    2-D grid city map. Nodes = intersections. Edges = roads.
    Each edge carries a normalised `density_score` (0.0–1.0) derived
    from CV-7 vehicle counts, used by ML-1 for real-time rerouting.
    """

    def __init__(self, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.G: nx.Graph = nx.grid_2d_graph(rows, cols)
        self._annotate_nodes()
        self._annotate_edges()

    def node_id(self, r: int, c: int) -> Tuple[int, int]:
        return (r, c)

    def random_node(self) -> Tuple[int, int]:
        return random.choice(list(self.G.nodes))

    def coords_of(self, node: Tuple[int, int]) -> Tuple[float, float]:
        lat = 25.317 + node[0] * 0.003   # Varanasi latitude anchor
        lon = 82.973 + node[1] * 0.003
        return round(lat, 6), round(lon, 6)

    def _annotate_nodes(self) -> None:
        for node in self.G.nodes:
            self.G.nodes[node].update({
                "signal":   SmartSignal(),
                "zone":     random.choice(["Residential","Commercial","Industrial","Park"]),
                "cameras":  random.randint(0, 3),
                "entities": [],
                "events":   [],
            })

    def _annotate_edges(self) -> None:
        for u, v in self.G.edges:
            cap = random.randint(100, 1_000)
            self.G[u][v].update({
                "latency":         random.randint(5, 50),
                "bandwidth":       cap,
                "status":          RoadStatus.OPEN,
                "traffic_density": 0,       # raw vehicle count (CV-7)
                "density_score":   0.0,     # normalised 0–1 for ML-1
                "length_m":        random.randint(80, 400),
            })

    # ── path-finding ─────────────────────────────────────────────────────────

    def shortest_path(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int],
        weight_fn=None,
    ) -> List[Tuple[int, int]]:
        def _default_weight(u, v, d):
            if d["status"] == RoadStatus.BLOCKED:
                return 1e9
            return d["latency"] * (1 + d["density_score"] * 3)   # density-penalised
        try:
            return nx.dijkstra_path(
                self.G, source, target,
                weight=weight_fn or _default_weight,
            )
        except nx.NetworkXNoPath:
            return []

    def tick_signals(self) -> None:
        for node in self.G.nodes:
            self.G.nodes[node]["signal"].tick()

    def open_edges(self) -> List[Tuple]:
        return [(u, v) for u, v, d in self.G.edges(data=True)
                if d["status"] == RoadStatus.OPEN]

    def block_edge(self, u, v) -> None:
        if self.G.has_edge(u, v):
            self.G[u][v]["status"] = RoadStatus.BLOCKED

    def unblock_edge(self, u, v) -> None:
        if self.G.has_edge(u, v):
            self.G[u][v]["status"] = RoadStatus.OPEN

    def avg_traffic_density(self) -> float:
        scores = [d["density_score"] for _, _, d in self.G.edges(data=True)]
        return round(sum(scores) / max(len(scores), 1), 3)

    def high_density_edges(self, threshold: float = CONGESTION_THRESHOLD) -> List[Tuple]:
        return [(u, v) for u, v, d in self.G.edges(data=True)
                if d["density_score"] >= threshold]

    def edge_density_map(self) -> Dict[str, float]:
        """Serialisable snapshot of every edge density — fed to the frontend map."""
        result = {}
        for u, v, d in self.G.edges(data=True):
            key = f"{u[0]},{u[1]}-{v[0]},{v[1]}"
            result[key] = round(d["density_score"], 3)
        return result

    def signal_state_map(self) -> Dict[str, str]:
        """Serialisable snapshot of every intersection signal — for the live map."""
        return {
            f"{n[0]},{n[1]}": self.G.nodes[n]["signal"].state.name
            for n in self.G.nodes
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  AGENT-BASED POPULATION  (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(Enum):
    AT_HOME   = "at_home"
    COMMUTING = "commuting"
    AT_WORK   = "at_work"
    RETURNING = "returning"
    IDLE      = "idle"


@dataclass
class Citizen:
    uid:          uuid.UUID
    home_node:    Tuple[int, int]
    work_node:    Tuple[int, int]
    trust_score:  float      = field(default_factory=lambda: round(random.uniform(0.3, 1.0), 2))
    state:        AgentState = AgentState.AT_HOME
    current_node: Optional[Tuple] = None
    path:         List[Tuple]     = field(default_factory=list)
    path_index:   int             = 0
    speed:        int             = field(default_factory=lambda: random.randint(1, 3))
    in_failure_zone: bool         = False

    def __post_init__(self):
        self.current_node = self.home_node

    def start_commute_to_work(self, skeleton: SpatialSkeleton) -> None:
        if self.state in (AgentState.AT_HOME, AgentState.IDLE):
            self.path       = skeleton.shortest_path(self.current_node, self.work_node)
            self.path_index = 0
            self.state      = AgentState.COMMUTING

    def start_commute_home(self, skeleton: SpatialSkeleton) -> None:
        if self.state in (AgentState.AT_WORK, AgentState.IDLE):
            self.path       = skeleton.shortest_path(self.current_node, self.home_node)
            self.path_index = 0
            self.state      = AgentState.RETURNING

    def tick(self, skeleton: SpatialSkeleton) -> None:
        if not self.path or self.state in (AgentState.AT_HOME, AgentState.AT_WORK):
            return
        for _ in range(self.speed):
            if self.path_index >= len(self.path) - 1:
                self._arrive(skeleton)
                break
            self._step(skeleton)

    def _step(self, skeleton: SpatialSkeleton) -> None:
        prev = self.path[self.path_index]
        nxt  = self.path[self.path_index + 1]
        if not skeleton.G.has_edge(prev, nxt):
            self.state = AgentState.IDLE
            return
        edge = skeleton.G[prev][nxt]
        if edge["status"] == RoadStatus.BLOCKED:
            self.state = AgentState.IDLE
            return
        edge["traffic_density"] = max(0, edge["traffic_density"] - 1)
        self.path_index        += 1
        self.current_node       = nxt
        skeleton.G[prev][nxt]["traffic_density"] += 1

    def _arrive(self, skeleton: SpatialSkeleton) -> None:
        if   self.state == AgentState.COMMUTING: self.current_node = self.work_node;  self.state = AgentState.AT_WORK
        elif self.state == AgentState.RETURNING: self.current_node = self.home_node;  self.state = AgentState.AT_HOME
        else:                                                                           self.state = AgentState.IDLE
        self.path = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid":        str(self.uid),
            "state":      self.state.value,
            "node":       self.current_node,
            "trust":      self.trust_score,
            "in_failure": self.in_failure_zone,
        }


class AgentPopulation:
    def __init__(self, skeleton: SpatialSkeleton, count: int = NUM_CITIZENS) -> None:
        self.skeleton = skeleton
        self.agents: List[Citizen] = []
        self._spawn(count)

    def _spawn(self, count: int) -> None:
        nodes = list(self.skeleton.G.nodes)
        for _ in range(count):
            home = random.choice(nodes)
            work = random.choice(nodes)
            while work == home:
                work = random.choice(nodes)
            self.agents.append(Citizen(uid=uuid.uuid4(), home_node=home, work_node=work))
        print(f"  [Population]  Spawned {count} citizens.")

    def tick(self, sim_tick: int) -> None:
        day_tick = sim_tick % TICKS_PER_DAY
        for agent in self.agents:
            if day_tick == WORK_START_TICK:  agent.start_commute_to_work(self.skeleton)
            elif day_tick == WORK_END_TICK:  agent.start_commute_home(self.skeleton)
            if agent.state == AgentState.IDLE:
                if day_tick < WORK_END_TICK: agent.start_commute_to_work(self.skeleton)
                else:                        agent.start_commute_home(self.skeleton)
            agent.tick(self.skeleton)

    def agents_at_node(self, node: Tuple) -> List[Citizen]:
        return [a for a in self.agents if a.current_node == node]

    def active_count(self) -> int:
        return sum(1 for a in self.agents if a.state in (AgentState.COMMUTING, AgentState.RETURNING))

    def agents_in_nodes(self, node_set: set) -> List[Citizen]:
        return [a for a in self.agents if a.current_node in node_set]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  INFRASTRUCTURE (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────

class PowerNodeType(Enum):
    PLANT      = "Plant"
    SUBSTATION = "Substation"


@dataclass
class PowerNode:
    nid:      str
    kind:     PowerNodeType
    capacity: float
    load:     float = 0.0
    online:   bool  = True

    @property
    def utilisation(self) -> float: return self.load / max(self.capacity, 0.001)
    @property
    def overloaded(self) -> bool:   return self.utilisation > 1.0


class PowerMesh:
    def __init__(self, skeleton: SpatialSkeleton) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._build(skeleton)

    def _build(self, skeleton: SpatialSkeleton) -> None:
        nodes = list(skeleton.G.nodes)
        for i in range(NUM_POWER_PLANTS):
            pid = f"PLANT-{i}"
            self.G.add_node(pid, power_node=PowerNode(pid, PowerNodeType.PLANT, 500.0, random.uniform(200, 400), True))
        for i in range(NUM_SUBSTATIONS):
            sid = f"SUB-{i}"
            geo = random.choice(nodes)
            self.G.add_node(sid, power_node=PowerNode(sid, PowerNodeType.SUBSTATION, 100.0, random.uniform(40, 95), True), geo_node=geo)
            self.G.add_edge(f"PLANT-{i % NUM_POWER_PLANTS}", sid, capacity_mw=150)
        subs = [n for n in self.G.nodes if n.startswith("SUB")]
        random.shuffle(subs)
        for i in range(0, len(subs)-1, 3):
            self.G.add_edge(subs[i], subs[i+1], capacity_mw=80)
        print(f"  [PowerMesh]   {NUM_POWER_PLANTS} plants, {NUM_SUBSTATIONS} substations wired.")

    def cascade_failure(self, origin_nid: str) -> List[str]:
        blacked, queue, visited = [], [origin_nid], set()
        while queue:
            nid = queue.pop()
            if nid in visited: continue
            visited.add(nid)
            self.G.nodes[nid]["power_node"].online = False
            blacked.append(nid)
            for nb in self.G.successors(nid):
                if nb not in visited and random.random() < CASCADE_PROB:
                    queue.append(nb)
        return blacked

    def tick(self) -> None:
        for nid in self.G.nodes:
            pn: PowerNode = self.G.nodes[nid]["power_node"]
            if not pn.online:
                if random.random() < 0.01: pn.online = True; pn.load = pn.capacity * 0.5
                continue
            pn.load += random.uniform(-5, 8)
            pn.load  = max(0, min(pn.load, pn.capacity * 1.3))
            if pn.overloaded and random.random() < 0.05:
                self.cascade_failure(nid)

    def grid_health(self) -> float:
        total  = len(self.G.nodes)
        online = sum(1 for n in self.G.nodes if self.G.nodes[n]["power_node"].online)
        return round(online / max(total, 1) * 100, 1)

    def offline_geo_nodes(self) -> set:
        return {self.G.nodes[n]["geo_node"] for n in self.G.nodes
                if not self.G.nodes[n]["power_node"].online and "geo_node" in self.G.nodes[n]}


@dataclass
class WaterEdge:
    source: str; target: str; flow: float = 100.0; burst: bool = False


class WaterVeinNetwork:
    RESERVOIR_COUNT    = 2
    PUMPING_STATIONS   = 6
    DISTRIBUTION_ZONES = 20

    def __init__(self, skeleton: SpatialSkeleton) -> None:
        self.G: nx.DiGraph          = nx.DiGraph()
        self._pressure: Dict[str, float] = {}
        self._build(skeleton)

    def _build(self, skeleton: SpatialSkeleton) -> None:
        nodes = list(skeleton.G.nodes)
        for i in range(self.RESERVOIR_COUNT):
            rid = f"RES-{i}"; self.G.add_node(rid, type="Reservoir", pressure=10.0); self._pressure[rid] = 10.0
        for i in range(self.PUMPING_STATIONS):
            pid = f"PUMP-{i}"; geo = random.choice(nodes)
            self.G.add_node(pid, type="Pump", pressure=8.0, geo_node=geo); self._pressure[pid] = 8.0
            self.G.add_edge(f"RES-{i % self.RESERVOIR_COUNT}", pid, flow=random.uniform(80, 120), burst=False)
        for i in range(self.DISTRIBUTION_ZONES):
            did = f"DIST-{i}"; geo = random.choice(nodes)
            self.G.add_node(did, type="Distribution", pressure=5.0, geo_node=geo); self._pressure[did] = 5.0
            self.G.add_edge(f"PUMP-{i % self.PUMPING_STATIONS}", did, flow=random.uniform(30, 80), burst=False)
        print(f"  [WaterVeins]  {self.RESERVOIR_COUNT} reservoirs → {self.PUMPING_STATIONS} pumps → {self.DISTRIBUTION_ZONES} zones.")

    def burst_pipe(self, source: str, target: str) -> List[str]:
        if not self.G.has_edge(source, target): return []
        self.G[source][target]["burst"] = True
        affected = list(nx.descendants(self.G, target)); affected.insert(0, target)
        for nid in affected: self._pressure[nid] = 0.0
        return affected

    def repair_pipe(self, source: str, target: str) -> None:
        if self.G.has_edge(source, target):
            self.G[source][target]["burst"] = False
            for nid in nx.descendants(self.G, target): self._pressure[nid] = random.uniform(3.0, 6.0)

    def tick(self) -> None:
        for u, v, d in self.G.edges(data=True):
            if not d["burst"] and random.random() < 0.001: self.burst_pipe(u, v)
            elif not d["burst"]: d["flow"] = max(0.0, d["flow"] + random.uniform(-2, 2))

    def zero_pressure_geo_nodes(self) -> set:
        return {self.G.nodes[n]["geo_node"] for n in self.G.nodes
                if self._pressure.get(n, 5.0) == 0.0 and "geo_node" in self.G.nodes[n]}

    def network_health(self) -> float:
        total = self.G.number_of_edges()
        ok    = sum(1 for _, _, d in self.G.edges(data=True) if not d["burst"])
        return round(ok / max(total, 1) * 100, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CV PIPELINE SIMULATORS  (CV-4, CV-5, CV-6, CV-7, CV-8)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CVDetection:
    """A single detection event from a simulated CV model pass."""
    model:      str
    node:       Tuple[int, int]
    class_name: str
    confidence: float
    tick:       int
    sensor_corr: bool = False  # True when secondary sensor confirmed (Anti-Chaos)


class CVPipelineSimulator:
    """
    Simulates the CV model inference loop over the city's camera network.
    Runs continuously inside the simulation tick, scanning a random subset
    of camera nodes each cycle.
    """

    # How many camera nodes to sample per tick
    SAMPLE_SIZE = 8

    def __init__(self, skeleton: SpatialSkeleton) -> None:
        self.skeleton  = skeleton
        self._detections: deque[CVDetection] = deque(maxlen=200)

    # ── CV-4: Weapon / Threat Detection ───────────────────────────────────────

    def run_cv4(self, tick: int) -> List[CVDetection]:
        """
        Scans sampled camera nodes for firearm/threat objects.
        Only generates a detection if the simulated model confidence ≥ 0.85
        AND a secondary microphone spike is correlated (Anti-Chaos).
        Probability of a genuine threat at any given camera: 0.3%.
        """
        detections = []
        sample = random.sample(list(self.skeleton.G.nodes), min(self.SAMPLE_SIZE, self.skeleton.G.number_of_nodes()))
        for node in sample:
            cams = self.skeleton.G.nodes[node]["cameras"]
            if cams == 0: continue
            # Base threat probability is very low
            if random.random() > 0.003 * cams: continue
            confidence = random.uniform(0.82, 0.98)
            if confidence < CV4_WEAPON_CONFIDENCE: continue
            # Anti-Chaos: require corroborating audio sensor
            audio_spike = random.random() < 0.75
            det = CVDetection(
                model="CV-4", node=node, class_name="FIREARM",
                confidence=round(confidence, 3), tick=tick, sensor_corr=audio_spike,
            )
            detections.append(det)
            self._detections.appendleft(det)
        return detections

    # ── CV-5: Fire / Smoke Detection ──────────────────────────────────────────

    def run_cv5(self, tick: int) -> List[CVDetection]:
        """
        Thermal + smoke classification. Validated against physical thermal
        sensor array (simulated). Fires only on dual confirmation.
        """
        detections = []
        sample = random.sample(list(self.skeleton.G.nodes), min(self.SAMPLE_SIZE, self.skeleton.G.number_of_nodes()))
        for node in sample:
            if random.random() > 0.002: continue
            confidence = random.uniform(0.76, 0.97)
            if confidence < CV5_FIRE_CONFIDENCE: continue
            # Thermal sensor corroboration
            thermal_spike = random.random() < 0.70
            det = CVDetection(
                model="CV-5", node=node, class_name="FIRE_SMOKE",
                confidence=round(confidence, 3), tick=tick, sensor_corr=thermal_spike,
            )
            detections.append(det)
            self._detections.appendleft(det)
        return detections

    # ── CV-6: Traffic Accident Detection ──────────────────────────────────────

    def run_cv6(self, tick: int, high_density_edges: List[Tuple]) -> List[CVDetection]:
        """
        Accident classification over high-density road segments.
        Proximity sensor array validates the detection (Anti-Chaos).
        """
        detections = []
        if not high_density_edges: return detections
        sample_edges = random.sample(high_density_edges, min(3, len(high_density_edges)))
        for u, v in sample_edges:
            if random.random() > 0.008: continue
            confidence = random.uniform(0.70, 0.95)
            if confidence < CV6_ACCIDENT_CONFIDENCE: continue
            prox_sensor = random.random() < 0.65
            det = CVDetection(
                model="CV-6", node=u, class_name="TRAFFIC_ACCIDENT",
                confidence=round(confidence, 3), tick=tick, sensor_corr=prox_sensor,
            )
            detections.append(det)
            self._detections.appendleft(det)
        return detections

    # ── CV-7: Vehicle Counting / Density Field Update ─────────────────────────

    def run_cv7(self, tick: int) -> None:
        """
        Counts vehicles at sampled intersections. Updates the density_score
        on each adjacent edge, which ML-1 consumes every ML1_SCAN_INTERVAL ticks.
        Raw traffic_density is set by AgentPopulation; CV-7 converts it to a
        normalised 0–1 density_score with a Gaussian noise overlay.
        """
        for u, v, d in self.skeleton.G.edges(data=True):
            raw   = d["traffic_density"]
            cap   = max(d["bandwidth"], 1)
            score = raw / cap
            # Add slight CV detection noise
            score += random.gauss(0, 0.015)
            d["density_score"] = round(max(0.0, min(1.0, score)), 3)

    # ── CV-8: ANPR / Number Plate OCR ─────────────────────────────────────────

    def run_cv8(self, tick: int, vehicle_db_snapshot: List[str]) -> List[Dict]:
        """
        Scans a random sample of camera nodes for vehicle plates.
        Matches against the vehicle registry to flag unregistered vehicles
        or wanted plate associations.
        Returns a list of plate-scan events (for audit log / ledger).
        """
        events = []
        sample = random.sample(list(self.skeleton.G.nodes), min(4, self.skeleton.G.number_of_nodes()))
        for node in sample:
            cams = self.skeleton.G.nodes[node]["cameras"]
            if cams == 0: continue
            # Simulate detecting a plate
            if random.random() > 0.15: continue
            plate     = random.choice(vehicle_db_snapshot) if vehicle_db_snapshot else None
            match_ok  = plate is not None and random.random() < 0.92
            events.append({
                "tick":        tick,
                "node":        list(node),
                "plate_read":  plate or "UNREADABLE",
                "registry_hit": match_ok,
                "flagged":     not match_ok,
            })
        return events

    def recent_detections(self, limit: int = 30) -> List[Dict]:
        return [
            {
                "model":       d.model,
                "node":        list(d.node),
                "class_name":  d.class_name,
                "confidence":  d.confidence,
                "tick":        d.tick,
                "sensor_corr": d.sensor_corr,
                "gps":         list(self.skeleton.coords_of(d.node)),
            }
            for d in list(self._detections)[:limit]
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ML-1: ROUTE OPTIMIZER / GREEN-WAVE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GreenWaveEvent:
    wave_id:      str
    tick_created: int
    origin_node:  Tuple[int, int]
    dest_node:    Tuple[int, int]
    path:         List[Tuple[int, int]]
    trigger:      str      # "EMERGENCY" | "CONGESTION_RELIEF"
    vehicle_type: str      # "AMBULANCE" | "FIRE" | "POLICE" | "AUTO"
    signals_forced: int
    active:       bool = True
    tick_resolved: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "wave_id":       self.wave_id,
            "tick_created":  self.tick_created,
            "origin_node":   list(self.origin_node),
            "dest_node":     list(self.dest_node),
            "path_length":   len(self.path),
            "path_nodes":    [list(n) for n in self.path[:8]],  # truncated for API
            "trigger":       self.trigger,
            "vehicle_type":  self.vehicle_type,
            "signals_forced": self.signals_forced,
            "active":        self.active,
        }


class ML1RouteOptimizer:
    """
    Continuous traffic management brain.

    Two operating modes:
    ─────────────────────────────────────────────────────────────────────
    A. ANTI-JAM LOOP  (runs every ML1_SCAN_INTERVAL ticks)
       Reads CV-7 density_score on every edge.
       When density > CONGESTION_THRESHOLD on a segment, it:
         1. Extends the GREEN phase of the upstream signal to flush the queue.
         2. Identifies an alternate parallel path and redistributes density
            by temporarily penalising the congested edge in the weight fn.

    B. EMERGENCY OVERRIDE  (triggered by IncidentEngine or authority dispatch)
       Given an origin and destination node of an emergency vehicle:
         1. Runs density-weighted A* across the SpatialSkeleton.
         2. Forces JIT GREEN on every signal along the computed corridor.
         3. Duration = estimated traversal ticks + 30-tick buffer.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, skeleton: SpatialSkeleton) -> None:
        self.skeleton         = skeleton
        self._active_waves:   List[GreenWaveEvent] = []
        self._wave_history:   deque[GreenWaveEvent] = deque(maxlen=50)
        self._jam_relief_log: deque[Dict] = deque(maxlen=100)

    # ── A. Anti-Jam Loop ──────────────────────────────────────────────────────

    def run_anti_jam(self, tick: int) -> List[Dict]:
        """
        Called every ML1_SCAN_INTERVAL ticks.
        Returns a list of relief events that occurred this cycle.
        """
        events      = []
        jammed_edges = self.skeleton.high_density_edges(CONGESTION_THRESHOLD)

        for u, v in jammed_edges:
            score = self.skeleton.G[u][v]["density_score"]

            # 1. Extend upstream green signal
            sig_u: SmartSignal = self.skeleton.G.nodes[u]["signal"]
            if not sig_u.forced_green:
                sig_u.force_green(duration_ticks=25)

            # 2. Find alternate route bypassing this edge
            # Temporarily set this edge weight to near-impassable
            original_latency = self.skeleton.G[u][v]["latency"]
            self.skeleton.G[u][v]["latency"] = 9999

            # Reroute vehicles that would have used u→v
            alt_path = self.skeleton.shortest_path(u, v)

            # Restore
            self.skeleton.G[u][v]["latency"] = original_latency

            event = {
                "tick":          tick,
                "segment":       [list(u), list(v)],
                "density_score": score,
                "action":        "SIGNAL_EXTENDED + ALT_ROUTE",
                "alt_hops":      len(alt_path),
            }
            events.append(event)
            self._jam_relief_log.appendleft(event)

        return events

    # ── B. Emergency Override ─────────────────────────────────────────────────

    def trigger_emergency_wave(
        self,
        origin:       Tuple[int, int],
        destination:  Tuple[int, int],
        vehicle_type: str,
        tick:         int,
        trigger:      str = "EMERGENCY",
    ) -> Optional[GreenWaveEvent]:
        """
        Computes the optimal route for an emergency vehicle and forces
        JIT green on every intersection node along that corridor.
        Returns the GreenWaveEvent or None if no path exists.
        """
        # Weight function: penalise density, ignore blocked roads for emergency
        def em_weight(u, v, d):
            return d["latency"] * (1 + d["density_score"])

        path = []
        try:
            path = nx.dijkstra_path(self.skeleton.G, origin, destination, weight=em_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        if len(path) < 2:
            return None

        # Estimate traversal time: avg 2 ticks per hop + buffer
        duration = len(path) * 2 + 30

        forced_count = 0
        for i, node in enumerate(path):
            if self.skeleton.G.has_node(node):
                sig: SmartSignal = self.skeleton.G.nodes[node]["signal"]
                # JIT: force green slightly ahead of ETA at this node
                jit_delay = max(0, i * 2 - 5)
                # For simplicity in the sim, we force immediately; in
                # production this would use a time-indexed scheduler.
                sig.force_green(duration_ticks=duration)
                forced_count += 1

        wave = GreenWaveEvent(
            wave_id       = f"GW-{uuid.uuid4().hex[:8].upper()}",
            tick_created  = tick,
            origin_node   = origin,
            dest_node     = destination,
            path          = path,
            trigger       = trigger,
            vehicle_type  = vehicle_type,
            signals_forced = forced_count,
        )
        self._active_waves.append(wave)
        self._wave_history.appendleft(wave)
        log.info("GREEN WAVE %s: %d nodes forced GREEN, path_len=%d", wave.wave_id, forced_count, len(path))
        return wave

    # ── Tick: expire completed waves ──────────────────────────────────────────

    def tick(self, current_tick: int) -> None:
        """Expire waves whose signals have self-released."""
        for wave in self._active_waves:
            if not wave.active:
                continue
            # Wave considered complete after estimated traversal time
            elapsed = current_tick - wave.tick_created
            if elapsed > len(wave.path) * 2 + 35:
                wave.active        = False
                wave.tick_resolved = current_tick

        self._active_waves = [w for w in self._active_waves if w.active]

    # ── API accessors ─────────────────────────────────────────────────────────

    def active_waves(self) -> List[Dict]:
        return [w.to_dict() for w in self._active_waves]

    def wave_history(self, limit: int = 20) -> List[Dict]:
        return [w.to_dict() for w in list(self._wave_history)[:limit]]

    def jam_relief_log(self, limit: int = 20) -> List[Dict]:
        return list(self._jam_relief_log)[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  INCIDENT ENGINE  (extended for autonomous dispatch)
# ──────────────────────────────────────────────────────────────────────────────

class IncidentType(Enum):
    ACCIDENT         = "Accident"
    THEFT            = "Crime_Theft"
    WEAPON_DETECTED  = "Crime_Weapon"
    FIRE_SMOKE       = "Fire_Smoke"
    PIPE_BURST       = "Infra_PipeBurst"
    GRID_BLACKOUT    = "Infra_GridBlackout"
    ROAD_DEBRIS      = "Accident_Debris"


class SeverityLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


# Authority routing table — incident type → responsible authority + vehicle type
AUTHORITY_DISPATCH = {
    IncidentType.WEAPON_DETECTED: ("Police",      "POLICE"),
    IncidentType.THEFT:           ("Police",      "POLICE"),
    IncidentType.ACCIDENT:        ("EMS",          "AMBULANCE"),
    IncidentType.FIRE_SMOKE:      ("Fire Brigade", "FIRE"),
    IncidentType.PIPE_BURST:      ("Jal Sansthan", "UTILITY"),
    IncidentType.GRID_BLACKOUT:   ("UPPCL",        "UTILITY"),
    IncidentType.ROAD_DEBRIS:     ("PWD",           "UTILITY"),
}


@dataclass
class IncidentPacket:
    incident_id:   str
    tick:          int
    sim_time:      str
    incident_type: IncidentType
    severity:      SeverityLevel
    gps:           Tuple[float, float]
    node:          Any
    edge:          Optional[Tuple]
    details:       Dict[str, Any]
    resolved:      bool = False
    # Autonomous dispatch fields
    authority:     str  = ""
    dispatch_tick: Optional[int] = None
    green_wave_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "incident_id":   self.incident_id,
            "tick":          self.tick,
            "sim_time":      self.sim_time,
            "type":          self.incident_type.value,
            "severity":      self.severity.name,
            "gps":           {"lat": self.gps[0], "lon": self.gps[1]},
            "node":          list(self.node) if self.node else None,
            "edge":          [list(self.edge[0]), list(self.edge[1])] if self.edge else None,
            "details":       self.details,
            "resolved":      self.resolved,
            "authority":     self.authority,
            "dispatch_tick": self.dispatch_tick,
            "green_wave_id": self.green_wave_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
        })

    def to_dict(self) -> Dict:
        d = json.loads(self.to_json())
        return d


class IncidentEngine:
    """
    Stochastic + CV-driven event generator.
    Also handles autonomous authority dispatch and triggers ML-1 for
    emergency Green Wave when severity is HIGH or CRITICAL.
    """

    def __init__(
        self,
        skeleton:    SpatialSkeleton,
        power_mesh:  PowerMesh,
        water_veins: WaterVeinNetwork,
        cv_pipeline: CVPipelineSimulator,
        ml1:         ML1RouteOptimizer,
    ) -> None:
        self.skeleton    = skeleton
        self.power_mesh  = power_mesh
        self.water_veins = water_veins
        self.cv          = cv_pipeline
        self.ml1         = ml1
        self.active_incidents: List[IncidentPacket] = []
        self._incident_log:    List[IncidentPacket] = []
        self._log_file = LOG_FILE.open("a")
        # Authority station nodes (random but fixed for the sim lifetime)
        nodes = list(skeleton.G.nodes)
        self._authority_nodes = {
            "Police":      random.choice(nodes),
            "EMS":         random.choice(nodes),
            "Fire Brigade": random.choice(nodes),
            "Jal Sansthan": random.choice(nodes),
            "UPPCL":       random.choice(nodes),
            "PWD":         random.choice(nodes),
        }

    # ── CV-driven incident generation ─────────────────────────────────────────

    def process_cv_detections(
        self,
        cv4_dets: List[CVDetection],
        cv5_dets: List[CVDetection],
        cv6_dets: List[CVDetection],
        tick:     int,
    ) -> List[IncidentPacket]:
        """
        Converts validated CV detections → IncidentPackets.
        Only detections where sensor_corr=True pass the Anti-Chaos gate.
        """
        new_incidents = []
        sim_time = self._tick_to_time(tick)

        for det in cv4_dets:
            if not det.sensor_corr: continue  # Anti-Chaos: dual-sensor required
            inc = self._make_incident(
                tick, sim_time, det.node, None,
                IncidentType.WEAPON_DETECTED, SeverityLevel.CRITICAL,
                {"cv_confidence": det.confidence, "model": "CV-4", "anti_chaos": True},
            )
            new_incidents.append(inc)

        for det in cv5_dets:
            if not det.sensor_corr: continue
            inc = self._make_incident(
                tick, sim_time, det.node, None,
                IncidentType.FIRE_SMOKE, SeverityLevel.HIGH,
                {"cv_confidence": det.confidence, "model": "CV-5", "anti_chaos": True},
            )
            new_incidents.append(inc)

        for det in cv6_dets:
            if not det.sensor_corr: continue
            inc = self._make_incident(
                tick, sim_time, det.node, None,
                IncidentType.ACCIDENT, SeverityLevel.HIGH,
                {"cv_confidence": det.confidence, "model": "CV-6", "anti_chaos": True},
            )
            new_incidents.append(inc)

        return new_incidents

    # ── Stochastic roll ───────────────────────────────────────────────────────

    def roll(self, tick: int) -> List[IncidentPacket]:
        new_incidents = []
        sim_time      = self._tick_to_time(tick)

        self.active_incidents = [i for i in self.active_incidents if tick - i.tick < 300]

        if random.random() < 0.35:
            inc = self._gen_accident(tick, sim_time)
            if inc: new_incidents.append(inc)
        if random.random() < 0.25:
            new_incidents.append(self._gen_crime(tick, sim_time))
        if random.random() < 0.20:
            inc = self._gen_infrastructure(tick, sim_time)
            if inc: new_incidents.append(inc)

        return new_incidents

    # ── Autonomous dispatch ───────────────────────────────────────────────────

    def _dispatch(self, inc: IncidentPacket, tick: int) -> None:
        """
        Autonomously routes the correct authority to the incident.
        For CRITICAL/HIGH severity, triggers ML-1 Green Wave from the
        nearest authority station to the incident node.
        """
        authority, vehicle_type = AUTHORITY_DISPATCH.get(
            inc.incident_type, ("Civic", "UTILITY")
        )
        inc.authority     = authority
        inc.dispatch_tick = tick

        # Green Wave for HIGH+ severity
        if inc.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL) and inc.node:
            origin      = self._authority_nodes.get(authority, self.skeleton.random_node())
            destination = tuple(inc.node) if isinstance(inc.node, list) else inc.node
            wave        = self.ml1.trigger_emergency_wave(
                origin, destination, vehicle_type, tick, trigger="EMERGENCY"
            )
            if wave:
                inc.green_wave_id = wave.wave_id

        log.info(
            "AUTO-DISPATCH: %s → %s | authority=%s | GW=%s",
            inc.incident_id, inc.incident_type.value, authority, inc.green_wave_id,
        )

    # ── Generators ────────────────────────────────────────────────────────────

    def _make_incident(
        self, tick, sim_time, node, edge,
        itype: IncidentType, severity: SeverityLevel, details: dict,
    ) -> IncidentPacket:
        gps = self.skeleton.coords_of(node) if node else (0.0, 0.0)
        inc = IncidentPacket(
            incident_id   = f"INC-{uuid.uuid4().hex[:8].upper()}",
            tick=tick, sim_time=sim_time, incident_type=itype, severity=severity,
            gps=gps, node=node, edge=edge, details=details,
        )
        self._commit(inc, tick)
        return inc

    def _commit(self, inc: IncidentPacket, tick: int) -> None:
        self.active_incidents.append(inc)
        self._incident_log.append(inc)
        self._log_file.write(inc.to_json() + "\n")
        self._log_file.flush()
        self._dispatch(inc, tick)

    def _gen_accident(self, tick: int, sim_time: str) -> Optional[IncidentPacket]:
        high = self.skeleton.high_density_edges(threshold=0.6)
        pool = high if high else self.skeleton.open_edges()
        if not pool: return None
        u, v = random.choice(pool)
        self.skeleton.block_edge(u, v)
        sev  = SeverityLevel.HIGH if self.skeleton.G[u][v]["density_score"] > 0.7 else SeverityLevel.MEDIUM
        return self._make_incident(tick, sim_time, u, (u, v), IncidentType.ACCIDENT, sev,
                                   {"density_score": self.skeleton.G[u][v]["density_score"], "road_blocked": True})

    def _gen_crime(self, tick: int, sim_time: str) -> IncidentPacket:
        node  = self.skeleton.random_node()
        ctype = random.choice([IncidentType.THEFT, IncidentType.WEAPON_DETECTED])
        sev   = SeverityLevel.CRITICAL if ctype == IncidentType.WEAPON_DETECTED else SeverityLevel.HIGH
        return self._make_incident(tick, sim_time, node, None, ctype, sev,
                                   {"zone": self.skeleton.G.nodes[node]["zone"],
                                    "cameras": self.skeleton.G.nodes[node]["cameras"]})

    def _gen_infrastructure(self, tick: int, sim_time: str) -> Optional[IncidentPacket]:
        kind = random.choice(["power", "water"])
        if kind == "power":
            online = [n for n in self.power_mesh.G.nodes if self.power_mesh.G.nodes[n]["power_node"].online]
            if not online: return None
            target  = random.choice(online)
            blacked = self.power_mesh.cascade_failure(target)
            geo     = self.power_mesh.G.nodes[target].get("geo_node", self.skeleton.random_node())
            sev     = SeverityLevel.CRITICAL if len(blacked) > 3 else SeverityLevel.HIGH
            return self._make_incident(tick, sim_time, geo, None, IncidentType.GRID_BLACKOUT, sev,
                                       {"origin_node": target, "blacked_out": blacked, "cascade_count": len(blacked)})
        else:
            edges = list(self.water_veins.G.edges)
            if not edges: return None
            src, tgt = random.choice(edges)
            affected = self.water_veins.burst_pipe(src, tgt)
            geo      = self.water_veins.G.nodes[tgt].get("geo_node", self.skeleton.random_node())
            sev      = SeverityLevel.HIGH if len(affected) > 5 else SeverityLevel.MEDIUM
            return self._make_incident(tick, sim_time, geo, None, IncidentType.PIPE_BURST, sev,
                                       {"burst_edge": [src, tgt], "zones_affected": affected, "downstream_count": len(affected)})

    @staticmethod
    def _tick_to_time(tick: int) -> str:
        d = tick % TICKS_PER_DAY; return f"{d//60:02d}:{d%60:02d}"

    def emergency_count(self) -> int: return len(self.active_incidents)

    def close(self) -> None: self._log_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  OPERATIONAL API  — shared live state for the FastAPI layer
# ──────────────────────────────────────────────────────────────────────────────

class OperationalAPI:
    """
    Public interface consumed by backend.py's FastAPI server.
    All methods return plain Python dicts/lists (JSON-serialisable).
    The FastAPI server runs in a thread and reads this object; the
    simulation loop writes to it.  No explicit locking is needed
    for CPython (GIL) but we keep mutations atomic.
    """

    def __init__(
        self,
        skeleton:    SpatialSkeleton,
        population:  AgentPopulation,
        power_mesh:  PowerMesh,
        water_veins: WaterVeinNetwork,
        incidents:   IncidentEngine,
        cv_pipeline: CVPipelineSimulator,
        ml1:         ML1RouteOptimizer,
    ) -> None:
        self.skeleton    = skeleton
        self.population  = population
        self.power_mesh  = power_mesh
        self.water_veins = water_veins
        self.incidents   = incidents
        self.cv          = cv_pipeline
        self.ml1         = ml1
        self._tick       = 0

    # ── Live map data ─────────────────────────────────────────────────────────

    def live_map(self) -> Dict[str, Any]:
        """
        Full snapshot for the frontend SVG road canvas.
        Includes every edge's density_score (for colour coding) and
        every node's signal state (for live signal badges).
        This is the primary polling endpoint for the Citizen Terminal.
        """
        return {
            "tick":           self._tick,
            "sim_time":       self._tick_to_sim_time(self._tick),
            "edge_densities": self.skeleton.edge_density_map(),
            "signal_states":  self.skeleton.signal_state_map(),
            "active_incidents": [i.to_dict() for i in self.incidents.active_incidents],
            "active_green_waves": self.ml1.active_waves(),
            "grid_health":    self.power_mesh.grid_health(),
            "water_health":   self.water_veins.network_health(),
            "avg_density":    self.skeleton.avg_traffic_density(),
            "active_npcs":    self.population.active_count(),
        }

    # ── Sentinel feed ─────────────────────────────────────────────────────────

    def sentinel_feed(self, limit: int = 30) -> Dict[str, Any]:
        """CV detection log + active incidents for the AI Sentinel tab."""
        return {
            "recent_detections": self.cv.recent_detections(limit),
            "active_incidents":  [i.to_dict() for i in self.incidents.active_incidents],
            "verified_only":     [i.to_dict() for i in self.incidents.active_incidents
                                  if i.details.get("anti_chaos")],
        }

    # ── Traffic intelligence ──────────────────────────────────────────────────

    def traffic_intelligence(self) -> Dict[str, Any]:
        """ML-1 output: active waves, jam log, top congested segments."""
        congested = sorted(
            [{"segment": [list(u), list(v)], "density": d["density_score"]}
             for u, v, d in self.skeleton.G.edges(data=True)],
            key=lambda x: x["density"], reverse=True,
        )[:10]
        return {
            "active_green_waves": self.ml1.active_waves(),
            "wave_history":       self.ml1.wave_history(10),
            "jam_relief_events":  self.ml1.jam_relief_log(10),
            "top_congested":      congested,
            "avg_density":        self.skeleton.avg_traffic_density(),
        }

    # ── Camera feed (per-node) ────────────────────────────────────────────────

    def get_camera_feed(self, node_id: Tuple) -> Dict[str, Any]:
        if not self.skeleton.G.has_node(node_id):
            return {"error": f"Node {node_id} not found."}
        nd       = self.skeleton.G.nodes[node_id]
        entities = self.population.agents_at_node(node_id)
        incidents = [
            {"id": i.incident_id, "type": i.incident_type.value,
             "severity": i.severity.name, "tick": i.tick}
            for i in self.incidents.active_incidents if i.node == node_id
        ]
        edges_info = {
            str(nbr): {
                "status":        self.skeleton.G[node_id][nbr]["status"].value,
                "density_score": self.skeleton.G[node_id][nbr]["density_score"],
                "traffic_count": self.skeleton.G[node_id][nbr]["traffic_density"],
            }
            for nbr in self.skeleton.G.neighbors(node_id)
        }
        return {
            "node":         list(node_id),
            "gps":          self.skeleton.coords_of(node_id),
            "zone":         nd["zone"],
            "cameras":      nd["cameras"],
            "signal":       nd["signal"].state.name,
            "forced_green": nd["signal"].forced_green,
            "entities":     [a.to_dict() for a in entities],
            "entity_count": len(entities),
            "incidents":    incidents,
            "edges":        edges_info,
        }

    # ── City snapshot ─────────────────────────────────────────────────────────

    def city_snapshot(self, tick: int) -> Dict[str, Any]:
        day_tick = tick % TICKS_PER_DAY; h, m = divmod(day_tick, 60)
        return {
            "tick":           tick,
            "sim_time":       f"{h:02d}:{m:02d}",
            "active_npcs":    self.population.active_count(),
            "total_citizens": len(self.population.agents),
            "emergencies":    self.incidents.emergency_count(),
            "grid_health":    self.power_mesh.grid_health(),
            "water_health":   self.water_veins.network_health(),
            "traffic_avg":    self.skeleton.avg_traffic_density(),
            "green_waves":    len(self.ml1.active_waves()),
            "cv_detections":  len(self.cv.recent_detections(5)),
        }

    # ── Utility credits ───────────────────────────────────────────────────────

    def compute_utility_credits(self) -> Dict[str, Any]:
        power_fail = self.power_mesh.offline_geo_nodes()
        water_fail = self.water_veins.zero_pressure_geo_nodes()
        zone       = power_fail | water_fail
        affected   = self.population.agents_in_nodes(zone)
        credits    = []
        for agent in affected:
            agent.in_failure_zone = True
            ip = agent.current_node in power_fail
            iw = agent.current_node in water_fail
            credits.append({"citizen_uid": str(agent.uid), "node": list(agent.current_node),
                            "power_outage": ip, "water_loss": iw,
                            "credit_tokens": (50 if ip else 0) + (30 if iw else 0)})
        for agent in self.population.agents:
            if agent.current_node not in zone: agent.in_failure_zone = False
        return {"failure_zone_size": len(zone), "citizens_affected": len(affected),
                "total_credits_issued": sum(c["credit_tokens"] for c in credits), "records": credits}

    @staticmethod
    def _tick_to_sim_time(tick: int) -> str:
        d = tick % TICKS_PER_DAY; return f"{d//60:02d}:{d%60:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# 8.  SIMULATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OmniCity AI  —  Autonomous Urban OS  v2.0.0                              ║
║   CV-4/5/6 Sentinel | CV-7/8 ANPR+Density | ML-1 Route Optimizer          ║
║   City Grid: {rows}×{cols}  |  Citizens: {citizens}                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


class OmniCitySimulation:
    """Top-level orchestrator."""

    def __init__(self) -> None:
        print(BANNER.format(rows=GRID_ROWS, cols=GRID_COLS, citizens=NUM_CITIZENS))
        print("  Initialising subsystems …")
        self.skeleton    = SpatialSkeleton(GRID_ROWS, GRID_COLS)
        self.population  = AgentPopulation(self.skeleton, NUM_CITIZENS)
        self.power_mesh  = PowerMesh(self.skeleton)
        self.water_veins = WaterVeinNetwork(self.skeleton)
        self.cv          = CVPipelineSimulator(self.skeleton)
        self.ml1         = ML1RouteOptimizer(self.skeleton)
        self.incidents   = IncidentEngine(
            self.skeleton, self.power_mesh, self.water_veins, self.cv, self.ml1,
        )
        self.api = OperationalAPI(
            self.skeleton, self.population, self.power_mesh, self.water_veins,
            self.incidents, self.cv, self.ml1,
        )
        self.tick = 0
        # Snapshot of vehicle plates for CV-8 (populated by backend on startup)
        self._vehicle_plates: List[str] = []
        print("\n  All subsystems online.  Starting autonomous loops …\n")
        print("─" * 82)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, tick_delay_s: float = 0.05) -> None:
        try:
            while True:
                self._step()
                if self.tick % 10 == 0:
                    self._print_heartbeat()
                time.sleep(tick_delay_s)
                self.tick += 1
        except KeyboardInterrupt:
            print("\n\n  [Simulation]  Interrupted.  Shutting down …")
            self.incidents.close()
            print(f"  [Simulation]  {self.tick:,} ticks completed.  Log → {LOG_FILE.resolve()}")

    def _step(self) -> None:
        self.api._tick = self.tick

        # 1. Signals
        self.skeleton.tick_signals()

        # 2. Population
        self.population.tick(self.tick)

        # 3. Infrastructure
        self.power_mesh.tick()
        self.water_veins.tick()

        # 4. CV pipelines (every tick)
        cv4 = self.cv.run_cv4(self.tick)
        cv5 = self.cv.run_cv5(self.tick)
        cv6 = self.cv.run_cv6(self.tick, self.skeleton.high_density_edges())
        self.cv.run_cv7(self.tick)
        anpr_events = self.cv.run_cv8(self.tick, self._vehicle_plates)

        # 5. CV-driven incident generation
        cv_incidents = self.incidents.process_cv_detections(cv4, cv5, cv6, self.tick)
        if cv_incidents:
            self._print_incidents(cv_incidents)

        # 6. Stochastic chaos roll every INCIDENT_ROLL_INTERVAL ticks
        if self.tick % INCIDENT_ROLL_INTERVAL == 0 and self.tick > 0:
            chaos_incs = self.incidents.roll(self.tick)
            if chaos_incs:
                self._print_incidents(chaos_incs)

        # 7. ML-1 anti-jam scan every ML1_SCAN_INTERVAL ticks
        if self.tick % ML1_SCAN_INTERVAL == 0 and self.tick > 0:
            self.ml1.run_anti_jam(self.tick)

        # 8. ML-1 wave expiry
        self.ml1.tick(self.tick)

        # 9. Billing hook every 500 ticks
        if self.tick % 500 == 0 and self.tick > 0:
            summary = self.api.compute_utility_credits()
            if summary["citizens_affected"] > 0:
                print(f"\n  💳  Smart-Billing  |  {summary['citizens_affected']} citizens  |  "
                      f"{summary['total_credits_issued']} tokens issued\n")

    def _print_heartbeat(self) -> None:
        snap = self.api.city_snapshot(self.tick)
        gh, wh, ta = snap["grid_health"], snap["water_health"], snap["traffic_avg"]
        print(
            f"  T{self.tick:>6} {snap['sim_time']}  │"
            f"  NPCs: {snap['active_npcs']:>4}/{snap['total_citizens']}  │"
            f"  🚨 Emg: {snap['emergencies']:>2}  │"
            f"  🟢 GW: {snap['green_waves']}  │"
            f"  ⚡ Grid: {gh:>5.1f}%  │"
            f"  💧 Water: {wh:>5.1f}%  │"
            f"  🚗 Density: {ta:.3f}"
        )

    def _print_incidents(self, incidents: List[IncidentPacket]) -> None:
        icons = {
            IncidentType.ACCIDENT:        "🚗",
            IncidentType.THEFT:           "🔓",
            IncidentType.WEAPON_DETECTED: "🔫",
            IncidentType.FIRE_SMOKE:      "🔥",
            IncidentType.PIPE_BURST:      "💧",
            IncidentType.GRID_BLACKOUT:   "⚡",
            IncidentType.ROAD_DEBRIS:     "⚠️ ",
        }
        for inc in incidents:
            gw = f" | GW={inc.green_wave_id}" if inc.green_wave_id else ""
            print(
                f"\n  {icons.get(inc.incident_type,'❓')} [{inc.incident_id}]  "
                f"{inc.incident_type.value}  SEV={inc.severity.name}  "
                f"AUTH={inc.authority}  GPS=({inc.gps[0]:.4f},{inc.gps[1]:.4f}){gw}\n"
            )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    sim = OmniCitySimulation()
    sim.run(tick_delay_s=0.02)
