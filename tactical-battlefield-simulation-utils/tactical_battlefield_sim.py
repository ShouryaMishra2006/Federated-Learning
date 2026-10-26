#!/usr/bin/env python3
"""
tactical_battlefield_sim.py
===========================
Member-1 deliverable: physical battlefield world (SimPy) + Tactical Translation Layer
that emits the exact 13-feature frame used by the federated baseline notebook.

PIPELINE
--------
  SimPy world (ambient weather, benign traffic, hostile campaigns)
        |
        v   per node, every 2 s window, K=8 sub-samples (4 Hz on-node feature rate)
  Region sensor models  ->  physical metrics (dB, um/s, nT, degC, dBm, ...)
        |
        v   TACTICAL TRANSLATION LAYER  (translate_to_packets)
  event-driven uplink policy (send-on-delta + alert keep-alives + heartbeat)
        |
        v   shared half-duplex radio channel (CSMA/CA, RSSI->SINR->PER, ARQ), all in SimPy
  node-side packet trace  ->  UNSW-NB15-style flow record
        |
        v
  13 features: dur spkts dpkts sbytes dbytes sload dload sjit djit tcprtt synack ackdat rate

MAPPING TABLE  (physical quantity -> traffic feature; this is the paper's "translation")
-------------------------------------------------------------------------------------
  how much the world changes   -> spkts / sbytes / rate    (send-on-delta + alert reports)
  which modalities are active  -> sbytes/spkts             (payload size differs per modality;
                                                            overhead-sat patch = RLE'd, so bigger
                                                            when more pixels change)
  temporal structure of events -> sjit / dur               (gunshot bursts, footstep cadence,
                                                            hover vs. fly-by change report timing)
  radio RSSI / noise floor     -> synack / ackdat / tcprtt (SINR -> packet error -> retransmit,
                                                            CCA back-off under jamming)
  cluster-wide event storms    -> synack / djit / dpkts    (shared channel + gateway CPU queueing)
  gateway ACK / alert polling  -> dpkts / dbytes / dload

SENSOR SUITES (as specified in the project brief)
  Region A  air strip      : overhead EO patch (8x8 flattened) + dB audio + speed laser + thermal array
  Region B  supply valley  : geophone + magnetometer + PIR
  Region C  perimeter      : piezo-acoustic fence cable + ambient probes (T/RH/P) + ad-hoc radio RSSI

GROUNDING (what is literature-backed vs. assumed) is documented in PARAMETER_NOTES at the bottom.

Usage in the notebook (drop-in for the UNSW/CICIDS partitions)
--------------------------------------------------------------
    from tactical_battlefield_sim import build_tactical_partitions, unsw_cols_map
    partitions, df_eval, raw = build_tactical_partitions(rows_per_client=20000)
    execute_federated_experiment(partitions, df_eval, model_type="FcNN", strategy="FedAvg")

Requires: pip install simpy numpy pandas
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy

# =============================================================================
# 1. CONSTANTS / FEATURE CONTRACT
# =============================================================================
FEATURES = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sload', 'dload',
            'sjit', 'djit', 'tcprtt', 'synack', 'ackdat', 'rate']
unsw_cols_map = FEATURES                      # alias so notebook code is unchanged

WINDOW_S = 2.0                                # one flow record per node per window
K_SUB = 8                                     # sub-samples per window  (4 Hz feature rate)
DT_SUB = WINDOW_S / K_SUB
C_SOUND = 343.0
PERCEPT_DB = 12.0                             # hostile source is labelled 'present' iff it reaches the node's own alert level (12 dB over noise)
HDR_BYTES = 31                                # 11 B MAC/6LoWPAN + 20 B transport header
MTU_PAYLOAD = 96
PL0_DB = 31.7                                 # free-space loss @1 m, ~915 MHz
HEARTBEAT_B = 14

# A-priori, globally agreed normalisation (no per-client fitting -> FL-safe).
# x_norm = log1p(x/ref) / log1p(max/ref), clipped to [0,1].
FEATURE_SPEC = {   # maxima ~1.4-2x the largest value seen over six clusters (incl. jamming episodes)
    'dur':    (0.05, 5.0),   'spkts':  (1.0, 40.0),    'dpkts': (1.0, 20.0),
    'sbytes': (10.0, 3000.0), 'dbytes': (10.0, 600.0),
    'sload':  (100.0, 1.2e5), 'dload':  (100.0, 6e4),
    'sjit':   (0.1, 1800.0), 'djit':   (0.1, 1800.0),   # ms
    'tcprtt': (0.001, 3.0),  'synack': (0.001, 2.5), 'ackdat': (0.001, 1.0),   # s
    'rate':   (1.0, 600.0),
}


def normalize_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for f, (ref, mx) in FEATURE_SPEC.items():
        v = np.log1p(raw[f].clip(lower=0).to_numpy(dtype=np.float64) / ref) / math.log1p(mx / ref)
        out[f] = np.clip(v, 0.0, 1.0).astype(np.float32)
    return out


# =============================================================================
# 2. EVENTS (anything physical that moves, emits, or radiates)
# =============================================================================
@dataclass
class Event:
    eid: int
    kind: str
    label: str                      # 'Benign' or an attack class
    t0: float
    wp_t: np.ndarray                # waypoint times (relative to t0)
    wp_xyz: np.ndarray              # (n,3) metres
    power: np.ndarray               # (n,) 0..1 engine/activity profile
    sig: Dict[str, object] = field(default_factory=dict)
    impulses: np.ndarray = field(default_factory=lambda: np.zeros(0))      # absolute times
    impulse_db: np.ndarray = field(default_factory=lambda: np.zeros(0))    # dB@1m (A) or mV@contact (C)

    @property
    def t1(self) -> float:
        return self.t0 + float(self.wp_t[-1])

    @property
    def hostile(self) -> bool:
        return self.label != 'Benign'

    def pos(self, t):
        tr = np.asarray(t, dtype=float) - self.t0
        return np.stack([np.interp(tr, self.wp_t, self.wp_xyz[:, i]) for i in range(3)], axis=-1)

    def pw(self, t):
        return np.interp(np.asarray(t, dtype=float) - self.t0, self.wp_t, self.power, left=0.0, right=0.0)


# =============================================================================
# 3. CONFIG
# =============================================================================
@dataclass
class RadioCfg:
    bitrate: float = 250e3
    tx_dbm: float = 14.0
    noise_floor_dbm: float = -114.0
    n_exp: float = 3.0
    sigma_db: float = 6.0
    shadow_rho: float = 0.9
    cca_dbm: float = -92.0
    slot_s: float = 0.002
    phy_overhead_bytes: int = 8
    gw_service_s: float = 0.0015
    rto_s: float = 0.06


@dataclass
class ClusterCfg:
    client: str
    region: str                                   # 'A' | 'B' | 'C'
    n_nodes: int = 6
    radio: RadioCfg = field(default_factory=RadioCfg)
    gw_pos: Tuple[float, float, float] = (0.0, -100.0, 4.0)
    t0_hour: float = 10.0
    wind_mean: float = 3.0
    wind_gust: float = 1.0
    rain_rate_h: float = 1.0
    temp_mean: float = 27.0
    temp_amp: float = 7.0
    amb_db: float = 46.0                          # A: ambient audio level (dB SPL)
    amb_centroid_hz: float = 300.0
    th_fov_deg: float = 70.0
    geo_noise_um: float = 3.0                     # B: geophone noise (um/s rms)
    mag_noise_nt: float = 0.5
    pz_noise_mv: float = 1.5                      # C: piezo noise (mV rms)
    benign_rate_h: Dict[str, float] = field(default_factory=dict)
    attack_mix: Dict[str, float] = field(default_factory=dict)
    attack_rate_h: float = 20.0                   # attack *campaigns* per hour


@dataclass
class Sense:
    series: Dict[str, np.ndarray] = field(default_factory=dict)   # per-modality K-sample series
    alert: Dict[str, np.ndarray] = field(default_factory=dict)    # per-modality alert flags
    sev: Dict[str, np.ndarray] = field(default_factory=dict)      # per-modality severity (0 = at alert level, 1 = ~30 dB above)
    phys: Dict[str, float] = field(default_factory=dict)          # window summary (for logging)
    ev_snr: Dict[int, np.ndarray] = field(default_factory=dict)   # per-event perceptibility (dB)

    def note(self, eid: int, snr: np.ndarray):
        cur = self.ev_snr.get(eid)
        self.ev_snr[eid] = snr if cur is None else np.maximum(cur, snr)


@dataclass
class Node:
    nid: int
    pos: np.ndarray
    aim: np.ndarray                  # thermal / EO boresight
    aim_las: np.ndarray              # speed-laser boresight
    phase: float
    d_gw: float
    shadow_db: float = 0.0
    x_last: Dict[str, float] = field(default_factory=dict)
    alert_prev: Dict[str, bool] = field(default_factory=dict)
    mag_drift: float = 0.0
    pir_prev: bool = False
    sat_bg: Optional[np.ndarray] = None
    sat_prev: Optional[np.ndarray] = None


# =============================================================================
# 4. WORLD
# =============================================================================
def build_nodes(cfg: ClusterCfg, rng: np.random.Generator, gw: np.ndarray) -> List[Node]:
    n, nodes = cfg.n_nodes, []
    for i in range(n):
        if cfg.region == 'A':      # runway along x
            x = np.linspace(-280, 280, n)[i] + rng.uniform(-15, 15)
            y = (1 if i % 2 == 0 else -1) * rng.uniform(35, 80)
            pos = np.array([x, y, 1.5])
            aim = np.array([0.0, -np.sign(y), 0.2]); aim /= np.linalg.norm(aim)
            aim_las = np.array([-np.sign(x) if abs(x) > 20 else 1.0, 0.0, 0.05])
            aim_las /= np.linalg.norm(aim_las)
        elif cfg.region == 'B':    # valley road along x, geophones at lateral offsets
            x = np.linspace(-300, 300, n)[i] + rng.uniform(-20, 20)
            y = (1 if i % 2 == 0 else -1) * rng.uniform(8, 60)
            pos = np.array([x, y, -0.3])
            aim = aim_las = np.array([0.0, 0.0, 1.0])
        else:                      # C: perimeter fence along x (y ~ 0)
            x = np.linspace(-250, 250, n)[i] + rng.uniform(-8, 8)
            pos = np.array([x, rng.uniform(-1.5, 1.5), 1.2])
            aim = aim_las = np.array([0.0, 1.0, 0.0])
        node = Node(nid=i, pos=pos, aim=aim, aim_las=aim_las,
                    phase=float(rng.uniform(0, WINDOW_S)),
                    d_gw=float(max(np.linalg.norm(pos - gw), 5.0)),
                    shadow_db=float(rng.normal(0, cfg.radio.sigma_db)))
        if cfg.region == 'A':
            node.sat_bg = rng.uniform(0.25, 0.55, 64)
            node.sat_prev = node.sat_bg * 0.8
        nodes.append(node)
    return nodes


class World:
    def __init__(self, env: simpy.Environment, cfg: ClusterCfg, rng: np.random.Generator):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.events: List[Event] = []
        self.rf_events: List[Event] = []
        self.ev_by_id: Dict[int, Event] = {}
        self._eid = 0
        self.channel = simpy.Resource(env, capacity=1)      # shared half-duplex medium
        self.gw_cpu = simpy.Resource(env, capacity=1)       # gateway control-plane service queue
        self.records: List[dict] = []
        self.campaigns: List[Tuple[str, float, int]] = []   # (attack class, start time, n events)
        self.gw = np.array(cfg.gw_pos, float)
        self.wind, self.rain, self.rain_until, self.rain_int = cfg.wind_mean, 0.0, -1.0, 0.0
        self.temp, self.rh, self.press, self.sun = cfg.temp_mean, 60.0, 1008.0, 0.0
        self.nodes = build_nodes(cfg, rng, self.gw)

    # -- events --------------------------------------------------------------
    def new_eid(self) -> int:
        self._eid += 1
        return self._eid

    def add_event(self, ev: Event):
        self.events.append(ev)
        self.ev_by_id[ev.eid] = ev
        if 'rf_kind' in ev.sig:
            self.rf_events.append(ev)
        if len(self.events) > 40:
            horizon = self.env.now - 20.0
            for e in self.events:
                if e.t1 < horizon:
                    self.ev_by_id.pop(e.eid, None)
            self.events = [e for e in self.events if e.t1 >= horizon]
            self.rf_events = [e for e in self.rf_events if e.t1 >= horizon]

    def live_events(self, t0: float, t1: float) -> List[Event]:
        return [e for e in self.events if e.t0 <= t1 and e.t1 >= t0]

    # -- radio environment ---------------------------------------------------
    def interference_dbm(self, pos: np.ndarray, t: float) -> float:
        """Total jammer/benign-RF interference power at `pos` at time t (dBm, -200 if none)."""
        mw = 0.0
        for ev in self.rf_events:
            if ev.sig['rf_kind'] in ('jam', 'burst') and ev.t0 <= t <= ev.t1:
                lvl = float(_rf_level_vec(ev, pos, np.array([t]))[0])
                if lvl > -150:
                    mw += 10 ** (lvl / 10)
        return 10 * math.log10(mw) if mw > 0 else -200.0

    def base_rssi(self, node: Node, t: float) -> float:
        r = self.cfg.radio
        return (r.tx_dbm - PL0_DB - 10 * r.n_exp * math.log10(node.d_gw)
                - node.shadow_db - float(_blockage_vec(self, node, np.array([t]))[0]))


def _rf_level_vec(ev: Event, pos: np.ndarray, tk: np.ndarray) -> np.ndarray:
    s = ev.sig
    r = max(float(np.linalg.norm(ev.wp_xyz[0] - pos)), 5.0)
    pr = float(s['rf_tx_dbm']) - PL0_DB - 10 * float(s['rf_n']) * math.log10(r)
    per = float(s['rf_period'])
    on = (tk >= ev.t0) & (tk <= ev.t1) & (((tk - ev.t0) % per) < float(s['rf_duty']) * per)
    return np.where(on, pr, -200.0)


def _blockage_vec(w: World, node: Node, tk: np.ndarray) -> np.ndarray:
    """Extra link loss (dB) when a large body crosses the node->gateway line of sight."""
    tot = np.zeros(len(tk))
    a, b = node.pos[:2], w.gw[:2]
    ab = b - a
    L2 = float(ab @ ab) + 1e-9
    for ev in w.events:
        blk = float(ev.sig.get('rf_block_db', 0.0))
        if blk <= 0 or ev.t0 > tk[-1] or ev.t1 < tk[0]:
            continue
        p = ev.pos(tk)[:, :2]
        u = np.clip(((p - a) @ ab) / L2, 0, 1)
        d = np.linalg.norm(p - (a + np.outer(u, ab)), axis=1)
        act = (tk >= ev.t0) & (tk <= ev.t1)
        tot += np.where(act & (d < 15.0), blk * (1 - d / 15.0), 0.0)
    return tot


def ambient_process(env: simpy.Environment, w: World):
    """Weather / diurnal state, 1 Hz."""
    cfg, rng = w.cfg, w.rng
    while True:
        yield env.timeout(1.0)
        hour = (cfg.t0_hour + env.now / 3600.0) % 24.0
        w.sun = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0)) if 6 <= hour <= 18 else 0.0
        w.wind = max(0.0, w.wind + 0.05 * (cfg.wind_mean - w.wind) + 0.35 * cfg.wind_gust * rng.normal())
        if env.now >= w.rain_until:
            w.rain = 0.0
            if rng.random() < cfg.rain_rate_h / 3600.0:
                w.rain_until = env.now + rng.uniform(60, 240)
                w.rain_int = rng.uniform(0.3, 1.0)
        else:
            w.rain = w.rain_int
        t_tgt = cfg.temp_mean + cfg.temp_amp * math.sin(2 * math.pi * (hour - 9.0) / 24.0) - 3.0 * w.rain
        w.temp += 0.02 * (t_tgt - w.temp) + 0.03 * rng.normal()
        rh_tgt = 70.0 - 2.0 * (w.temp - cfg.temp_mean) + 25.0 * w.rain
        w.rh = float(np.clip(w.rh + 0.04 * (rh_tgt - w.rh) + 0.15 * rng.normal(), 5, 100))
        w.press += 0.01 * ((1008.0 - 3.0 * w.rain) - w.press) + 0.02 * rng.normal()


# =============================================================================
# 5. SENSOR MODELS  (physical world -> per-modality series)
# =============================================================================
def _absorb(f_hz: float) -> float:
    """Atmospheric absorption, dB/km (order-of-magnitude, 4-15 dB/km over 0.3-1 kHz)."""
    return 3.0 + 0.012 * f_hz


def _geom(ev: Event, node: Node, tk: np.ndarray):
    p = ev.pos(tk) - node.pos
    r = np.maximum(np.linalg.norm(p, axis=1), 1.0)
    r2 = np.maximum(np.linalg.norm(ev.pos(tk + 0.05) - node.pos, axis=1), 1.0)
    return r, (r2 - r) / 0.05, p                      # distance, radial velocity (+ = receding), vector


def _tk(t0: float) -> np.ndarray:
    return t0 + (np.arange(K_SUB) + 0.5) * DT_SUB


def _amb_audio_db(w: World) -> float:
    return w.cfg.amb_db + 8.0 * math.log10(1 + (w.wind / 2.5) ** 2) + 14.0 * w.rain


GSD_M = 16.0                                       # overhead patch ground-sample distance
_GX, _GY = np.meshgrid((np.arange(8) - 3.5) * GSD_M, (np.arange(8) - 3.5) * GSD_M)
_GX, _GY = _GX.ravel(), _GY.ravel()


def _sat_frame(node: Node, objs, w: World):
    gain = 0.6 + 0.35 * w.sun
    img = gain * node.sat_bg + w.rng.normal(0, 0.012, 64)
    c_eff = {}
    for ev, p in objs:
        size, con = float(ev.sig['size_m']), float(ev.sig['sat_contrast'])
        s_px = max(size / GSD_M, 0.6) / 2.0
        blob = np.exp(-(((_GX - p[0]) / GSD_M) ** 2 + ((_GY - p[1]) / GSD_M) ** 2) / (2 * s_px ** 2))
        c = con * min(1.0, (size / GSD_M) ** 2)
        img += c * blob
        c_eff[ev.eid] = c * float(blob.max())
    return img, c_eff


def sense_A(w: World, node: Node, t0: float) -> Sense:
    """Region A (air strip): overhead EO patch + dB audio + speed laser + thermal array."""
    cfg, rng = w.cfg, w.rng
    tk, out = _tk(t0), Sense()
    amb = _amb_audio_db(w)
    P_amb = 10 ** (amb / 10)
    P = np.full(K_SUB, P_amb)
    cen = P_amb * cfg.amb_centroid_hz * np.ones(K_SUB)
    peak = np.full(K_SUB, -np.inf)
    dT = np.zeros(K_SUB)
    las_v, las_lock = np.zeros(K_SUB), np.zeros(K_SUB, bool)
    sat_objs = []
    cos_th, cos_las = math.cos(math.radians(cfg.th_fov_deg)), math.cos(math.radians(20.0))

    for ev in w.live_events(t0, t0 + WINDOW_S):
        s = ev.sig
        r, vr, p = _geom(ev, node, tk)
        pw = ev.pw(tk)
        snr = np.full(K_SUB, -99.0)
        if float(s.get('spl_1m', 0)) > 0:                           # continuous source (engine / rotor)
            L = (float(s['spl_1m']) + 10 * np.log10(np.maximum(pw, 1e-3)) - 20 * np.log10(r)
                 - _absorb(float(s['f_hz'])) * r / 1000.0)
            Pe = np.where(pw > 0, 10 ** (L / 10), 0.0)
            f = float(s['f_hz']) * C_SOUND / (C_SOUND + np.clip(vr, -150, 150))   # Doppler
            P += Pe
            cen += Pe * f
            snr = np.maximum(snr, np.where(pw > 0, 10 * np.log10(np.maximum(Pe, 1e-12) / P_amb), -99.0))
        if ev.impulses.size:                                         # gunshots (N-wave + muzzle blast)
            for i in np.nonzero((ev.impulses >= t0) & (ev.impulses < t0 + WINDOW_S))[0]:
                k = min(int((ev.impulses[i] - t0) / DT_SUB), K_SUB - 1)
                pk = float(ev.impulse_db[i]) - 20 * math.log10(r[k]) - _absorb(2000.0) * r[k] / 1000.0
                Pi = 10 ** ((pk - 10 * math.log10(DT_SUB / 0.002)) / 10)   # 2 ms pulse in a 250 ms bin
                P[k] += Pi
                cen[k] += Pi * 1500.0
                peak[k] = max(peak[k], pk)
                snr[k] = max(snr[k], pk - (amb + 12.0))
        if float(s.get('dT_src', 0)) > 0:                            # exhaust plume seen by thermal array
            dTe = (float(s['dT_src']) * pw * (float(s['r0']) / np.maximum(r, float(s['r0']))) ** 2
                   * np.exp(-r / 3000.0))
            dTe = np.where((p @ node.aim) / r >= cos_th, dTe, 0.0)
            dT += dTe
            snr = np.maximum(snr, np.where(dTe > 0, 20 * np.log10(np.maximum(dTe, 1e-6) / 0.15), -99.0))
        if float(s.get('rcs_range', 0)) > 0:                         # speed laser lock
            lock = (((p @ node.aim_las) / r >= cos_las) & (r >= 10) & (r <= float(s['rcs_range'])) & (pw > 0))
            las_lock |= lock
            las_v = np.maximum(las_v, np.where(lock, np.abs(vr), 0.0))
            snr = np.maximum(snr, np.where(lock, PERCEPT_DB + 2.0, -99.0))
        if float(s.get('size_m', 0)) > 0:
            sat_objs.append((ev, p))
        out.note(ev.eid, snr)

    sat = np.zeros(K_SUB)                                            # overhead frames @ 1 Hz
    half = K_SUB // 2
    for k0 in (0, half):
        objs = [(ev, p[k0]) for ev, p in sat_objs
                if abs(p[k0][0]) < 64 + float(ev.sig['size_m']) and abs(p[k0][1]) < 64 + float(ev.sig['size_m'])]
        frame, c_eff = _sat_frame(node, objs, w)
        sat[k0:k0 + half] = float(np.mean(np.abs(frame - node.sat_prev) > 0.05))
        node.sat_prev = frame
        for eid, c in c_eff.items():
            arr = np.full(K_SUB, -99.0)
            arr[k0:k0 + half] = 20 * math.log10(max(c, 1e-6) / 0.012)
            out.note(eid, arr)

    Leq = 10 * np.log10(P) + rng.normal(0, 0.8, K_SUB)
    pk_all = np.maximum(peak, Leq + 3.0 + rng.normal(0, 0.7, K_SUB))
    th_abs = w.temp + dT + rng.normal(0, 0.15, K_SUB)
    las = np.where(las_lock, las_v, 0.0)
    out.series = {'aud': Leq, 'th': th_abs, 'las': las, 'sat': sat}
    out.alert = {'aud': (Leq - amb) >= 12.0, 'th': dT >= 2.0, 'las': las_lock & (las_v >= 2.0), 'sat': sat >= 0.05}
    out.sev = {'aud': np.clip((Leq - amb - 12.0) / 30.0, 0, None), 'th': np.clip((dT - 2.0) / 20.0, 0, None),
               'las': np.clip((las - 2.0) / 60.0, 0, None), 'sat': sat}
    out.phys = dict(aud_leq_db=float(Leq.mean()), aud_peak_db=float(pk_all.max()),
                    aud_centroid_hz=float((cen / P).mean()), th_max_c=float(th_abs.max()),
                    las_speed_ms=float(las.max()), sat_changed_frac=float(sat.max()))
    return out


def sense_B(w: World, node: Node, t0: float) -> Sense:
    """Region B (supply valley): geophone + magnetometer + PIR."""
    cfg, rng = w.cfg, w.rng
    tk, out = _tk(t0), Sense()
    n0 = cfg.geo_noise_um * (1 + 0.05 * w.wind ** 2 + 1.5 * w.rain)
    noise_p = (n0 * np.exp(rng.normal(0, 0.10, K_SUB))) ** 2
    bl, bm, bh = noise_p * 0.45, noise_p * 0.30, noise_p * 0.25       # low / mid / high band power
    mag = np.zeros(K_SUB)
    p_fa = 0.004 + 0.02 * w.sun + 0.002 * w.wind + 0.03 * w.rain
    p_none = np.ones(K_SUB)

    for ev in w.live_events(t0, t0 + WINDOW_S):
        s = ev.sig
        r, _, _ = _geom(ev, node, tk)
        pw, act = ev.pw(tk), (tk >= ev.t0) & (tk <= ev.t1)
        snr = np.full(K_SUB, -99.0)
        if float(s.get('seis_a1m', 0)) > 0:
            a = float(s['seis_a1m']) * pw * r ** (-float(s['seis_alpha']))
            md = float(s.get('mod_depth', 0.0))
            a = a * np.maximum(1 + md * np.cos(2 * np.pi * float(s.get('mod_hz', 1.0)) * (tk - ev.t0)
                                              + float(s.get('phi', 0.0))), 0.05)
            a = a * np.exp(rng.normal(0, 0.08, K_SUB))
            pe = a ** 2
            band = s['band']
            bl += pe * band[0]; bm += pe * band[1]; bh += pe * band[2]
            snr = np.maximum(snr, np.where(pw > 0, 10 * np.log10(np.maximum(pe, 1e-12) / noise_p), -99.0))
        if float(s.get('mag_K', 0)) > 0:
            b = float(s['mag_sign']) * float(s['mag_K']) / np.maximum(r, 2.0) ** 3
            mag += np.where(act, b, 0.0)
            snr = np.maximum(snr, np.where(act, 20 * np.log10(np.maximum(np.abs(b), 1e-9) / cfg.mag_noise_nt), -99.0))
        if float(s.get('pir_contrast', 0)) > 0:
            v = np.linalg.norm(ev.pos(tk + 0.05) - ev.pos(tk), axis=1) / 0.05
            R = float(s['pir_range'])
            pi_ = np.where(act & (r < R), np.clip(float(s['pir_contrast']) * (1 - r / R) * np.minimum(1, v / 0.3) * 1.1,
                                                   0, 0.95), 0.0)
            p_none *= (1 - pi_)
            snr = np.maximum(snr, np.where(pi_ > 0, 10 * np.log10(np.maximum(pi_, 1e-9) / p_fa), -99.0))
        out.note(ev.eid, snr)

    node.mag_drift += -0.1 * node.mag_drift + 0.2 * rng.normal()
    mag_abs = np.abs(mag + node.mag_drift + rng.normal(0, cfg.mag_noise_nt, K_SUB))
    tot_p = bl + bm + bh
    rms = np.sqrt(tot_p)
    geo_db = 20 * np.log10(rms)
    pir = (rng.random(K_SUB) < (1 - p_none * (1 - p_fa))).astype(float)
    pir_alert = (pir > 0) & (np.concatenate([[float(node.pir_prev)], pir[:-1]]) > 0)
    node.pir_prev = bool(pir[-1])
    out.series = {'geo': geo_db, 'mag': mag_abs, 'pir': pir}
    out.alert = {'geo': geo_db >= 20 * math.log10(n0) + 12.0, 'mag': mag_abs >= 6.0, 'pir': pir_alert}
    out.sev = {'geo': np.clip((geo_db - (20 * math.log10(n0) + 12.0)) / 30.0, 0, None),
               'mag': np.clip(np.log10(np.maximum(mag_abs, 1e-3) / 6.0) / 1.5, 0, None), 'pir': pir_alert * 0.5}
    out.phys = dict(geo_rms_um=float(rms.mean()), geo_peak_um=float(rms.max()),
                    geo_low_frac=float((bl / tot_p).mean()), geo_high_frac=float((bh / tot_p).mean()),
                    mag_peak_nt=float(mag_abs.max()), pir_count=float(pir.sum()))
    return out


def sense_C(w: World, node: Node, t0: float) -> Sense:
    """Region C (perimeter): piezo-acoustic fence + ambient probes + ad-hoc radio RSSI."""
    cfg, rng, rad = w.cfg, w.rng, w.cfg.radio
    tk, out = _tk(t0), Sense()
    n_pz = cfg.pz_noise_mv * (1 + 0.35 * w.wind ** 1.3 + 2.0 * w.rain)
    P = (n_pz * np.exp(rng.normal(0, 0.08, K_SUB))) ** 2
    imp = rng.poisson(0.004 + 0.06 * w.rain + 0.004 * w.wind, K_SUB).astype(float)  # wind/rain ticks > 4 sigma
    floor_mw = 10 ** (rad.noise_floor_dbm / 10)
    I_mw = np.zeros(K_SUB)
    unk = np.full(K_SUB, rad.noise_floor_dbm - 12.0)
    blk = _blockage_vec(w, node, tk)

    for ev in w.live_events(t0, t0 + WINDOW_S):
        s = ev.sig
        r, _, _ = _geom(ev, node, tk)
        pw = ev.pw(tk)
        snr = np.full(K_SUB, -99.0)
        att = np.exp(-r / float(s.get('pz_lambda', 25.0))) / np.sqrt(1 + r / 5.0)
        if float(s.get('pz_a1m', 0)) > 0:
            a = float(s['pz_a1m']) * pw * att
            P += a ** 2
            snr = np.maximum(snr, np.where(pw > 0, 20 * np.log10(np.maximum(a, 1e-9) / n_pz), -99.0))
        if ev.impulses.size:
            for i in np.nonzero((ev.impulses >= t0) & (ev.impulses < t0 + WINDOW_S))[0]:
                k = min(int((ev.impulses[i] - t0) / DT_SUB), K_SUB - 1)
                amp = float(ev.impulse_db[i]) * float(att[k])
                P[k] += amp ** 2 * (0.02 / DT_SUB)
                imp[k] += float(amp > 4 * n_pz)
                snr[k] = max(snr[k], 20 * math.log10(max(amp, 1e-9) / n_pz))
        if 'rf_kind' in s:
            lvl = _rf_level_vec(ev, node.pos, tk)
            if s['rf_kind'] in ('jam', 'burst'):
                I_mw += np.where(lvl > -150, 10 ** (lvl / 10), 0.0)
                snr = np.maximum(snr, np.where(lvl > -150, lvl - rad.noise_floor_dbm, -99.0))
            else:                                                       # rogue beacon
                unk = np.maximum(unk, np.where(lvl > -150, lvl, unk))
                snr = np.maximum(snr, np.where(lvl > -150, lvl - rad.noise_floor_dbm, -99.0))
        out.note(ev.eid, snr)

    pz_db = 20 * np.log10(np.sqrt(P))
    rssi = (rad.tx_dbm - PL0_DB - 10 * rad.n_exp * math.log10(node.d_gw) - node.shadow_db - blk
            + rng.normal(0, 1.0, K_SUB))
    nf = 10 * np.log10(floor_mw + I_mw) + rng.normal(0, 0.8, K_SUB)
    temp = w.temp + rng.normal(0, 0.05, K_SUB)
    rh = w.rh + rng.normal(0, 0.4, K_SUB)
    prs = w.press + rng.normal(0, 0.03, K_SUB)
    unk = unk + rng.normal(0, 1.0, K_SUB)
    out.series = {'pz': pz_db, 'imp': imp, 'T': temp, 'RH': rh, 'P': prs, 'rssi': rssi, 'nf': nf, 'unk': unk}
    out.alert = {'pz': pz_db >= 20 * math.log10(n_pz) + 12.0, 'imp': imp >= 1.0,
                 'rssi': (rssi - rssi.mean()) <= -10.0, 'nf': nf >= rad.noise_floor_dbm + 10.0,
                 'unk': unk >= rad.noise_floor_dbm + 10.0}
    out.sev = {'pz': np.clip((pz_db - (20 * math.log10(n_pz) + 12.0)) / 30.0, 0, None), 'imp': imp / 3.0,
               'rssi': np.clip((-(rssi - rssi.mean()) - 10.0) / 20.0, 0, None),
               'nf': np.clip((nf - (rad.noise_floor_dbm + 10.0)) / 30.0, 0, None),
               'unk': np.clip((unk - (rad.noise_floor_dbm + 10.0)) / 40.0, 0, None)}
    snr_db = rssi - nf
    out.phys = dict(pz_rms_mv=float(np.sqrt(P).mean()), pz_impulses=float(imp.sum()), amb_temp_c=float(temp.mean()),
                    amb_rh=float(rh.mean()), amb_press_hpa=float(prs.mean()), rssi_dbm=float(rssi.mean()),
                    noise_floor_dbm=float(nf.mean()), snr_db=float(snr_db.mean()), unk_dbm=float(unk.max()))
    return out


SENSE = {'A': sense_A, 'B': sense_B, 'C': sense_C}


def label_window(w: World, sense: Sense) -> Tuple[int, str, str]:
    """Ground truth: a hostile source is 'in' the window iff it is perceptible at this node."""
    best_h = best_b = None
    for eid, snr in sense.ev_snr.items():
        ev = w.ev_by_id.get(eid)
        if ev is None:
            continue
        mx = float(snr.max())
        if not (int((snr >= PERCEPT_DB).sum()) >= 2 or mx >= PERCEPT_DB + 9.0):   # >=2 sub-samples, or one strong one
            continue
        if ev.hostile:
            if best_h is None or mx > best_h[0]:
                best_h = (mx, ev)
        elif best_b is None or mx > best_b[0]:
            best_b = (mx, ev)
    if best_h:
        return 1, best_h[1].label, best_h[1].kind
    if best_b:
        return 0, 'Benign', best_b[1].kind
    return 0, 'Benign', 'ambient'


# =============================================================================
# 6. TACTICAL TRANSLATION LAYER   (physical metrics  ->  uplink packet plan)
# =============================================================================
# modality : (send-on-delta threshold, normal report payload B, alert report payload B)
MOD_SPEC = {
    'aud': (4.0, 8, 20), 'th': (0.8, 4, 20), 'las': (1.5, 4, 10), 'sat': (0.04, 6, 16),
    'geo': (5.0, 6, 24), 'mag': (2.5, 4, 12), 'pir': (0.5, 2, 4),
    'pz': (4.5, 6, 24), 'imp': (1.0, 2, 12), 'T': (0.4, 4, 4), 'RH': (2.0, 4, 4), 'P': (0.4, 4, 4),
    'rssi': (6.0, 3, 14), 'nf': (4.0, 3, 14), 'unk': (6.0, 3, 16),
}


def translate_to_packets(node: Node, sense: Sense, rng: np.random.Generator, widx: int = 0) -> List[Tuple[float, int]]:
    """
    Event-driven telemetry policy used by real UGS / WSN nodes:
      * send-on-delta : a modality reports when its value moved >= delta since the last report
      * alert keep-alive : while a modality is above its alert level it re-sends a rich payload.  Cadence and
                           payload scale with severity (how far above the alert level): every 1 s (weak),
                           0.5 s (moderate), 0.25 s (strong); payload 0.7x-1.3x of the nominal alert size
                           (longer spectra / more pixels / more tracks to describe a stronger source)
      * heartbeat : one health packet per window (14-29 B: battery, uptime, counters)
      * housekeeping : every 5th window a 30-80 B sensor digest, every 12th a 40-90 B routing/neighbour update
                       (node-staggered) -> benign traffic is not a single point
    Reports that fall in the same 250 ms sub-sample share one frame (MTU 96 B payload).
    Returns [(release_offset_in_window_s, frame_bytes), ...] sorted by time.
    """
    payload = np.zeros(K_SUB, dtype=int)
    for mod, x in sense.series.items():
        delta, nb, ab = MOD_SPEC[mod]
        last = node.x_last.get(mod, float(x[0]))
        al = sense.alert.get(mod)
        sv = sense.sev.get(mod)
        prev = node.alert_prev.get(mod, False)
        for k in range(K_SUB):
            if abs(float(x[k]) - last) >= delta:
                payload[k] += nb
                last = float(x[k])
            if al is not None and bool(al[k]):
                s_k = float(min(sv[k], 1.5)) if sv is not None else 0.5
                period = 1 if s_k >= 0.6 else (2 if s_k >= 0.25 else 4)
                if k % period == 0 or not prev:
                    if mod == 'sat':
                        payload[k] += 16 + int(min(1.0, float(x[k])) * 128)
                    else:
                        payload[k] += int(round(ab * (0.7 + 0.6 * min(s_k, 1.0)) * rng.uniform(0.9, 1.1)))
            prev = bool(al[k]) if al is not None else False
        node.x_last[mod], node.alert_prev[mod] = last, prev
    payload[int(rng.integers(K_SUB))] += HEARTBEAT_B + int(rng.integers(0, 16))
    if (widx + node.nid) % 5 == 0:
        payload[int(rng.integers(K_SUB))] += int(rng.integers(30, 80))
    if (widx + 3 * node.nid) % 12 == 0:
        payload[int(rng.integers(K_SUB))] += int(rng.integers(40, 90))
    pkts = []
    for k in range(K_SUB):
        rem = int(payload[k])
        while rem > 0:
            c = min(rem, MTU_PAYLOAD)
            pkts.append(((k + 0.5) * DT_SUB, HDR_BYTES + c))
            rem -= c
    return pkts


# =============================================================================
# 7. RADIO TRANSPORT (SimPy)  ->  packet trace  ->  13 features
# =============================================================================
def packet_error_prob(sinr_db: float, nbytes: int) -> float:
    """Non-coherent FSK BER = 0.5 exp(-SNR/2); PER = 1-(1-BER)^bits."""
    ber = 0.5 * math.exp(-0.5 * 10 ** (min(sinr_db, 40.0) / 10))
    return 1.0 - (1.0 - ber) ** (8 * nbytes)


class _Trace:
    __slots__ = ('up', 'dn', 'synack', 'ackdat', 'hs_ok')

    def __init__(self):
        self.up, self.dn, self.synack, self.ackdat, self.hs_ok = [], [], 0.0, 0.0, False


def _gw_service(w: World):
    with w.gw_cpu.request() as rq:
        yield rq
        yield w.env.timeout(float(w.rng.lognormal(math.log(w.cfg.radio.gw_service_s), 0.4)))


def _tx(w: World, node: Node, nbytes: int, rssi: float, tr: _Trace, direction: str):
    """One frame: random backoff -> CCA (back-off while jammed) -> hold shared channel -> SINR/PER draw.
    Trace is a tap at the *node* radio: every transmitted uplink frame is counted; downlink only if received."""
    env, rad, rng = w.env, w.cfg.radio, w.rng
    tx_pos = node.pos if direction == 'up' else w.gw
    rx_pos = w.gw if direction == 'up' else node.pos
    yield env.timeout(int(rng.integers(0, 8)) * rad.slot_s)
    be = 3
    for _ in range(5):
        if w.interference_dbm(tx_pos, env.now) < rad.cca_dbm:
            break
        yield env.timeout(int(rng.integers(0, 2 ** be)) * rad.slot_s)
        be = min(be + 1, 7)
    else:
        return False                                              # channel-access failure
    with w.channel.request() as req:
        yield req
        yield env.timeout((nbytes + rad.phy_overhead_bytes) * 8.0 / rad.bitrate)
    i_rx = w.interference_dbm(rx_pos, env.now)
    noise_mw = 10 ** (rad.noise_floor_dbm / 10) + (10 ** (i_rx / 10) if i_rx > -150 else 0.0)
    sinr = rssi + float(rng.normal(0, 1.2)) - 10 * math.log10(noise_mw)
    ok = rng.random() > packet_error_prob(sinr, nbytes)
    if direction == 'up':
        tr.up.append((env.now, nbytes))
    elif ok:
        tr.dn.append((env.now, nbytes))
    return ok


def _handshake(w: World, node: Node, rssi: float, tr: _Trace):
    env, rad, rng = w.env, w.cfg.radio, w.rng
    t_syn = env.now
    for a in range(4):
        if (yield from _tx(w, node, HDR_BYTES, rssi, tr, 'up')):
            yield from _gw_service(w)
            if (yield from _tx(w, node, HDR_BYTES, rssi, tr, 'dn')):
                tr.synack = env.now - t_syn
                t_sa = env.now
                yield env.timeout(float(rng.uniform(0.0003, 0.001)))      # node stack turnaround
                yield from _tx(w, node, HDR_BYTES, rssi, tr, 'up')
                tr.ackdat = env.now - t_sa
                tr.hs_ok = True
                return True
        yield env.timeout(rad.rto_s * (2 ** a) * 1.5)
    return False


def _send_data(w: World, node: Node, nbytes: int, rssi: float, tr: _Trace, want_ack: bool):
    env, rad = w.env, w.cfg.radio
    for attempt in range(4):
        if (yield from _tx(w, node, nbytes, rssi, tr, 'up')):
            if not want_ack:
                return True
            yield from _gw_service(w)
            if (yield from _tx(w, node, HDR_BYTES, rssi, tr, 'dn')):
                return True
        yield env.timeout(rad.rto_s * (2 ** attempt))
    return False


def _flow_features(tr: _Trace, t_open: float) -> Dict[str, float]:
    up_t = np.array([t for t, _ in tr.up]); dn_t = np.array([t for t, _ in tr.dn])
    t_end = max(up_t[-1] if len(up_t) else t_open, dn_t[-1] if len(dn_t) else t_open)
    dur = max(float(t_end - t_open), 1e-3)
    spk, dpk = len(tr.up), len(tr.dn)
    sb, db = sum(b for _, b in tr.up), sum(b for _, b in tr.dn)
    sj = float(np.std(np.diff(up_t)) * 1e3) if spk >= 3 else 0.0
    dj = float(np.std(np.diff(dn_t)) * 1e3) if dpk >= 3 else 0.0
    return dict(dur=dur, spkts=spk, dpkts=dpk, sbytes=sb, dbytes=db, sload=sb * 8.0 / dur, dload=db * 8.0 / dur,
                sjit=sj, djit=dj, tcprtt=(tr.synack + tr.ackdat) if tr.hs_ok else 0.0,
                synack=tr.synack if tr.hs_ok else 0.0, ackdat=tr.ackdat if tr.hs_ok else 0.0,
                rate=(spk + dpk - 1) / dur)


def uplink_session(w: World, node: Node, pkts: List[Tuple[float, int]], meta: dict):
    """One report epoch = one flow: SYN/SYN-ACK/ACK, paced data frames (ACK every 2nd + last), FIN."""
    env = w.env
    yield env.timeout(pkts[0][0])
    t_open = env.now
    rssi = w.base_rssi(node, t_open)
    tr = _Trace()
    if (yield from _handshake(w, node, rssi, tr)):
        first, unacked = pkts[0][0], 0
        for i, (rel, nbytes) in enumerate(pkts):
            wait = (t_open + (rel - first)) - env.now
            if wait > 0:
                yield env.timeout(wait)
            if env.now - t_open > 8.0:
                break
            unacked += 1
            want_ack = unacked >= 2 or i == len(pkts) - 1
            ok = yield from _send_data(w, node, nbytes, rssi, tr, want_ack)
            if want_ack:
                unacked = 0
            if not ok:
                break
        if (yield from _tx(w, node, HDR_BYTES, rssi, tr, 'up')):                   # FIN
            yield from _gw_service(w)
            yield from _tx(w, node, HDR_BYTES, rssi, tr, 'dn')                      # FIN-ACK
    rec = _flow_features(tr, t_open)
    rec.update(meta)
    rec['t_start'] = t_open
    rec['phys_link_rssi_dbm'] = rssi
    w.records.append(rec)


def node_process(w: World, node: Node):
    env, rng, cfg = w.env, w.rng, w.cfg
    sense_fn = SENSE[cfg.region]
    r = cfg.radio
    yield env.timeout(node.phase)
    widx = 0
    while True:
        t0 = env.now
        yield env.timeout(WINDOW_S)
        node.shadow_db = (r.shadow_rho * node.shadow_db
                          + r.sigma_db * math.sqrt(1 - r.shadow_rho ** 2) * float(rng.normal()))
        s = sense_fn(w, node, t0)
        label, atype, scen = label_window(w, s)
        pkts = translate_to_packets(node, s, rng, widx)
        meta = dict(client=cfg.client, region=cfg.region, node_id=node.nid, window_idx=widx, window_start=t0,
                    scenario=scen, attack_type=atype, label=label,
                    **{f'phys_{k}': v for k, v in s.phys.items()})
        env.process(uplink_session(w, node, pkts, meta))
        widx += 1


# =============================================================================
# 8. EVENT FACTORIES (benign + hostile)       -> each returns an Event starting at time t
# =============================================================================
def _mk(w, kind, label, t, wp_t, wp_xyz, power, sig, impulses=None, impulse_db=None) -> Event:
    return Event(eid=w.new_eid(), kind=kind, label=label, t0=t, wp_t=np.asarray(wp_t, float),
                 wp_xyz=np.asarray(wp_xyz, float).reshape(-1, 3), power=np.asarray(power, float), sig=sig,
                 impulses=np.zeros(0) if impulses is None else np.asarray(impulses, float),
                 impulse_db=np.zeros(0) if impulse_db is None else np.asarray(impulse_db, float))


def _line(p0, p1, speed):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    T = max(float(np.linalg.norm(p1 - p0)) / speed, 1.0)
    return [0.0, T], [p0, p1]


def _stop_go(w, p0, d_unit, v_rng, cycles, move_rng, stop_rng, idle_pw, flip=False):
    """Stop-and-go trajectory: (move m s @ v) -> (idle s s) repeated."""
    rng = w.rng
    t, p = 0.0, np.asarray(p0, float)
    ts, ps, pw = [0.0], [p.copy()], [1.0]
    d = np.asarray(d_unit, float)
    for _ in range(cycles):
        m, v = rng.uniform(*move_rng), rng.uniform(*v_rng)
        t += m; p = p + d * v * m
        ts.append(t); ps.append(p.copy()); pw.append(1.0)
        s = rng.uniform(*stop_rng)
        t += s
        ts.append(t); ps.append(p.copy()); pw.append(idle_pw)
        if flip:
            d = -d
    return ts, ps, pw


# ---- Region A -----------------------------------------------------------------
UAV_TYPES = {  # MMAUD platform list; source level / spectral centroid are ASSUMED (see notes)
    'mavic2': (80.0, 520.0), 'mavic3': (81.0, 480.0), 'avata': (85.0, 650.0),
    'phantom4': (83.0, 430.0), 'm300': (92.0, 330.0)}


def ev_aircraft_takeoff(w, t):
    rng = w.rng
    d = rng.choice([-1, 1]); a = rng.uniform(2.6, 3.4); v_r = rng.uniform(65, 78); t_roll = v_r / a
    s_r = 0.5 * a * t_roll ** 2
    ts = [0, 4, 4 + t_roll * 0.5, 4 + t_roll, 4 + t_roll + 6, 4 + t_roll + 12]
    xs = -d * 420 + d * np.array([0, 0, 0.5 * a * (t_roll * 0.5) ** 2, s_r, s_r + v_r * 6, s_r + v_r * 12])
    zs = [0, 0, 0, 0, 25, 70]
    y = rng.uniform(-6, 6)
    sig = dict(spl_1m=rng.uniform(148, 155), f_hz=rng.uniform(450, 900), dT_src=rng.uniform(380, 520), r0=3.0,
               rcs_range=900.0, size_m=35.0, sat_contrast=0.40, rf_block_db=10.0)
    return _mk(w, 'aircraft_takeoff', 'Benign', t, ts, np.stack([xs, np.full(6, y), zs], 1),
               [0.25, 0.45, 0.75, 1.0, 1.0, 0.85], sig)


def ev_aircraft_landing(w, t):
    rng = w.rng
    d = rng.choice([-1, 1]); y = rng.uniform(-6, 6)
    ts = [0, 6, 9, 20, 32]
    xs = -d * 520 + d * np.array([0, 420, 610, 1000, 1160])
    zs = [70, 15, 0, 0, 0]
    sig = dict(spl_1m=rng.uniform(138, 146), f_hz=rng.uniform(350, 700), dT_src=rng.uniform(150, 260), r0=3.0,
               rcs_range=900.0, size_m=30.0, sat_contrast=0.38, rf_block_db=9.0)
    return _mk(w, 'aircraft_landing', 'Benign', t, ts, np.stack([xs, np.full(5, y), zs], 1),
               [0.5, 0.45, 0.6, 0.3, 0.2], sig)


def ev_ground_vehicle(w, t):
    rng = w.rng
    y = rng.choice([-1, 1]) * rng.uniform(12, 20)
    x0, x1 = rng.uniform(-350, 350), rng.uniform(-350, 350)
    ts, ps = _line([x0, y, 0.5], [x1, y, 0.5], rng.uniform(3, 9))
    sig = dict(spl_1m=rng.uniform(90, 98), f_hz=rng.uniform(100, 180), dT_src=70.0, r0=1.0,
               rcs_range=400.0, size_m=6.0, sat_contrast=0.25)
    return _mk(w, 'ground_vehicle', 'Benign', t, ts, ps, [0.8, 0.8], sig)


def ev_uav(w, t):
    rng = w.rng
    typ = str(rng.choice(list(UAV_TYPES)))
    spl, f0 = UAV_TYPES[typ]
    node = w.nodes[int(rng.integers(len(w.nodes)))]
    th = rng.uniform(0, 2 * math.pi)
    start = node.pos + np.array([280 * math.cos(th), 280 * math.sin(th), 0.0]); start[2] = 40.0
    rh, hz = rng.uniform(6, 20), rng.uniform(6, 25)
    ang = rng.uniform(0, 2 * math.pi)
    hov = node.pos + np.array([rh * math.cos(ang), rh * math.sin(ang), 0.0]); hov[2] = hz
    v = rng.uniform(6, 14)
    t_app = float(np.linalg.norm(hov - start)) / v
    t_hov, t_ret = rng.uniform(25, 60), rng.uniform(15, 30)
    away = hov + np.array([200 * math.cos(th + 0.8), 200 * math.sin(th + 0.8), 0.0])
    ts = [0, t_app, t_app + t_hov * 0.5, t_app + t_hov, t_app + t_hov + t_ret]
    ps = [start, hov, hov + rng.normal(0, 2, 3), hov + rng.normal(0, 2, 3), away]
    sig = dict(spl_1m=spl + rng.normal(0, 1.5), f_hz=f0 * rng.uniform(0.9, 1.1), dT_src=30.0, r0=0.3,
               rcs_range=60.0, uav=typ)
    return _mk(w, 'uav_' + typ, 'UAV_Intrusion', t, ts, ps, [0.9, 0.8, 0.75, 0.8, 0.9], sig)


def ev_gunfire(w, t):
    rng = w.rng
    node = w.nodes[int(rng.integers(len(w.nodes)))]
    th = rng.uniform(0, 2 * math.pi)
    pos = node.pos + np.array([rng.uniform(60, 220) * math.cos(th), rng.uniform(60, 220) * math.sin(th), 1.5 - node.pos[2]])
    times, tt = [], 0.0
    for _ in range(int(rng.integers(2, 6))):                      # engagement waves
        tt += rng.uniform(1.5, 5.0)
        n, rps = int(rng.integers(2, 7)), rng.uniform(8, 11)
        times += list(tt + np.arange(n) / rps)
        tt += n / rps
    T = tt + 2.0
    db = rng.normal(158.0, 3.0, len(times))                       # 149-167 dB peak @ ~1-4 m
    return _mk(w, 'gunfire', 'Gunshot_Fire', t, [0, T], [pos, pos], [0, 0], {},
               impulses=t + np.array(times), impulse_db=db)


def ev_overflight(w, t):
    rng = w.rng
    d = rng.choice([-1, 1]); y = rng.choice([-1, 1]) * rng.uniform(30, 120); z = rng.uniform(40, 120)
    v = rng.uniform(160, 230)
    ts, ps = _line([-d * 1000, y, z], [d * 1000, y, z], v)
    sig = dict(spl_1m=rng.uniform(153, 158), f_hz=rng.uniform(600, 1000), dT_src=rng.uniform(500, 700), r0=3.0,
               rcs_range=900.0, size_m=18.0, sat_contrast=0.35, rf_block_db=8.0)
    return _mk(w, 'hostile_overflight', 'Hostile_Overflight', t, ts, ps, [0.95, 0.95], sig)


# ---- Region B -----------------------------------------------------------------
def _truck_sig(rng):
    return dict(seis_a1m=18000.0 * rng.uniform(0.8, 1.2), seis_alpha=1.5, band=[0.10, 0.50, 0.40],
                mod_depth=0.20, mod_hz=0.7, phi=rng.uniform(0, 6.28), mag_K=6e4 * rng.uniform(0.8, 1.2),
                mag_sign=float(rng.choice([-1, 1])), pir_contrast=0.5, pir_range=12.0, rf_block_db=6.0)


def ev_supply_truck(w, t):
    rng = w.rng
    d = rng.choice([-1, 1]); y = rng.uniform(-2, 2)
    ts, ps = _line([-d * 450, y, 0], [d * 450, y, 0], rng.uniform(6, 11))
    return _mk(w, 'supply_truck', 'Benign', t, ts, ps, [1.0, 1.0], _truck_sig(rng))


def ev_foot_patrol(w, t):
    rng = w.rng
    n = int(rng.integers(3, 6)); d = rng.choice([-1, 1]); y = rng.uniform(2, 12) * rng.choice([-1, 1])
    x0 = rng.uniform(-250, 250)
    ts, ps = _line([x0, y, 0], [x0 + d * rng.uniform(120, 300), y, 0], rng.uniform(1.3, 1.6))
    sig = dict(seis_a1m=600.0 * math.sqrt(n), seis_alpha=1.5, band=[0.05, 0.20, 0.75], mod_depth=0.7,
               mod_hz=rng.uniform(1.6, 2.0), phi=rng.uniform(0, 6.28), mag_K=60.0 * n, mag_sign=float(rng.choice([-1, 1])),
               pir_contrast=0.8, pir_range=10.0)
    return _mk(w, 'foot_patrol', 'Benign', t, ts, ps, [1.0, 1.0], sig)


def ev_wildlife(w, t):
    rng = w.rng
    n = int(rng.integers(1, 4))
    p0 = np.array([rng.uniform(-250, 250), rng.uniform(-50, 50), 0.0])
    ts, ps, pw = _stop_go(w, p0, [rng.choice([-1, 1]), rng.uniform(-0.5, 0.5), 0], (0.5, 2.0), 5, (3, 10), (3, 12), 0.1)
    sig = dict(seis_a1m=250.0 * math.sqrt(n), seis_alpha=1.6, band=[0.10, 0.30, 0.60], mod_depth=0.9,
               mod_hz=rng.uniform(0.4, 1.5), phi=rng.uniform(0, 6.28), pir_contrast=0.7, pir_range=7.0)
    return _mk(w, 'wildlife', 'Benign', t, ts, ps, pw, sig)


def ev_infiltration(w, t):
    rng = w.rng
    n = int(rng.integers(2, 5))
    node = w.nodes[int(rng.integers(len(w.nodes)))]
    side = rng.choice([-1, 1])
    ts, ps, pw = _stop_go(w, [0.0, 0.0, 0.0], [0.0, -side, 0.0], (0.6, 1.0), 6, (8, 16), (6, 20), 0.12)
    end = np.array([node.pos[0] + rng.uniform(-15, 15), node.pos[1] + side * rng.uniform(6, 18), 0.0])
    start = end - (ps[-1] - ps[0])
    ps = [p + start for p in ps]
    sig = dict(seis_a1m=600.0 * math.sqrt(n) * rng.uniform(0.6, 0.9), seis_alpha=1.5, band=[0.03, 0.17, 0.80],
               mod_depth=0.8, mod_hz=rng.uniform(0.9, 1.5), phi=rng.uniform(0, 6.28), mag_K=90.0 * n,
               mag_sign=float(rng.choice([-1, 1])), pir_contrast=0.9, pir_range=10.0)
    return _mk(w, 'infiltration', 'Infiltration', t, ts, ps, pw, sig)


def ev_armored(w, t):
    rng = w.rng
    d = rng.choice([-1, 1]); y = rng.uniform(-6, 6)
    ts, ps = _line([-d * 420, y, 0], [d * 420, y, 0], rng.uniform(7, 13))
    sig = dict(seis_a1m=12000.0 * rng.uniform(0.8, 1.2), seis_alpha=1.3, band=[0.45, 0.35, 0.20], mod_depth=0.25,
               mod_hz=0.9, phi=rng.uniform(0, 6.28), mag_K=5e5 * rng.uniform(0.8, 1.2),
               mag_sign=float(rng.choice([-1, 1])), pir_contrast=0.6, pir_range=15.0, rf_block_db=10.0)
    return _mk(w, 'armored_incursion', 'Armored_Incursion', t, ts, ps, [1.0, 1.0], sig)


def ev_truck_mimicry(w, t):
    """Same per-sample physics as a supply truck (same sig generator) but a probing stop-and-go cadence."""
    rng = w.rng
    d = rng.choice([-1, 1]); y = rng.uniform(-2, 2)
    ts, ps, pw = _stop_go(w, [-d * 250, y, 0], [d, 0, 0], (6, 11), int(rng.integers(4, 8)), (4, 9), (3, 7), 0.20,
                          flip=bool(rng.random() < 0.5))
    return _mk(w, 'truck_mimicry', 'Truck_Mimicry', t, ts, ps, pw, _truck_sig(rng))


# ---- Region C -----------------------------------------------------------------
def ev_patrol_vehicle(w, t):
    rng = w.rng
    d = rng.choice([-1, 1]); y = rng.choice([-1, 1]) * rng.uniform(6, 10)
    ts, ps = _line([-d * 330, y, 0.5], [d * 330, y, 0.5], rng.uniform(5, 9))
    return _mk(w, 'patrol_vehicle', 'Benign', t, ts, ps, [1.0, 1.0],
               dict(pz_a1m=14.0 * rng.uniform(0.7, 1.3), pz_lambda=40.0, rf_block_db=6.0))


def ev_animal(w, t):
    rng = w.rng
    p = np.array([rng.uniform(-250, 250), rng.uniform(-1, 1), 0.3])
    T = rng.uniform(20, 60)
    n = int(rng.uniform(0.5, 1.5) * T)
    times = np.sort(t + rng.uniform(0, T, n))
    return _mk(w, 'animal', 'Benign', t, [0, T], [p, p], [0, 0], {}, impulses=times,
               impulse_db=rng.uniform(10, 40, n))


def _rf_sig(kind, tx, duty, period, n=2.5):
    return dict(rf_kind=kind, rf_tx_dbm=tx, rf_duty=duty, rf_period=period, rf_n=n)


def ev_rf_burst(w, t):
    rng = w.rng
    th = rng.uniform(0, 2 * math.pi); r = rng.uniform(400, 600)
    p = np.array([r * math.cos(th), r * math.sin(th), 5.0])
    T = rng.uniform(3, 10)
    return _mk(w, 'rf_burst', 'Benign', t, [0, T], [p, p], [0, 0],
               _rf_sig('burst', rng.uniform(-16, -8), rng.uniform(0.3, 1.0), rng.uniform(0.5, 2.0), n=2.5))


def ev_fence_breach(w, t):
    rng = w.rng
    node = w.nodes[int(rng.integers(len(w.nodes)))]
    p = np.array([node.pos[0] + rng.uniform(-45, 45), rng.uniform(-1, 1), 1.0])
    mode = str(rng.choice(['cut', 'climb']))
    times, amps, tt = [], [], 0.0
    if mode == 'cut':
        for _ in range(int(rng.integers(6, 13))):
            tt += rng.uniform(1.0, 3.5)
            n, f = int(rng.integers(4, 11)), rng.uniform(2, 4)
            times += list(tt + np.arange(n) / f); amps += list(rng.uniform(80, 200, n)); tt += n / f
    else:
        for _ in range(int(rng.integers(10, 20))):
            tt += rng.uniform(0.7, 1.3)
            times.append(tt); amps.append(rng.uniform(150, 400))
    T = tt + 3.0
    return _mk(w, 'fence_' + mode, 'Fence_Breach', t, [0, T], [p, p], [0, 0], {},
               impulses=t + np.array(times), impulse_db=np.array(amps))


def ev_jam(w, t):
    rng = w.rng
    th = rng.uniform(0, 2 * math.pi); r = rng.uniform(250, 700)
    p = np.array([r * math.cos(th), r * math.sin(th), 3.0])
    T = rng.uniform(25, 90)
    return _mk(w, 'rf_jamming', 'RF_Jamming', t, [0, T], [p, p], [0, 0],
               _rf_sig('jam', rng.uniform(22, 40), rng.uniform(0.35, 1.0), rng.uniform(0.2, 1.5)))


def ev_rogue(w, t):
    """Rogue/spoofed node: strong unknown beacon + frame flood that loads the shared channel (real contention)."""
    rng = w.rng
    th = rng.uniform(0, 2 * math.pi); r = rng.uniform(20, 120)
    p = np.array([r * math.cos(th), r * math.sin(th), 1.0])
    T = rng.uniform(25, 80)
    ev = _mk(w, 'rogue_emitter', 'Rogue_Emitter', t, [0, T], [p, p], [0, 0],
             _rf_sig('rogue', rng.uniform(-5, 10), 0.5, 0.5, n=2.7))
    w.env.process(_rogue_flood(w, ev, rate_hz=float(rng.uniform(20, 55))))
    return ev


def _rogue_flood(w: World, ev: Event, rate_hz: float):
    env, rad = w.env, w.cfg.radio
    while env.now < ev.t1:
        yield env.timeout(float(w.rng.exponential(1.0 / rate_hz)))
        with w.channel.request() as req:
            yield req
            yield env.timeout((60 + rad.phy_overhead_bytes) * 8.0 / rad.bitrate)


BENIGN_FACTORIES = {
    'aircraft_takeoff': ev_aircraft_takeoff, 'aircraft_landing': ev_aircraft_landing, 'ground_vehicle': ev_ground_vehicle,
    'supply_truck': ev_supply_truck, 'foot_patrol': ev_foot_patrol, 'wildlife': ev_wildlife,
    'patrol_vehicle': ev_patrol_vehicle, 'animal': ev_animal, 'rf_burst': ev_rf_burst}

# attack class -> (factory, events per campaign range, gap between events (s))
ATTACK_FACTORIES = {
    'UAV_Intrusion': (ev_uav, (2, 4), (5, 40)), 'Gunshot_Fire': (ev_gunfire, (1, 3), (10, 60)),
    'Hostile_Overflight': (ev_overflight, (1, 3), (8, 40)),
    'Infiltration': (ev_infiltration, (2, 4), (10, 60)), 'Armored_Incursion': (ev_armored, (1, 2), (10, 60)),
    'Truck_Mimicry': (ev_truck_mimicry, (1, 3), (10, 50)),
    'Fence_Breach': (ev_fence_breach, (2, 4), (10, 60)), 'RF_Jamming': (ev_jam, (1, 3), (10, 50)),
    'Rogue_Emitter': (ev_rogue, (1, 2), (10, 50))}


def benign_generator(env, w: World, kind: str, rate_h: float):
    while True:
        yield env.timeout(float(w.rng.exponential(3600.0 / rate_h)))
        w.add_event(BENIGN_FACTORIES[kind](w, env.now))


def _campaign(env, w: World, name: str, n_ev: int):
    fac, _, gap = ATTACK_FACTORIES[name]
    for _ in range(n_ev):
        ev = fac(w, env.now)
        w.add_event(ev)
        yield env.timeout(ev.t1 - ev.t0 + float(w.rng.uniform(*gap)))


def attack_generator(env, w: World):
    """Hostile campaigns arrive as a Poisson process and run concurrently (adversaries do not queue)."""
    names = list(w.cfg.attack_mix)
    probs = np.array([w.cfg.attack_mix[n] for n in names], float); probs /= probs.sum()
    while True:
        yield env.timeout(float(w.rng.exponential(3600.0 / w.cfg.attack_rate_h)))
        name = names[int(w.rng.choice(len(names), p=probs))]
        lo, hi = ATTACK_FACTORIES[name][1]
        n_ev = int(w.rng.integers(lo, hi + 1))
        w.campaigns.append((name, env.now, n_ev))
        env.process(_campaign(env, w, name, n_ev))


# =============================================================================
# 9. CLUSTERS (= federated clients D1..D6)  AND  DATASET BUILDER
# =============================================================================
def _radio(region: str, dn: float = 0.0, ds: float = 0.0) -> RadioCfg:
    base = {'A': RadioCfg(bitrate=1e6, tx_dbm=14, noise_floor_dbm=-108, n_exp=2.1, sigma_db=4.0),
            'B': RadioCfg(bitrate=250e3, tx_dbm=14, noise_floor_dbm=-114, n_exp=3.0, sigma_db=7.0),
            'C': RadioCfg(bitrate=100e3, tx_dbm=10, noise_floor_dbm=-118, n_exp=2.7, sigma_db=6.0)}[region]
    base.n_exp += dn; base.sigma_db += ds
    return base


# attack_mix = per-CAMPAIGN probabilities and attack_rate_h = campaigns/hour, both solved by calibrate_cluster() so
# attack windows ~20 % of each client with intended window shares (D1 UAV35/Gun65, D2 Ovf45/Gun35/UAV20, D3 Inf30/Mim70,
# D4 Arm65/Inf35, D5 Fence30/Jam70, D6 Jam40/Rogue40/Fence20).  Re-run calibrate_cluster() after changing any physics.
CLUSTERS: Dict[str, ClusterCfg] = {
    # Region A: air strip ------------------------------------------------------------------------
    'D1': ClusterCfg('D1', 'A', radio=_radio('A'), gw_pos=(0, -120, 4), t0_hour=9.0, wind_mean=4.5, amb_db=40.0,
                     benign_rate_h={'aircraft_takeoff': 5, 'aircraft_landing': 5, 'ground_vehicle': 10},
                     attack_mix={'UAV_Intrusion': 0.446, 'Gunshot_Fire': 0.554}, attack_rate_h=57.0),
    'D2': ClusterCfg('D2', 'A', radio=_radio('A', 0.2, 1.0), gw_pos=(0, 120, 4), t0_hour=14.0, wind_mean=3.5, amb_db=38.0,
                     benign_rate_h={'aircraft_takeoff': 4, 'aircraft_landing': 6, 'ground_vehicle': 14},
                     attack_mix={'Hostile_Overflight': 0.436, 'Gunshot_Fire': 0.383, 'UAV_Intrusion': 0.181}, attack_rate_h=41.5),
    # Region B: supply valley --------------------------------------------------------------------
    'D3': ClusterCfg('D3', 'B', radio=_radio('B'), gw_pos=(0, 90, 6), t0_hour=6.5, wind_mean=2.5, geo_noise_um=1.5,
                     benign_rate_h={'supply_truck': 2.0, 'foot_patrol': 6, 'wildlife': 14},
                     attack_mix={'Infiltration': 0.634, 'Truck_Mimicry': 0.366}, attack_rate_h=34.0),
    'D4': ClusterCfg('D4', 'B', radio=_radio('B', 0.2, 1.0), gw_pos=(0, -90, 6), t0_hour=16.0, wind_mean=2.0, geo_noise_um=1.3,
                     benign_rate_h={'supply_truck': 1.5, 'foot_patrol': 8, 'wildlife': 10},
                     attack_mix={'Armored_Incursion': 0.299, 'Infiltration': 0.701}, attack_rate_h=25.8),
    # Region C: perimeter ------------------------------------------------------------------------
    'D5': ClusterCfg('D5', 'C', radio=_radio('C'), gw_pos=(0, -50, 3), t0_hour=11.0, wind_mean=3.5, pz_noise_mv=1.5,
                     benign_rate_h={'patrol_vehicle': 6, 'animal': 14, 'rf_burst': 10},
                     attack_mix={'Fence_Breach': 0.837, 'RF_Jamming': 0.163}, attack_rate_h=33.7),
    'D6': ClusterCfg('D6', 'C', radio=_radio('C', 0.2, 1.0), gw_pos=(0, 50, 3), t0_hour=19.0, wind_mean=4.5, pz_noise_mv=1.8,
                     benign_rate_h={'patrol_vehicle': 8, 'animal': 10, 'rf_burst': 14},
                     attack_mix={'RF_Jamming': 0.114, 'Rogue_Emitter': 0.149, 'Fence_Breach': 0.737}, attack_rate_h=29.8),
}


def simulate_cluster(cfg: ClusterCfg, n_rows: int, seed: int, warmup_s: float = 60.0,
                     verbose: bool = False) -> pd.DataFrame:
    """Run one cluster's SimPy world until >= n_rows flow records exist; return them chronologically."""
    env = simpy.Environment()
    rng = np.random.default_rng(seed)
    w = World(env, cfg, rng)
    env.process(ambient_process(env, w))
    for kind, rate in cfg.benign_rate_h.items():
        env.process(benign_generator(env, w, kind, rate))
    env.process(attack_generator(env, w))
    for node in w.nodes:
        env.process(node_process(w, node))
    target = n_rows + int(warmup_s / WINDOW_S * cfg.n_nodes) + 60
    t_wall = time.time()
    while len(w.records) < target:
        env.run(until=env.now + 300.0)
        if verbose:
            print(f"    [{cfg.client}] sim t={env.now:7.0f}s  records={len(w.records):6d}/{target}", end='\r')
    env.run(until=env.now + 12.0)                                  # let open sessions finish
    df = pd.DataFrame(w.records).sort_values('t_start', kind='stable')
    df = df[df['t_start'] >= warmup_s].iloc[:n_rows].reset_index(drop=True)
    if verbose:
        print(f"    [{cfg.client}] {len(df)} rows  |  sim {env.now/3600:.2f} h  |  wall {time.time()-t_wall:.0f}s" + ' ' * 20)
    df.attrs['campaigns'] = list(w.campaigns)
    return df


def _finalize(raw: pd.DataFrame, domain: str) -> pd.DataFrame:
    df = normalize_features(raw)
    df['label'] = df['label'].astype(int)
    df['source_domain'] = domain
    keep = FEATURES + ['attack_type', 'label', 'source_domain', 'client', 'region', 'node_id', 'scenario', 't_start']
    return df[keep]


def build_tactical_partitions(rows_per_client: int = 20000, eval_total: int = 35451, seed: int = 2026,
                              clusters: Optional[Dict[str, ClusterCfg]] = None, verbose: bool = True):
    """
    Returns (partitions, df_eval, raw)
      partitions : {'D1'..'D6': DataFrame}  13 normalised features + attack_type/label/source_domain
                   (+ client, region, node_id, scenario, t_start for the temporal guard). Chronological per node.
      df_eval    : held-out episodes (independent seeds) from all six clusters, same schema.
      raw        : {'D1'..'D6': DataFrame}  un-normalised features + all physical sensor metrics (for the paper).
    """
    clusters = clusters or CLUSTERS
    partitions, raw, ev_parts = {}, {}, []
    per_eval = int(math.ceil(eval_total / len(clusters)))
    for i, (cid, cfg) in enumerate(clusters.items()):
        if verbose:
            print(f"-> simulating {cid}  (Region {cfg.region}, {cfg.n_nodes} nodes)")
        raw[cid] = simulate_cluster(cfg, rows_per_client, seed + 101 * i, verbose=verbose)
        partitions[cid] = _finalize(raw[cid], f'REGION_{cfg.region}')
        ev_raw = simulate_cluster(cfg, per_eval, seed + 9000 + 101 * i, verbose=False)
        ev_parts.append(_finalize(ev_raw, f'REGION_{cfg.region}'))
    df_eval = pd.concat(ev_parts, ignore_index=True).iloc[:eval_total]
    return partitions, df_eval, raw


REGION_CLIENTS = {'A': ['D1', 'D2'], 'B': ['D3', 'D4'], 'C': ['D5', 'D6']}


def load_region_data(region: str, rows_per_client: int = 20000, eval_rows: int = 6000, seed: int = 2026,
                     verbose: bool = True):
    """Generate + preprocess ONE region (two clients). Returns (partitions, df_eval, raw) for that region only.
    Uses the same per-client seeds as build_tactical_partitions, so a client's data is identical either way."""
    reg = region.upper()
    ids = REGION_CLIENTS[reg]
    all_ids = list(CLUSTERS)
    parts, raw, ev = {}, {}, []
    for cid in ids:
        i = all_ids.index(cid)
        if verbose:
            print(f"-> Region {reg} / {cid}")
        raw[cid] = simulate_cluster(CLUSTERS[cid], rows_per_client, seed + 101 * i)
        parts[cid] = _finalize(raw[cid], f'REGION_{reg}')
        ev.append(_finalize(simulate_cluster(CLUSTERS[cid], int(math.ceil(eval_rows / len(ids))), seed + 9000 + 101 * i),
                            f'REGION_{reg}'))
    return parts, pd.concat(ev, ignore_index=True).iloc[:eval_rows], raw


def audit_partitions(partitions: Dict[str, pd.DataFrame], df_eval: Optional[pd.DataFrame] = None):
    """Counterpart of the notebook's blueprint audit: sizes, benign share, class ledger, feature contract."""
    ok = True
    for cid, df in partitions.items():
        vc = df['attack_type'].value_counts()
        print(f"{cid} ({df['source_domain'].iloc[0]}): n={len(df):,}  benign={vc.get('Benign', 0)/len(df):.1%}")
        for k, v in vc.items():
            print(f"    {k:<20}{v:>7,}  ({v/len(df):.2%})")
        bad = [c for c in FEATURES if df[c].isna().any() or df[c].min() < 0 or df[c].max() > 1]
        ok &= not bad
        if bad:
            print("    !! feature contract violated:", bad)
    if df_eval is not None:
        print(f"EVAL n={len(df_eval):,}  benign={(df_eval.label == 0).mean():.1%}  domains={df_eval.source_domain.value_counts().to_dict()}")
    print("feature contract (13 cols, float, in [0,1], no NaN):", "OK" if ok else "FAILED")


def calibrate_cluster(cfg: ClusterCfg, shares: Dict[str, float], target_frac: float = 0.20,
                      rows: int = 14000, iters: int = 3, seed: int = 23, verbose: bool = True):
    """
    Solve for (campaign weights, campaign rate/h) so that attack windows are `target_frac` of all windows with the
    intended per-class *window* shares.  Re-run this whenever you change a physical parameter.
    Step 1 measures windows-per-campaign y_c per class; step 2 sets weights ~ share_c / y_c; step 3 refines
    multiplicatively against the observed mix.
    """
    import copy
    names = list(shares)
    y = {}
    for c in names:
        t = copy.deepcopy(cfg); t.attack_mix = {c: 1.0}; t.attack_rate_h = 25.0
        df = simulate_cluster(t, rows, seed)
        tmax = df.t_start.max()
        n_c = max(sum(1 for x in df.attrs['campaigns'] if x[1] < tmax), 1)
        y[c] = max((df.attack_type == c).sum() / n_c, 1.0)
    sh = {c: shares[c] / sum(shares.values()) for c in names}
    wts = {c: sh[c] / y[c] for c in names}
    tot = sum(wts.values()); wts = {c: v / tot for c, v in wts.items()}
    eyield = sum(wts[c] * y[c] for c in names)
    rate = target_frac * cfg.n_nodes * 1800.0 / eyield
    for it in range(iters):
        t = copy.deepcopy(cfg); t.attack_mix = dict(wts); t.attack_rate_h = rate
        df = simulate_cluster(t, rows, seed + 1 + it)
        frac = float(df.label.mean())
        obs = {c: max((df.attack_type == c).sum() / max((df.label == 1).sum(), 1), 0.02) for c in names}
        if verbose:
            print(f"   [{cfg.client}] iter {it}: attack frac {frac:.3f}  shares " + ", ".join(f"{c}={obs[c]:.2f}" for c in names))
        wts = {c: wts[c] * sh[c] / obs[c] for c in names}
        tot = sum(wts.values()); wts = {c: v / tot for c, v in wts.items()}
        rate *= target_frac / max(frac, 0.01)
    return wts, rate


# =============================================================================
# 10. CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Generate the tactical D1..D6 datasets with the SimPy battlefield simulator")
    ap.add_argument('--rows', type=int, default=20000, help='flow records per client')
    ap.add_argument('--eval', type=int, default=35451, help='total held-out evaluation rows')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', type=str, default='tactical_data')
    ap.add_argument('--raw', action='store_true', help='also save un-normalised features + physical metrics')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts, ev, raw = build_tactical_partitions(a.rows, a.eval, a.seed)
    audit_partitions(parts, ev)
    for cid, df in parts.items():
        df.to_csv(os.path.join(a.out, f'{cid}.csv'), index=False)
        if a.raw:
            raw[cid].to_csv(os.path.join(a.out, f'{cid}_raw_physical.csv'), index=False)
    ev.to_csv(os.path.join(a.out, 'eval.csv'), index=False)
    print(f"saved to ./{a.out}/")


# -----------------------------------------------------------------------------
PARAMETER_NOTES = """
LITERATURE-BACKED
  MMAUD (Yuan et al., arXiv 2402.03706): platform list Mavic2/Mavic3/Avata/Phantom4/M300; audio + lidar + radar
      modalities; ambient heavy-machinery noise sequences; UAV LiDAR hits unlikely beyond ~70 m (-> rcs_range=60 m).
  AeroSonicDB YPAD-0523 (arXiv 2311.06368): 625 low-flying aircraft recordings, 18-60 s event length, 3 km / 1 km
      trigger distances, 10 s "no aircraft" background clips (-> 20-60 s aircraft events + benign background).
  Gunshot acoustics: peak SPL 140-175 dB, typical rifles 150-165 dB, impulse < ~2 ms (CDC/NIOSH impulse-noise
      papers; Nash, Internoise; MIL-STD-1474 1 m measurements)  -> impulse_db ~ N(158, 3) @1 m, 2 ms pulse.
  Seismic UGS: land-vehicle band ~5-20 Hz; tracked vehicles add a 1.5-8.5 Hz Rayleigh band, wheeled 20-150 Hz;
      footsteps ~0.1-50 Hz (US 4975891, EP 0213619, US 8077036)  -> band[] weights.
  Radio: log-distance path loss with log-normal shadowing; measured LoRa exponents 1.6-2.3 (open) with
      sigma up to ~10 dB (Bonn study, FAST-LoRa, ns-3 LoRaWAN)  -> n=2.1/2.7/3.0, sigma=4/6/7 dB.

ASSUMED / ENGINEERING JUDGEMENT  (calibrate before claiming quantitative realism in the paper)
  UAV source levels & spectral centroids; jet SPL@1 m and exhaust dT; geophone a1m amplitudes; magnetic moments;
  PIR contrasts; piezo amplitudes; node placement; event arrival rates; every transport constant (bit rates,
  CSMA slots, RTO, gateway service time); send-on-delta thresholds and payload sizes.
"""

if __name__ == '__main__':
    main()
