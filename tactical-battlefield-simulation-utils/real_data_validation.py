#!/usr/bin/env python3
"""
real_data_validation.py
=======================
Loads the REAL datasets from the validation survey, converts them to the same physical units the simulator logs
(dBm, dB, probability, metres), and compares them with the simulator's raw physical output.

WHAT THIS CAN AND CANNOT SUPPORT (say this to your professor, it is what the evidence allows)
---------------------------------------------------------------------------------------------
  CAN:    (1) the simulator's radio parameters (path-loss exponent, shadowing sigma) sit inside ranges measured in
              real field/lab studies;  (2) jamming raises the in-band energy reading in real Wi-Fi spectral scans
              and in the simulator, in the same direction;  (3) real low-power links show a "grey zone" of
              intermediate packet success (Intel Lab), and the simulator's link model shows one too.
  CANNOT: that the 13 UNSW-style flow features equal real battlefield traffic.  No public battlefield sensor
          dataset with packet traces was found; the real sets are indoor / chamber / road / savanna.  The
          comparison therefore happens at the PHYSICAL layer, BEFORE the translation layer.

Real-data layouts that were NOT opened (WiFiSpectralJam folder structure, licences) are marked UNVERIFIED in the
docstrings; run `inspect_tree(folder)` on your download first and adapt `wsj_label_from_path` if needed.

Requires: numpy pandas scipy (matplotlib optional for plot_validation)
"""
from __future__ import annotations

import math
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------------------------------
# 1. PUBLISHED REFERENCE VALUES (from the survey; use when the raw files are not downloaded)
# --------------------------------------------------------------------------------------------------
# WiFiSpectralJam arXiv 2608.15728 (preprint), file-level means. Chamber, jammer 20 cm, no legit traffic.
WSJ_PUBLISHED = pd.DataFrame(
    [('active', 'benign', -88.06, 47.31, 81.42, -104.61), ('active', 'jammed', -73.05, 108.72, 66.45, -102.00),
     ('passive', 'benign', -86.86, 52.57, 80.16, -102.05), ('passive', 'jammed', -83.71, 52.16, 75.84, -102.68)],
    columns=['scan_mode', 'state', 'rssi_dbm', 'max_magnitude', 'total_gain_db', 'noise_floor_dbm'])

# Parameter ranges measured in the literature (survey table). scope -> (lo, hi, sources)
PARAM_EVIDENCE = {
    'n_exp': {
        'A': (1.95, 2.31, 'LOS: Widodo 2017 (2.4 GHz XBee), Boonlom 2026 (LoRa 923 MHz)'),
        'B': (3.22, 5.00, 'Obstructed: forest 3.22, millet field 4.15-5.0 (Salem 2026, Boonlom 2026)'),
        'C': (1.95, 3.58, 'Between LOS 1.95 and NLOS 3.58 (no direct perimeter measurement; interpolated)')},
    'sigma_db': {
        'A': (4.25, 5.30, 'Field 4.25 dB (Salem), campus up to 5.3 dB (Suarez 2025)'),
        'B': (4.25, 12.6, 'Field 4.25 dB to desert campus 12.6 dB (AlHammadi 2019)'),
        'C': (4.25, 12.6, 'Field 4.25 dB to desert campus 12.6 dB')},
}
TOL = {'n_exp': 0.30, 'sigma_db': 0.75}      # "near" = within this distance outside the measured range


# --------------------------------------------------------------------------------------------------
# 2. LOADERS + PRE-PROCESSING OF REAL DATA  (output units = simulator units)
# --------------------------------------------------------------------------------------------------
def inspect_tree(folder: str, max_files: int = 5):
    """Print folder layout and the head of a few CSVs, so you can confirm the real column names."""
    for root, _, files in os.walk(folder):
        csvs = [f for f in files if f.lower().endswith(('.csv', '.txt'))]
        print(f"{root}: {len(files)} files")
        for f in csvs[:max_files]:
            print('  ', f)
            try:
                print(pd.read_csv(os.path.join(root, f), nrows=3).to_string(max_cols=12))
            except Exception as e:                                     # noqa: BLE001
                print('   (unreadable as csv:', e, ')')


def load_intel_links(connectivity_txt: str, mote_locs_txt: str) -> pd.DataFrame:
    """
    Intel Berkeley Lab link table (licence: free to use with acknowledgement, per the dataset page).
    connectivity.txt rows: sender receiver P(success);  mote_locs.txt rows: moteid x y (metres).
    Pre-processing: drop receiver id 0 (unexplained in the docs), drop rows without a value (the stray '0 31'
    line), drop dead senders (all-zero outgoing rows, e.g. mote 5), attach metric distance.
    Returns columns: sender, receiver, p_success, dist_m.
    """
    loc = pd.read_csv(mote_locs_txt, sep=r'\s+', header=None, names=['moteid', 'x', 'y']).dropna()
    con = pd.read_csv(connectivity_txt, sep=r'\s+', header=None, names=['sender', 'receiver', 'p_success'],
                      engine='python', on_bad_lines='skip')
    con = con.dropna()
    con = con[(con.receiver != 0) & (con.sender != con.receiver)]
    alive = con.groupby('sender')['p_success'].sum()
    con = con[con.sender.isin(alive[alive > 0].index)]
    con = (con.merge(loc.add_prefix('s_'), left_on='sender', right_on='s_moteid')
              .merge(loc.add_prefix('r_'), left_on='receiver', right_on='r_moteid'))
    con['dist_m'] = np.hypot(con.s_x - con.r_x, con.s_y - con.r_y)
    return con[['sender', 'receiver', 'p_success', 'dist_m']].reset_index(drop=True)


def wsj_label_from_path(path: str) -> Tuple[str, str]:
    """UNVERIFIED heuristic: the real Kaggle folder layout was not opened. Edit after running inspect_tree()."""
    p = path.lower()
    return ('passive' if 'passive' in p else 'active', 'jammed' if 'jam' in p and 'benign' not in p else 'benign')


WSJ_COLS = ['freq1', 'noise', 'max_magnitude', 'total_gain_db', 'base_pwr_db', 'rssi', 'relpwr_db', 'avgpwr_db']


def load_wifispectraljam(folder: str, label_fn: Callable[[str], Tuple[str, str]] = wsj_label_from_path,
                         max_files: int = 400, seed: int = 0) -> pd.DataFrame:
    """
    One row per CSV file (the paper reports file-level means, so we do the same).
    Pre-processing: rssi_dbm = rssi - 95 (stated in the paper); noise is already dBm.
    Raw files have no timestamps (observation order only), so no time features are built.
    """
    paths = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f.lower().endswith('.csv')]
    rng = np.random.default_rng(seed)
    if len(paths) > max_files:
        paths = list(rng.choice(paths, max_files, replace=False))
    rows = []
    for p in paths:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in WSJ_COLS)
        except Exception:                                              # noqa: BLE001
            continue
        if not {'rssi', 'noise'} <= set(d.columns):
            continue
        mode, state = label_fn(p)
        rows.append(dict(file=p, scan_mode=mode, state=state, rssi_dbm=(d['rssi'] - 95.0).mean(),
                         noise_floor_dbm=d['noise'].mean(), max_magnitude=d.get('max_magnitude', pd.Series([np.nan])).mean(),
                         total_gain_db=d.get('total_gain_db', pd.Series([np.nan])).mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# 3. SIMULATOR-SIDE EXTRACTION
# --------------------------------------------------------------------------------------------------
def _radio_funcs():
    from tactical_battlefield_sim import PL0_DB, packet_error_prob          # lazy: sim needs simpy
    return PL0_DB, packet_error_prob


def sim_link_samples(radio, n_links: int = 5000, payload_bytes: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Packet success of simulated links with the simulator's OWN radio model (path loss + log-normal shadowing
    + non-coherent FSK PER). Distances are drawn uniformly from the audible range, like motes scattered in a lab.
    """
    PL0, per = _radio_funcs()
    rng = np.random.default_rng(seed)
    def p_of(d, shadow):
        sinr = radio.tx_dbm - PL0 - 10 * radio.n_exp * np.log10(d) - shadow - radio.noise_floor_dbm
        return np.array([1.0 - per(float(s), payload_bytes) for s in np.atleast_1d(sinr)])
    ds = np.geomspace(1.0, 1e6, 800)
    below = p_of(ds, 0.0) < 0.02
    d_edge = ds[np.argmax(below)] if below.any() else ds[-1]                      # no-shadow distance where PRR falls to 2 %
    d = rng.uniform(0.1 * d_edge, d_edge * 1.3, n_links)
    sh = rng.normal(0, radio.sigma_db, n_links)
    return pd.DataFrame({'dist_m': d, 'p_success': p_of(d, sh)})


def grey_zone_stats(p: np.ndarray, audible: float = 0.02, lo: float = 0.1, hi: float = 0.9) -> Dict[str, float]:
    """Among audible links, the fraction with intermediate success (the 'transitional region' of low-power links)."""
    p = np.asarray(p, float)
    a = p[p > audible]
    return dict(n_audible=int(a.size), grey_fraction=float(((a > lo) & (a < hi)).mean()) if a.size else np.nan,
                dead_fraction=float((p <= audible).mean()), good_fraction=float((a >= hi).mean()) if a.size else np.nan)


def sim_jamming_shift(raw_C: pd.DataFrame) -> pd.DataFrame:
    """
    Benign vs RF_Jamming windows in simulator Region C raw output (columns written by simulate_cluster).
    Mapping assumption (state it explicitly): the real 'rssi' field is TOTAL received energy, so it corresponds to
    the simulator's phys_noise_floor_dbm (= noise + jammer power in band), NOT to phys_rssi_dbm (wanted-link signal).
    """
    need = ['phys_noise_floor_dbm', 'phys_snr_db', 'phys_rssi_dbm']
    miss = [c for c in need if c not in raw_C.columns]
    if miss:
        raise KeyError(f"raw frame lacks {miss}; pass raw_C['D5'] / raw_C['D6'] (Region C)")
    ben = raw_C[raw_C.label == 0]
    jam = raw_C[raw_C.attack_type == 'RF_Jamming']
    rows = []
    for col, meaning in [('phys_noise_floor_dbm', 'in-band energy (real: rssi total energy)'),
                         ('phys_rssi_dbm', 'wanted link signal (real: n/a)'),
                         ('phys_snr_db', 'link SNR')]:
        b, j = ben[col].dropna(), jam[col].dropna()
        if len(b) and len(j):
            ks = stats.ks_2samp(b, j)
            rows.append(dict(sim_metric=col, meaning=meaning, benign_median=b.median(), jam_median=j.median(),
                             shift_db=j.median() - b.median(), ks_stat=ks.statistic, n_benign=len(b), n_jam=len(j)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# 4. COMPARISONS / AUDIT
# --------------------------------------------------------------------------------------------------
def parameter_audit(clusters: Dict[str, object]) -> pd.DataFrame:
    """Is each cluster's radio parameter inside the measured range for its terrain? (inside / near / outside)"""
    rows = []
    for cid, cfg in clusters.items():
        for par in ('n_exp', 'sigma_db'):
            v = getattr(cfg.radio, par)
            lo, hi, src = PARAM_EVIDENCE[par][cfg.region]
            gap = max(lo - v, v - hi, 0.0)
            status = 'inside' if gap == 0 else ('near' if gap <= TOL[par] else 'outside')
            rows.append(dict(client=cid, region=cfg.region, parameter=par, sim_value=v, measured_lo=lo,
                             measured_hi=hi, status=status, source=src))
    return pd.DataFrame(rows)


def jamming_comparison(sim_shift: pd.DataFrame, wsj: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Real jammed-minus-benign shifts (downloaded files if given, else the paper's published means) vs simulator."""
    ref = wsj if wsj is not None and len(wsj) else WSJ_PUBLISHED
    src = 'downloaded files' if ref is not WSJ_PUBLISHED else 'paper means (preprint)'
    rows = []
    for mode, g in ref.groupby('scan_mode'):
        m = g.groupby('state')[['rssi_dbm', 'noise_floor_dbm']].mean()
        if {'benign', 'jammed'} <= set(m.index):
            rows.append(dict(source=f'WiFiSpectralJam {mode} ({src})',
                             energy_shift_db=m.loc['jammed', 'rssi_dbm'] - m.loc['benign', 'rssi_dbm'],
                             noise_field_shift_db=m.loc['jammed', 'noise_floor_dbm'] - m.loc['benign', 'noise_floor_dbm']))
    e = sim_shift[sim_shift.sim_metric == 'phys_noise_floor_dbm']
    if len(e):
        rows.append(dict(source='SIMULATOR Region C (in-band energy)', energy_shift_db=float(e.shift_db.iloc[0]),
                         noise_field_shift_db=np.nan))
    return pd.DataFrame(rows)


LIMITATIONS = """\
Honest limits to state alongside the results
 1. No real battlefield dataset with packet traces exists in the survey; every real set is indoor, chamber, road,
    savanna or campus. Claim: "parameters and radio-layer behaviour are consistent with measured ranges",
    NOT "the generated data equals real battlefield data".
 2. WiFiSpectralJam / Ali RF Jamming are Wi-Fi spectral scans in an RF-shielded chamber (jammer 20 cm). Compare
    direction and ordering of the energy shift, not its magnitude. WiFiSpectralJam is a preprint; licence UNVERIFIED.
 3. Real receivers lower their gain under jamming (-15 dB in active scans); the simulator does not model this yet.
 4. Intel Lab is indoor 2004, averaged over 5 weeks (no time dynamics, no RSSI): it supports the existence and
    rough width of the link-quality grey zone only.
 5. Seismic, acoustic, PIR, magnetometer, thermal and piezo channels have no matched public data; they remain
    literature-parameterised (see PARAMETER_NOTES in the simulator).
"""


def validation_report(raw: Dict[str, pd.DataFrame], clusters: Dict[str, object],
                      intel_dir: Optional[str] = None, wsj_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """raw = {'D1'..'D6': raw frames} as returned by load_region_data / build_tactical_partitions."""
    rep: Dict[str, pd.DataFrame] = {'parameter_audit': parameter_audit(clusters)}
    rawC = pd.concat([raw[c] for c in raw if clusters[c].region == 'C'], ignore_index=True) if any(
        clusters[c].region == 'C' for c in raw) else None
    wsj = load_wifispectraljam(wsj_dir) if wsj_dir else None
    if rawC is not None:
        rep['sim_jamming_shift'] = sim_jamming_shift(rawC)
        rep['jamming_comparison'] = jamming_comparison(rep['sim_jamming_shift'], wsj)
    rows = []
    if intel_dir:
        il = load_intel_links(os.path.join(intel_dir, 'connectivity.txt'), os.path.join(intel_dir, 'mote_locs.txt'))
        rep['intel_links'] = il
        rows.append(dict(source='Intel Lab (real, indoor 2004)', **grey_zone_stats(il.p_success.values),
                         spearman_dist_p=float(stats.spearmanr(il.dist_m, il.p_success)[0])))
    for cid, cfg in clusters.items():
        s = sim_link_samples(cfg.radio)
        rows.append(dict(source=f'SIM {cid} (Region {cfg.region})', **grey_zone_stats(s.p_success.values),
                         spearman_dist_p=float(stats.spearmanr(s.dist_m, s.p_success)[0])))
    rep['link_grey_zone'] = pd.DataFrame(rows)
    return rep


def save_report(rep: Dict[str, pd.DataFrame], outdir: str = 'validation_out'):
    os.makedirs(outdir, exist_ok=True)
    for k, v in rep.items():
        v.to_csv(os.path.join(outdir, f'{k}.csv'), index=False)
    with open(os.path.join(outdir, 'LIMITATIONS.txt'), 'w') as f:
        f.write(LIMITATIONS)
    print(f'saved {len(rep)} tables + LIMITATIONS.txt to ./{outdir}/')


def plot_validation(rep: Dict[str, pd.DataFrame], path: str = 'validation_out/validation.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if 'intel_links' in rep:
        il = rep['intel_links']
        ax[0].scatter(il.dist_m, il.p_success, s=6, alpha=.4, label='Intel Lab links (real)')
    ax[0].set(xlabel='distance (m)', ylabel='P(success)', title='Real links: grey zone of intermediate success')
    ax[0].legend()
    jc = rep.get('jamming_comparison')
    if jc is not None and len(jc):
        ax[1].barh(jc.source, jc.energy_shift_db)
        ax[1].set(xlabel='jammed - benign shift in in-band energy (dB)', title='Jamming raises measured energy')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150)
    print('saved', path)
