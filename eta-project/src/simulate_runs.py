"""
simulate_runs.py
-----------------
Generates synthetic real-time train-running data on top of train_stops.csv.

For a sample of trains, simulates N_DAYS of "actual" runs, injecting:
  - base speed noise (log-normal)
  - temporary speed restrictions (TSR) on random segments, persisting for
    a multi-day window (Markov-style persistence)
  - congestion effect, proxied by schedule density in a rolling time window
  - precedence penalties (lower-priority train type held up when overlapping
    with a higher-priority train on the same section)
  - weather effect (season/zone-based fog probability, North Indian winter
    fog belt gets elevated fog probability in Dec-Feb)
  - small Gaussian noise at every stop

Output: data/processed/simulated_runs.csv
  One row per (train_number, sim_date, stop_seq), with both the scheduled
  static fields and the new simulated/actual fields + ML feature columns.
"""

import csv
import math
import random
from collections import defaultdict
from datetime import date, timedelta

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train_stops.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "simulated_runs.csv")

N_DAYS = 30
N_TRAINS = 300
START_DATE = date(2026, 1, 1)  # winter start -> lets us show fog-belt effect
SEED = 42

random.seed(SEED)

# Priority ranking: lower number = higher priority
PRIORITY = {
    "Raj": 1, "Vande Bharat": 1, "VNDB": 1,
    "Shtb": 2, "Shatabdi": 2,
    "Drnt": 3, "Duronto": 3,
    "SF": 4, "Superfast": 4,
    "Exp": 5, "Express": 5, "Mail": 5,
    "Pass": 7, "Passenger": 7, "DEMU": 7, "MEMU": 7,
}
DEFAULT_PRIORITY = 6

# Fog-prone zones (North Indian winter fog belt, per the earlier research)
FOG_ZONES = {"NR", "NCR", "ECR", "NER", "NFR"}
FOG_MONTHS = {12, 1, 2}

# Cruise-speed baseline by train type, km/h (used only to sanity check /
# scale noise, not to recompute travel time from scratch)
TYPE_SPEED_FACTOR = {
    "Raj": 1.0, "VNDB": 1.05, "Shtb": 1.0, "Drnt": 0.95,
    "SF": 0.9, "Exp": 0.8, "Mail": 0.75, "Pass": 0.55,
    "DEMU": 0.6, "MEMU": 0.6,
}


def priority_of(train_type):
    return PRIORITY.get(train_type, DEFAULT_PRIORITY)


def load_stops():
    trains = defaultdict(list)
    with open(IN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trains[row["train_number"]].append(row)
    for tn in trains:
        trains[tn].sort(key=lambda r: int(r["stop_seq"]))
    return trains


def pick_sample_trains(trains, n):
    """Sample trains stratified roughly by type, so we get a mix, not just
    whatever sorts first."""
    by_type = defaultdict(list)
    for tn, stops in trains.items():
        ttype = stops[0]["train_type"] or "Unknown"
        by_type[ttype].append(tn)

    all_types = list(by_type.keys())
    random.shuffle(all_types)
    sample = []
    # round-robin across types until we hit n or run out
    idx = 0
    pools = {t: list(v) for t, v in by_type.items()}
    for pool in pools.values():
        random.shuffle(pool)
    while len(sample) < n and any(pools.values()):
        t = all_types[idx % len(all_types)]
        if pools[t]:
            sample.append(pools[t].pop())
        idx += 1
        if idx > 100000:
            break
    return sample


def generate_tsr_sections(trains, sample_trains, tsr_fraction=0.08):
    """Pre-select a set of (train_number, stop_seq) segment-start points that
    are under a TSR, each with a start day and duration (days) and severity."""
    tsr_segments = {}
    for tn in sample_trains:
        stops = trains[tn]
        n_segments = len(stops) - 1
        n_tsr = max(0, round(n_segments * tsr_fraction))
        chosen = random.sample(range(1, len(stops)), n_tsr) if n_segments > 0 else []
        for seq in chosen:
            start_day = random.randint(0, N_DAYS - 1)
            duration = random.randint(3, 10)  # TSR persists 3-10 days
            severity = random.choice([15, 30, 50])  # capped speed km/h
            tsr_segments[(tn, seq)] = (start_day, duration, severity)
    return tsr_segments


def weather_state(zone, sim_date):
    """Very simple Markov-ish weather draw per (zone, date). Not tracking
    day-to-day persistence across the whole network for simplicity; just a
    seasonally-weighted independent draw per station per day, which is
    sufficient for feature-generation purposes at prototype scale."""
    if zone in FOG_ZONES and sim_date.month in FOG_MONTHS:
        return random.choices(["clear", "foggy"], weights=[0.55, 0.45])[0]
    elif zone in FOG_ZONES:
        return random.choices(["clear", "foggy"], weights=[0.92, 0.08])[0]
    else:
        return random.choices(["clear", "foggy"], weights=[0.97, 0.03])[0]


def main():
    print("Loading static stop skeleton...")
    trains = load_stops()
    print(f"Loaded {len(trains)} trains total.")

    sample_trains = pick_sample_trains(trains, N_TRAINS)
    print(f"Sampled {len(sample_trains)} trains for simulation.")

    print("Pre-generating TSR segments...")
    tsr_segments = generate_tsr_sections(trains, sample_trains)
    print(f"Generated {len(tsr_segments)} TSR-affected segments.")

    fieldnames = [
        "train_number", "train_name", "train_type", "train_zone",
        "sim_date", "stop_seq", "station_code", "station_name", "station_zone",
        "lat", "lon", "dist_from_prev_km", "cum_distance_km",
        "scheduled_arrival_min", "actual_arrival_min", "delay_minutes",
        "halt_minutes", "tsr_active", "tsr_speed_kmph",
        "congestion_index", "precedence_penalty_min",
        "weather_state", "priority_rank",
    ]

    rows_written = 0
    print("Simulating runs...")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tn in sample_trains:
            stops = trains[tn]
            ttype = stops[0]["train_type"] or "Unknown"
            prio = priority_of(ttype)
            speed_factor = TYPE_SPEED_FACTOR.get(ttype, 0.75)

            for day_offset in range(N_DAYS):
                sim_date = START_DATE + timedelta(days=day_offset)

                cum_delay = 0.0  # minutes of accumulated delay, carried stop to stop
                sched_cum_min = 0.0  # scheduled cumulative minutes from origin

                for stop in stops:
                    seq = int(stop["stop_seq"])
                    seg_travel = stop["seg_travel_minutes"]
                    seg_travel = float(seg_travel) if seg_travel not in ("", None) else 0.0

                    sched_cum_min += seg_travel

                    # --- TSR effect ---
                    tsr_active = 0
                    tsr_speed = ""
                    key = (tn, seq)
                    if key in tsr_segments:
                        start_day, duration, severity = tsr_segments[key]
                        if start_day <= day_offset < start_day + duration:
                            tsr_active = 1
                            tsr_speed = severity
                            dist = stop["dist_from_prev_km"]
                            dist = float(dist) if dist not in ("", None) else 0.0
                            # extra minutes lost vs normal running at ~speed_factor*70kmh baseline
                            normal_speed = max(speed_factor * 70, 20)
                            if severity > 0 and dist > 0:
                                extra_min = dist * (60 / severity - 60 / normal_speed)
                                cum_delay += max(0.0, extra_min) * 1.3  # amplify slightly so TSR is a clear, learnable signal

                    # --- congestion effect (proxy: random draw scaled by stop density) ---
                    # Rare event per stop; higher-priority trains are less exposed
                    # (they get precedence at junctions, so lower congestion risk).
                    # IMPORTANT: congestion_index stored below reflects the ACTUAL
                    # congestion pressure at this stop (not just a dice roll used
                    # internally) so it's a meaningful ML feature.
                    congestion_roll = random.random()
                    congestion_threshold = 0.93 + 0.01 * max(0, 4 - prio)  # premium trains: rarer hits
                    congestion_hit = congestion_roll > min(congestion_threshold, 0.97)
                    if congestion_hit:
                        congestion_index = round(random.uniform(0.6, 1.0), 3)
                        cum_delay += random.uniform(3, 12) * (1.0 if prio >= 5 else 0.4)
                    else:
                        congestion_index = round(random.uniform(0.0, 0.4), 3)

                    # --- precedence penalty: only lower-priority trains get held up,
                    # and the higher the priority, the more protected the train is. ---
                    precedence_penalty = 0.0
                    if prio <= 2:
                        pass  # top-priority trains (Rajdhani/Vande Bharat/Shatabdi) essentially never held
                    elif prio <= 4 and random.random() < 0.01:
                        precedence_penalty = random.uniform(2, 8)
                        cum_delay += precedence_penalty
                    elif prio >= 5 and random.random() < 0.04:
                        precedence_penalty = random.uniform(3, 15)
                        cum_delay += precedence_penalty

                    # --- weather effect: premium trains still get delayed by fog
                    # (physics doesn't care about priority) but recover faster
                    # afterwards via a bigger recovery allowance below. ---
                    zone_for_weather = stop["station_zone"] if stop["station_zone"] != "UNK" else stop["train_zone"]
                    w_state = weather_state(zone_for_weather, sim_date)
                    if w_state == "foggy" and random.random() < 0.15:
                        cum_delay += random.uniform(3, 15)

                    # --- base noise (log-normal multiplicative on segment time) ---
                    if seg_travel > 0:
                        noise_factor = random.lognormvariate(0, 0.08)
                        noise_min = seg_travel * (noise_factor - 1)
                        cum_delay += noise_min

                    # small chance of an unscheduled signal halt (rarer for premium trains)
                    halt_prob = 0.01 if prio >= 4 else 0.004
                    if random.random() < halt_prob:
                        cum_delay += random.uniform(2, 12)

                    # --- recovery: on relatively loose (non-urban, longer) segments,
                    # a delayed train can claw back some time using timetable slack.
                    # Premium trains (Rajdhani/Shatabdi/Vande Bharat) get priority
                    # dispatching and can recover delay faster than lower-priority
                    # trains, reflecting real operational practice.
                    if cum_delay > 0 and seg_travel > 3:
                        recovery_rate = random.uniform(0.05, 0.14) if prio <= 2 else random.uniform(0.03, 0.10)
                        recovery = min(cum_delay, seg_travel * recovery_rate)
                        cum_delay -= recovery

                    cum_delay = max(0.0, cum_delay)
                    actual_arrival_min = sched_cum_min + cum_delay

                    writer.writerow({
                        "train_number": tn,
                        "train_name": stop["train_name"],
                        "train_type": ttype,
                        "train_zone": stop["train_zone"],
                        "sim_date": sim_date.isoformat(),
                        "stop_seq": seq,
                        "station_code": stop["station_code"],
                        "station_name": stop["station_name"],
                        "station_zone": stop["station_zone"],
                        "lat": stop["lat"],
                        "lon": stop["lon"],
                        "dist_from_prev_km": stop["dist_from_prev_km"],
                        "cum_distance_km": stop["cum_distance_km"],
                        "scheduled_arrival_min": round(sched_cum_min, 1),
                        "actual_arrival_min": round(actual_arrival_min, 1),
                        "delay_minutes": round(cum_delay, 1),
                        "halt_minutes": stop["halt_minutes"],
                        "tsr_active": tsr_active,
                        "tsr_speed_kmph": tsr_speed,
                        "congestion_index": congestion_index,
                        "precedence_penalty_min": round(precedence_penalty, 1),
                        "weather_state": w_state,
                        "priority_rank": prio,
                    })
                    rows_written += 1

    print(f"Done. Rows written: {rows_written}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
