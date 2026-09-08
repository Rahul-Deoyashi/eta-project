"""
build_dataset.py
-----------------
Joins datameet/railways raw files (stations.json, trains.json, schedules.json)
into a single clean, sequence-ordered, feature-ready CSV:

    data/processed/train_stops.csv

Each row = one stop of one train, with:
    train_number, train_name, train_type, zone
    stop_seq (0-indexed order within the train's journey)
    station_code, station_name, station_zone, lat, lon
    day (day-of-journey, 1 = origin day)
    arrival, departure (HH:MM:SS strings, 'None' at origin/destination)
    halt_minutes (0 if it's a technical stop with no dwell)
    dist_from_prev_km (haversine distance from previous stop; NaN for origin)
    cum_distance_km (cumulative distance from origin)
    seg_travel_minutes (scheduled travel time from previous stop's departure
                         to this stop's arrival; NaN for origin)
"""

import json
import math
import csv
from datetime import datetime, timedelta

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "train_stops.csv")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_time(day, hhmmss):
    """Convert (day, 'HH:MM:SS') into a comparable absolute-minute value.
    day=1 is the origin day. Returns None if time is 'None'."""
    if hhmmss == "None" or hhmmss is None:
        return None
    h, m, s = map(int, hhmmss.split(":"))
    return (day - 1) * 24 * 60 + h * 60 + m


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("Loading raw files...")
    with open(f"{RAW}/stations.json") as f:
        stations_raw = json.load(f)["features"]
    with open(f"{RAW}/trains.json") as f:
        trains_raw = json.load(f)["features"]
    with open(f"{RAW}/schedules.json") as f:
        schedules_raw = json.load(f)

    # --- Lookups ---
    station_lookup = {}
    skipped_null_geom = 0
    for s in stations_raw:
        p = s["properties"]
        geom = s.get("geometry")
        if geom is None or geom.get("coordinates") is None:
            skipped_null_geom += 1
            continue
        coords = geom["coordinates"]  # [lon, lat]
        zone = p.get("zone")
        if zone in (None, "?", ""):
            zone = "UNK"
        station_lookup[p["code"]] = {
            "name": p.get("name"),
            "zone": zone,
            "lat": coords[1],
            "lon": coords[0],
        }

    train_lookup = {}
    for t in trains_raw:
        p = t["properties"]
        train_lookup[p["number"]] = {
            "name": p.get("name"),
            "type": p.get("type"),
            "zone": p.get("zone"),
            "distance_total_km": p.get("distance"),
        }

    print(f"Loaded {len(station_lookup)} stations (skipped {skipped_null_geom} with no "
          f"coordinates), {len(train_lookup)} trains, {len(schedules_raw)} schedule rows.")

    # --- Group schedule rows by train, sorted by 'id' (this gives correct stop order) ---
    print("Grouping and sorting stops per train...")
    by_train = {}
    for r in schedules_raw:
        by_train.setdefault(r["train_number"], []).append(r)

    for tn in by_train:
        by_train[tn].sort(key=lambda r: r["id"])

    print("Computing derived features and writing CSV...")
    fieldnames = [
        "train_number", "train_name", "train_type", "train_zone",
        "stop_seq", "station_code", "station_name", "station_zone",
        "lat", "lon", "day", "arrival", "departure",
        "halt_minutes", "dist_from_prev_km", "cum_distance_km",
        "seg_travel_minutes",
    ]

    skipped_missing_station = 0
    rows_written = 0

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for train_number, stops in by_train.items():
            tinfo = train_lookup.get(train_number, {})
            cum_distance = 0.0
            prev_lat = prev_lon = None
            prev_depart_abs_min = None

            for seq, stop in enumerate(stops):
                code = stop["station_code"]
                sinfo = station_lookup.get(code)
                if sinfo is None:
                    skipped_missing_station += 1
                    continue

                lat, lon = sinfo["lat"], sinfo["lon"]

                # halt duration
                arr_min = parse_time(stop["day"], stop["arrival"])
                dep_min = parse_time(stop["day"], stop["departure"])
                if arr_min is not None and dep_min is not None:
                    halt = dep_min - arr_min
                else:
                    halt = 0

                # distance from previous stop
                if prev_lat is not None:
                    dist = haversine_km(prev_lat, prev_lon, lat, lon)
                    cum_distance += dist
                else:
                    dist = None  # origin

                # segment travel time = this arrival - previous departure
                if prev_depart_abs_min is not None and arr_min is not None:
                    seg_travel = arr_min - prev_depart_abs_min
                else:
                    seg_travel = None

                writer.writerow({
                    "train_number": train_number,
                    "train_name": tinfo.get("name", stop.get("train_name")),
                    "train_type": tinfo.get("type"),
                    "train_zone": tinfo.get("zone"),
                    "stop_seq": seq,
                    "station_code": code,
                    "station_name": sinfo["name"],
                    "station_zone": sinfo["zone"],
                    "lat": lat,
                    "lon": lon,
                    "day": stop["day"],
                    "arrival": stop["arrival"],
                    "departure": stop["departure"],
                    "halt_minutes": halt,
                    "dist_from_prev_km": round(dist, 3) if dist is not None else "",
                    "cum_distance_km": round(cum_distance, 3),
                    "seg_travel_minutes": seg_travel if seg_travel is not None else "",
                })
                rows_written += 1

                prev_lat, prev_lon = lat, lon
                prev_depart_abs_min = dep_min if dep_min is not None else arr_min

    print(f"Done. Rows written: {rows_written}. Skipped (missing station): {skipped_missing_station}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
