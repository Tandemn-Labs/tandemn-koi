#!/usr/bin/env python3
"""
simulation/sim_ctl.py — Interactive control panel for mock Orca.

Usage:
    python simulation/sim_ctl.py [--orca http://localhost:26336] [--koi http://localhost:8090]

Commands:
    state                          — show simulator + Koi state
    kill <replica_id>              — kill a replica (simulates EC2 death)
    tps <replica_id> <value>       — set replica base TPS
    add                            — add a new replica (instant)
    complete                       — force job completion
    koi                            — show Koi /health + /jobs
    poll [seconds]                 — continuous polling (default 10s)
    help                           — show this help
    quit                           — exit
"""

import json
import sys
import time

import requests

ORCA = "http://localhost:26336"
KOI = "http://localhost:8090"


def get(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def post(url, data=None):
    try:
        r = requests.post(url, json=data or {}, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def show_state():
    print("\n=== Mock Orca State ===")
    state = get(f"{ORCA}/sim/state")
    for jid, info in state.items():
        print(f"  Job: {jid} [{info['status']}]")
        print(f"    Chunks: {info['chunks']}  |  Aggregate TPS: {info['aggregate_tps']}")
        print(f"    Replicas: {info['replicas_alive']} alive / {info['replicas_total']} total")
        for rid, r in info.get("replicas", {}).items():
            status = "ALIVE" if r["phase"] == "running" else r["phase"].upper()
            print(f"      {rid}: {status}  TPS={r['tps']:.0f}  {r['gpu']} TP={r['tp']} PP={r['pp']}")

    print("\n=== Koi State ===")
    health = get(f"{KOI}/health")
    print(f"  Health: {health.get('status', '?')}  |  Tracked: {health.get('tracked_jobs', 0)}  |  "
          f"Decisions: {health.get('memory_decisions', 0)}  Outcomes: {health.get('memory_outcomes', 0)}")
    jobs = get(f"{KOI}/jobs")
    for j in jobs.get("jobs", []):
        print(f"  {j['job_id']}: {j['status']}  TPS={j['smoothed_tps']:.0f}  "
              f"headroom={j['slo_headroom_pct']:.0f}%  elapsed={j['elapsed_hours']:.2f}h  "
              f"done={j['tokens_completed']:,}/{j['tokens_completed']+j['tokens_remaining']:,}")
    print()


def show_koi():
    print("\n=== Koi Health ===")
    print(json.dumps(get(f"{KOI}/health"), indent=2))
    print("\n=== Koi Jobs ===")
    print(json.dumps(get(f"{KOI}/jobs"), indent=2))
    print()


def poll(interval=10):
    print(f"Polling every {interval}s (Ctrl+C to stop)\n")
    try:
        i = 0
        while True:
            i += 1
            state = get(f"{ORCA}/sim/state")
            jobs = get(f"{KOI}/jobs")
            ts = time.strftime("%H:%M:%S")

            for jid, info in state.items():
                alive = info["replicas_alive"]
                total = info["replicas_total"]
                print(f"[{ts} #{i:>3}] {jid}: {info['chunks']}  aggTPS={info['aggregate_tps']:.0f}  "
                      f"replicas={alive}/{total}")

            for j in jobs.get("jobs", []):
                print(f"  koi/{j['job_id']}: {j['status']}  TPS={j['smoothed_tps']:.0f}  "
                      f"headroom={j['slo_headroom_pct']:.0f}%")

            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped polling.")


def main():
    global ORCA, KOI
    args = sys.argv[1:]
    if "--orca" in args:
        ORCA = args[args.index("--orca") + 1]
    if "--koi" in args:
        KOI = args[args.index("--koi") + 1]

    print(f"Mock Orca: {ORCA}  |  Koi: {KOI}")
    print("Type 'help' for commands.\n")

    while True:
        try:
            line = input("sim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd == "state":
            show_state()
        elif cmd == "koi":
            show_koi()
        elif cmd == "kill" and len(parts) >= 2:
            print(post(f"{ORCA}/sim/kill-replica/{parts[1]}"))
        elif cmd == "tps" and len(parts) >= 3:
            print(post(f"{ORCA}/sim/set-tps/{parts[1]}", {"tps": float(parts[2])}))
        elif cmd == "add":
            print(post(f"{ORCA}/sim/add-replica"))
        elif cmd == "complete":
            print(post(f"{ORCA}/sim/complete"))
        elif cmd == "poll":
            interval = int(parts[1]) if len(parts) > 1 else 10
            poll(interval)
        else:
            print(f"Unknown command: {cmd}. Type 'help'.")


if __name__ == "__main__":
    main()
