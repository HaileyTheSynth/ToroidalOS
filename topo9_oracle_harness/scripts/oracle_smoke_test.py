from __future__ import annotations
import argparse
from scripts.oracle import OracleConfig, Topo9Oracle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", required=True)
    args = ap.parse_args()

    cfg = OracleConfig(iso_path=args.iso)
    with Topo9Oracle(cfg) as o:
        print("BOOT OK")
        print(o.send_cmd("HELP")[:400])
        print(o.send_cmd("LIST 6"))
        print(o.send_cmd("HEDGE LIST"))
        print(o.send_cmd("ACCESS 3 0"))
        print(o.send_cmd("ACCESS 5 1"))
        print(o.send_cmd("HEDGE ADD TOPIC 3 5 7 9"))
        print(o.send_cmd("COHERENT 3 5"))
        print(o.send_cmd("EVOLVE 20"))
        print(o.send_cmd("TICK 5"))
        print(o.send_cmd("CURVATURE"))
        print(o.send_cmd("BRIDGES 200"))

if __name__ == "__main__":
    main()
