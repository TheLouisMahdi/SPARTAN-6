"""Synthetic temporal traces against the Python reference, RTL simulation or real UART."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PC"))
from blink_timing import BinocularTimer, quantize_score


def traces():
    """Named (expected totals, expected long totals, capture-ms/score pairs), no human data."""
    return {
        "fast_one_closed_sample": (1, 0, [(0,0,0),(33,1,1),(66,0,0)]),
        "below_30ms": (0, 0, [(0,0,0),(33,1,1),(53,0,0)]),
        "slow_2_seconds": (1, 1, [(0,0,0)] + [(t,1,1) for t in range(33,2034,100)] + [(2066,0,0)]),
        "asymmetric_overlap": (1, 0, [(0,0,0),(33,1,0),(66,1,1),(99,0,1),(132,0,0)]),
        "late_second_eye": (0, 0, [(0,0,0),(33,1,0),(133,1,0),(199,1,1),(232,0,1),(265,0,0)]),
        "wink": (0, 0, [(0,0,0),(33,1,0),(66,1,0),(99,0,0)]),
        "crossed_winks": (0, 0, [(0,0,0),(33,1,0),(66,0,1),(99,0,0)]),
        "missing_eye": (0, 0, [(0,0,0),(33,1,1),(50,None,1),(66,0,0)]),
        "capture_gap": (0, 0, [(0,0,0),(33,1,1),(500,0,0)]),
        "start_closed": (0, 0, [(0,1,1),(33,1,1),(66,0,0)]),
        "gap_250_boundary": (1, 0, [(0,0,0),(33,1,1),(283,0,0)]),
        "gap_251_rejected": (0, 0, [(0,0,0),(33,1,1),(284,0,0)]),
        "unknown_deadband": (0, 0, [(0,.5,.5),(33,1,1),(66,0,0)]),
        "hysteresis": (1, 0, [(0,0,0),(33,.55,.55),(66,1,1),(99,.45,.45),(132,0,0)]),
        "uint32_wrap": (1, 0, [(0xffffffe0,0,0),(0xfffffffe,1,1),(32,0,0)]),
        "sync_150_boundary": (1, 0, [(0,0,0),(30,1,0),(180,1,1),(210,0,1),(240,0,0)]),
        "reclosing_in_partial_cycle": (0, 0, [(0,0,0),(30,1,0),(60,1,1),(90,0,1),(120,1,1),(150,0,0)]),
        "two_cycles": (2, 0, [(0,0,0),(33,1,1),(66,0,0),(99,1,1),(132,0,0)]),
    }


def golden_rows():
    rows = []
    for case, (count, long_count, observations) in traces().items():
        timer = BinocularTimer()
        for index, (t, left, right) in enumerate(observations):
            result = timer.update(left, right, t)
            scores = [quantize_score(left), quantize_score(right)]
            rows.append(dict(case=case, reset=index==0, timestamp_ms=t,
                left=left,right=right, valid=sum((s is not None)<<i for i,s in enumerate(scores)),
                scores=[s or 0 for s in scores], expected=asdict(result)))
        assert (timer.count,timer.long_count)==(count,long_count),case
    return rows


def simulate(output, rows):
    vectors, expected = [], []
    for row in rows:
        vectors.append((int(row["reset"])<<66) | (row["timestamp_ms"]<<34) |
                       (row["valid"]<<32) | (row["scores"][0]<<16) | row["scores"][1])
        r = row["expected"]
        word=(int(r["event"])<<260) | (r["valid_mask"]<<258) | (r["closed_mask"]<<256)
        for index, name in enumerate(("count","long_count","left_current_ms","right_current_ms",
                                      "last_left_ms","last_right_ms","last_overlap_ms","last_duration_ms")):
            word |= r[name] << (224-32*index)
        expected.append(word)
    (output/"vectors.hex").write_text("".join(f"{v:017x}\n" for v in vectors))
    (output/"expected.hex").write_text("".join(f"{v:066x}\n" for v in expected))
    (output/"suite.vh").write_text(f"`define SAMPLES {len(rows)}\n")
    binary=ROOT/"RESULTS/tools/iverilog/ucrt64/bin"
    env=dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"])
    command=[str(binary/"iverilog.exe"),"-g2001","-I",str(output),"-s","tb_timing","-o","bench.vvp",
             str(ROOT/"FPGA/rtl/binocular_timer.v"),str(ROOT/"RESEARCH/tb_timing.v")]
    with (output/"compile.log").open("w") as log:
        subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    with (output/"simulation.log").open("w") as log:
        p=subprocess.run([str(binary/"vvp.exe"),"bench.vvp"],cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
    if p.returncode or "TIMING_PASS" not in (output/"simulation.log").read_text():
        raise RuntimeError("RTL timing mismatch; see simulation.log")


def board(output, rows, port, baud=115200):
    from fpga_host import FPGAClient
    client=FPGAClient(port,baud=baud)
    records=[]
    try:
        status=client.ping()
        if not status.get("binocular_timing"):
            raise RuntimeError("Program the new binocular timing image first")
        for index,row in enumerate(rows):
            if row["reset"]:client.reset_blinks()
            actual,rtt=client.eye_timing(row["left"],row["right"],row["timestamp_ms"],20000+index)
            records.append(dict(row, actual=asdict(actual), roundtrip_ms=rtt))
            if asdict(actual)!=row["expected"]:
                raise RuntimeError(f"Physical timing mismatch at {row['case']}/{row['timestamp_ms']}")
        return dict(port=port,hardware=status,bad_packets=client.reader.bad_packets)
    finally:
        (output/"board_rows.json").write_text(json.dumps(records,indent=2)+"\n")
        try:
            client.reset_blinks();client.invalidate(0)
        finally:client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("simulate","board"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",default="COM26")
    parser.add_argument("--baud",type=int,default=115200,help="Use 1000000 for the canonical pretrained firmware")
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(status="FAIL",mode=args.mode,scope="synthetic engineering evidence, not blink-detection accuracy")
    try:
        rows=golden_rows();(output/"goldens.json").write_text(json.dumps(rows,indent=2)+"\n")
        if args.mode=="simulate":simulate(output,rows)
        else:result.update(board(output,rows,args.port,args.baud))
        result.update(status="PASS",cases=len(traces()),samples=len(rows))
    except Exception as exc:
        result["error"]=str(exc)
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
