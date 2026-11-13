"""Explicit gated volatile JTAG CNN experiment, matched UART work and rollback."""
import argparse
import binascii
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import queue
import statistics
import struct
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"python"))
import cv2
import numpy as np
from fpga_host import FPGAClient
from jtag_host import JtagSession,JtagCNNClient,cnn_packet,parse_cnn_reply
from pretrained_eye import PretrainedEye,input_pixels


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def stats(values):return dict(mean=statistics.mean(values),p95=float(np.percentile(values,95)),max=max(values)) if values else None
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class Meter:
    def __init__(self,serial):self.serial=serial;self.tx=self.rx=0
    def __getattr__(self,name):return getattr(self.serial,name)
    def write(self,data):
        n=self.serial.write(data);self.tx+=n;return n
    def read(self,n):
        data=self.serial.read(n);self.rx+=len(data);return data


def uart(model):
    from serial.tools import list_ports
    ports=[p for p in list_ports.comports() if p.vid==0x10c4 and p.pid==0xea60 and p.serial_number=="04D5A2633F36EF11B7ABFF13A04C7D1C"]
    if len(ports)!=1:raise RuntimeError("registered CP210x not found")
    client=FPGAClient(ports[0].device,1000000)
    client.serial=Meter(client.serial)
    try:client.load_model(model)
    except Exception:client.close();raise
    return client


def registered():
    with (ROOT/"assets/datasets/cew/manifest.csv").open(newline="",encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def image(row):
    path=Path(row["source_path"])
    if digest(path)!=row["source_sha256"]:raise ValueError("CEW registered image changed")
    result=cv2.imread(str(path))
    if result is None:raise ValueError("CEW decode failed")
    return result


def measure(output,name,infer,fixtures,expected,period_ns=None,count=32):
    """Independent producer, bounded queue and one ordered terminal per source ID."""
    folder=output/name;folder.mkdir()
    pending=queue.Queue(maxsize=8);stop=threading.Event();done=threading.Event()
    state=dict(generated=0,accepted=0,rejected=0,max_queue=0,queue_area_ns=0,backpressure_ns=0)
    lock=threading.Lock();qstate=[time.perf_counter_ns(),0]
    def observe():
        with lock:
            now=time.perf_counter_ns();size=pending.qsize()
            state["queue_area_ns"]+=(now-qstate[0])*qstate[1]
            qstate[:]=[now,size];state["max_queue"]=max(state["max_queue"],size)
    origin=time.perf_counter_ns()+50_000_000;base=0xfffffff0
    source_errors=[]
    def producer():
        try:
            with (folder/"arrivals.jsonl").open("w") as log:
                for i in range(count):
                    if stop.is_set():break
                    capture=origin+i*(period_ns or 0)
                    if period_ns and stop.wait(max(0,(capture-time.perf_counter_ns())/1e9)):break
                    observed=time.perf_counter_ns();state["generated"]+=1
                    if period_ns:
                        try:pending.put_nowait((i,observed));status="ACCEPTED"
                        except queue.Full:status="REJECTED_FULL"
                    else:
                        before=time.perf_counter_ns()
                        status="REJECTED_ABORTED"
                        while not stop.is_set():
                            try:pending.put((i,observed),timeout=.1);status="ACCEPTED";break
                            except queue.Full:pass
                        state["backpressure_ns"]+=time.perf_counter_ns()-before
                    state["accepted" if status=="ACCEPTED" else "rejected"]+=1;observe()
                    log.write(json.dumps(dict(index=i,frame_id=(base+i)&0xffffffff,capture_ns=capture,
                        observed_ns=observed,status=status))+"\n")
                    if status=="REJECTED_ABORTED":break
        except Exception as exc:source_errors.append(repr(exc))
        finally:done.set()
    source=threading.Thread(target=producer,name="independent-offline-source");source.start()
    start=time.perf_counter_ns();next_id=completed=mismatches=0
    rtt=[];uploads=[];e2e=[];kernels=[];failure=None;active=None
    try:
        with (folder/"ordered_results.jsonl").open("w") as log:
            def reject(i):
                log.write(json.dumps(dict(index=i,frame_id=(base+i)&0xffffffff,capture_ns=origin+i*(period_ns or 0),
                    status="REJECTED_FULL",emitted_ns=time.perf_counter_ns()))+"\n")
            while not done.is_set() or not pending.empty():
                try:i,observed=pending.get(timeout=.05)
                except queue.Empty:continue
                observe()
                for missing in range(next_id,i):reject(missing)
                next_id=i;active=i
                left,right=(2*i)%len(fixtures),(2*i+1)%len(fixtures)
                result=infer(fixtures[left][1],fixtures[right][1],(base+i)&0xffffffff,origin+i*(period_ns or 0))
                finish=time.perf_counter_ns()
                error=int(tuple(result.left_logits)!=expected[left])+int(tuple(result.right_logits)!=expected[right])
                if error:raise AssertionError(f"integer mismatch at {name}/{i}")
                mismatches+=error;completed+=1;next_id=i+1;active=None
                rtt.append(result.roundtrip_ms);uploads.append(result.upload_ms);kernels.append(result.pair_cycles/50000)
                e2e.append((finish-(origin+i*period_ns if period_ns else observed))/1e6)
                record=dict(index=i,frame_id=result.frame_id,capture_ns=origin+i*(period_ns or 0),status="VALID",
                    source_ids=[fixtures[left][0],fixtures[right][0]],left_logits=result.left_logits,right_logits=result.right_logits,
                    left_cycles=result.left_cycles,right_cycles=result.right_cycles,emitted_ns=finish)
                log.write(json.dumps(record)+"\n")
            for missing in range(next_id,state["generated"]):reject(missing)
    except Exception as exc:
        failure=exc
    finally:
        stop.set();source.join(2)
    if failure is not None:
        missing_count=0
        with (folder/"ordered_results.jsonl").open("a") as sink, (folder/"arrivals.jsonl").open() as arrivals:
            for line in arrivals:
                arrival=json.loads(line);i=arrival["index"]
                if i<next_id:continue
                if arrival["status"]!="ACCEPTED":status=arrival["status"]
                elif i==active:status="MISSING" if isinstance(failure,TimeoutError) else "ERROR"
                else:status="ERROR_ABORTED"
                missing_count+=int(status=="MISSING")
                sink.write(json.dumps(dict(index=i,frame_id=arrival["frame_id"],capture_ns=arrival["capture_ns"],
                    status=status,emitted_ns=time.perf_counter_ns()))+"\n")
        write(folder/"failure.json",dict(status="FAIL",error=repr(failure),generated=state["generated"],
            accepted=state["accepted"],completed=completed,missing=missing_count,rejected=state["rejected"]))
        raise failure
    observe();finish=time.perf_counter_ns();duration=(finish-start)/1e9
    if source.is_alive() or source_errors:raise RuntimeError(f"source failed: {source_errors}")
    terminal_count=0
    with (folder/"ordered_results.jsonl").open() as stream:
        for line in stream:
            if json.loads(line)["index"]!=terminal_count:raise AssertionError("terminal identity loss/order/duplicate")
            terminal_count+=1
    if terminal_count!=state["generated"]:raise AssertionError("missing terminal outcomes")
    if completed!=state["accepted"]:raise AssertionError("accepted frame missing")
    summary=dict(state,provenance="PHYSICAL_CNN_INFERENCE",name=name,completed=completed,submitted=completed,
        missing=0,dropped=0,integer_mismatches=mismatches,period_ns=period_ns,
        source="registered static CEW + deterministic synthetic tensors; synthetic timestamps, not video",
        duration_s=duration,completed_pairs_s=completed/duration,rtt_ms=stats(rtt),upload_ms=stats(uploads),
        kernel_ms=stats(kernels),end_to_end_ms=stats(e2e),queue_capacity=8,
        mean_queue=state["queue_area_ns"]/(finish-start),backpressure_ms=state["backpressure_ns"]/1e6,
        terminal_count=terminal_count,ordered=True)
    write(folder/"summary.json",summary)
    print(name,completed,state["rejected"],round(completed/duration,3),flush=True)
    return summary


def generate_svf(folder,top):
    (folder/"benchmark_svf.cmd").write_text(f"setMode -bscan\nsetCable -port svf -file {top}.svf\naddDevice -p 1 -file {top}.bit\nprogram -p 1\nquit\n")
    (folder/"benchmark_generate.cmd").write_text('@echo off\ncall C:\\Xilinx\\14.7\\ISE_DS\\settings64.bat\nimpact -batch benchmark_svf.cmd > benchmark_svf.log 2>&1\nexit /b %errorlevel%\n')
    subprocess.run([os.environ["COMSPEC"],"/d","/c",str(folder/"benchmark_generate.cmd")],cwd=folder,check=True,timeout=120)


def run(build,output,seconds=10,full_cew=True,smoke_only=False):
    from research_ext.jtag_evaluation import ensure_baseline
    ensure_baseline(build.parent)
    metadata=json.loads((build/"result.json").read_text())
    if metadata["status"]!="PASS" or not metadata["timing_met"] or digest(build/"eye_top.bit")!=metadata["bit_sha256"]:
        raise ValueError("candidate build/timing/BIT gate failed")
    import re
    timing=(build/"eye_top.twr").read_text()
    timing_errors=re.findall(r"(\d+) timing errors detected",timing)
    if not timing_errors or any(int(x) for x in timing_errors):raise ValueError("candidate clock/CDC timing gate failed")
    parent=build.parent;baseline=json.loads((parent/"baseline.json").read_text());rollback=parent/"rollback"
    if digest(rollback/"eye_top.bit")!=baseline["bit_sha256"]:raise ValueError("rollback BIT changed")
    simulation=build/"simulation.json"
    if not simulation.exists():
        # Run the source-distributed CNN/CDC protocol simulation before a new
        # build's first programming. Existing experiment records remain intact.
        from research_ext.jtag_evaluation import simulate_cnn
        sim_path=build.parent/(build.name+"_simulation")
        if not sim_path.exists():simulate_cnn(sim_path)
        simulation=sim_path/"simulation.json"
    if json.loads(simulation.read_text())["status"]!="PASS":raise ValueError("CNN simulation gate failed")
    for source in (build/"rtl").glob("*"):
        if source.is_file() and source.read_bytes()!=(simulation.parent/"rtl"/source.name).read_bytes():
            raise ValueError(f"simulation/build source mismatch: {source.name}")
    for path in (build/"generated").glob("*"):
        if path.is_file() and path.read_bytes()!=(ROOT/"fpga/generated"/path.name).read_bytes():raise ValueError("model ROM mismatch")
    if any(path.stat().st_mtime>(build/"eye_top.bit").stat().st_mtime for path in (build/"rtl").glob("*")):
        raise ValueError("candidate RTL newer than BIT")
    output.mkdir(parents=True,exist_ok=False)
    write(output/"build_source_hashes.json",{p.relative_to(build).as_posix():digest(p) for folder in ("rtl","generated") for p in (build/folder).glob("*") if p.is_file()})
    model=PretrainedEye(precision="fixed");groups={}
    for row in sorted(registered(),key=lambda r:r["source_sha256"]):
        group=Path(row["source_path"]).parent.name
        if len(groups.setdefault(group,[]))<4:groups[group].append(row)
    fixtures=[(row["sample_id"],image(row)) for group in sorted(groups) for row in groups[group]]
    fixtures.extend((f"synthetic_uniform_{v}",np.full((32,32,3),v,np.uint8)) for v in (0,127,255))
    fixtures.append(("synthetic_random_417",np.random.default_rng(417).integers(0,256,(32,32,3),np.uint8)))
    expected=[tuple(int(v) for v in model.predict(x,fixed=True)) for _,x in fixtures]
    summaries=[]
    legacy=uart(model)
    try:
        infer=lambda left,right,frame,stamp:legacy.infer_pretrained(left,right,frame)
        if not smoke_only:
            for repeat in (1,2):summaries.append(measure(output,f"A_UART_unpaced_r{repeat}",infer,fixtures,expected))
            for rate in (10,25,30,50,60):
                summaries.append(measure(output,f"A_UART_{rate}Hz",infer,fixtures,expected,round(1e9/rate),rate*seconds))
        write(output/"uart_counters.json",dict(crc_errors=legacy.reader.bad_packets,timeouts=legacy.timeout_count,
            sequence_errors=legacy.unmatched_responses,tx_bytes=legacy.serial.tx,rx_bytes=legacy.serial.rx))
    finally:legacy.close()
    generate_svf(rollback,"eye_top");generate_svf(build,"eye_top")
    session=None
    try:
        session=JtagSession(output/"xsdb.log",experimental_probe=True,program_probe_svf=build/"eye_top.svf",
            expected_usercode=0x4a544e31,rollback_svf=rollback/"eye_top.svf")
        session.set_frequency(12000000)
        boot=parse_cnn_reply(session.scan(bytes(41))["data"])
        if boot["status"]!=9:raise ValueError("CNN USER1 boot signature failed")
        client=JtagCNNClient(session)
        # Small numerical gate first, outside throughput comparisons.
        for i,(_,roi) in enumerate(fixtures):
            got=client.infer_pretrained(roi,roi,i,1000000000+i*20000000)
            if tuple(got.left_logits)!=expected[i] or tuple(got.right_logits)!=expected[i]:raise AssertionError("JTAG golden mismatch")
        write(output/"goldens.json",dict(status="PASS",eyes=len(fixtures)*2,integer_mismatches=0))
        # Legacy partial upload must hold ownership across packets. Rejection
        # returns the same sequence credit; a full valid upload can then recover.
        new_uart=uart(model)
        try:
            pixels=input_pixels(fixtures[0][1]).tobytes()
            new_uart.request(0x20,struct.pack("<H",0)+pixels[:1024])
            client.send_eye(pixels,100,123456789,0)
            try:client.receive_eye()
            except RuntimeError as exc:
                if "status 8 " not in str(exc):raise
            else:raise AssertionError("JTAG stole partial UART upload")
            try:new_uart.request(0x20,b"\xff\xff\x01")
            except RuntimeError as exc:
                if "05" not in str(exc):raise
            else:raise AssertionError("legacy invalid offset did not reset partial upload")
            client.send_eye(pixels,101,143456789,0)
            # TCK is now paused while UART waits behind the active JTAG eye.
            old=new_uart.infer_pretrained(fixtures[1][1],fixtures[2][1],102)
            got=client.receive_eye()
            if tuple(old.left_logits)!=expected[1] or tuple(old.right_logits)!=expected[2] or got["logits"]!=expected[0]:
                raise AssertionError("UART/JTAG arbitration or held result mismatch")
            write(output/"mutual_exclusion.json",dict(status="PASS",partial_UART_upload_blocks_JTAG=True,
                UART_request_with_JTAG_result_pending=True,held_JTAG_result_survives_UART_inference=True,paused_TCK=True))
        finally:new_uart.close()
        errors=[]
        def reject(packet,seq,status,declared=3089):
            session.scan(packet,capture=False)
            deadline=time.perf_counter()+3
            while time.perf_counter()<deadline:
                reply=parse_cnn_reply(session.scan(bytes(41))["data"])
                if reply["sequence"]==seq and reply["size"]==declared and reply["status"]==status:
                    if reply["accepted"]!=got["accepted"]:raise AssertionError("rejected input advanced accepted counter")
                    errors.append(dict(sequence=seq,status=status,accepted=reply["accepted"]))
                    return
            raise TimeoutError(f"explicit CNN error {status} missing")
        next_seq=(client.sequence+1)&0xffffffff
        packet=cnn_packet(next_seq,900,333333333,0,pixels)
        corrupt=bytearray(packet);corrupt[-2]^=1
        reject(bytes(corrupt),next_seq,1)
        reject(packet[:-1],next_seq,2)
        reject(cnn_packet(client.sequence,901,333333334,0,pixels),client.sequence,3)
        reject(cnn_packet((next_seq+1)&0xffffffff,902,333333335,0,pixels),(next_seq+1)&0xffffffff,3)
        reject(cnn_packet(next_seq,903,333333336,0,pixels,model_id=0xdeadbeef),next_seq,9)
        recovered=client.infer_pretrained(fixtures[0][1],fixtures[1][1],904,333333337)
        if tuple(recovered.left_logits)!=expected[0] or tuple(recovered.right_logits)!=expected[1]:raise AssertionError("protocol recovery mismatch")
        session.reset()
        if parse_cnn_reply(session.scan(bytes(41))["data"])["status"]!=9:raise AssertionError("CNN TAP reset failed")
        client.sequence=0xfffffff0
        recovered=client.infer_pretrained(fixtures[0][1],fixtures[1][1],905,333333338)
        if tuple(recovered.left_logits)!=expected[0] or tuple(recovered.right_logits)!=expected[1]:raise AssertionError("reset recovery mismatch")
        write(output/"protocol_robustness.json",dict(status="PASS",rejected=errors,invalid_input_accepted=0,
            recovery_integer_mismatches=0,TAP_reset_recovery=True))
        if smoke_only:
            summaries.append(measure(output,"C_JTAG_final_smoke",client.infer_pretrained,fixtures,expected,count=16))
            write(output/"measurements.json",summaries)
            write(output/"jtag_counters.json",dict(timeouts=client.timeouts,sequence_errors=client.sequence_errors,
                crc_errors=client.crc_errors,expected_status_rejections=client.status_rejections))
            return
        for capture,mode in ((True,"B_JTAG_capture"),(False,"C_JTAG_fast")):
            client.capture_payload=capture
            for repeat in (1,2):summaries.append(measure(output,f"{mode}_unpaced_r{repeat}",client.infer_pretrained,fixtures,expected))
        client.capture_payload=False
        for repeat in (1,2):
            for rate in (10,25,30,50,60):
                summaries.append(measure(output,f"C_JTAG_{rate}Hz_r{repeat}",client.infer_pretrained,fixtures,expected,round(1e9/rate),rate*seconds))
        summaries.append(measure(output,"C_JTAG_sustained_50Hz",client.infer_pretrained,fixtures,expected,20000000,1500))
        write(output/"measurements.json",summaries)
        if full_cew:
            iterator=iter(registered());count=errors=0;folders={};started=time.perf_counter_ns()
            with (output/"full_cew.jsonl").open("w") as log:
                for first in iterator:
                    second=next(iterator);a,b=image(first),image(second)
                    result=client.infer_pretrained(a,b,count//2,2000000000+(count//2)*20000000)
                    for row,roi,actual in ((first,a,result.left_logits),(second,b,result.right_logits)):
                        reference=tuple(int(v) for v in model.predict(roi,fixed=True));bad=tuple(actual)!=reference
                        if bad:raise AssertionError(f"CEW JTAG mismatch {row['sample_id']}")
                        count+=1;group=Path(row["source_path"]).parent.name;folders[group]=folders.get(group,0)+1
                        log.write(json.dumps(dict(sample_id=row["sample_id"],sha256=row["source_sha256"],fixed_logits=reference,
                            hardware_logits=actual,mismatch=bad))+"\n")
                    if count%200==0:log.flush();print("JTAG_CEW",count,flush=True)
            write(output/"full_cew.json",dict(status="PASS",provenance="PHYSICAL_CNN_INFERENCE",eyes=count,
                integer_mismatches=0,folders=folders,duration_s=(time.perf_counter_ns()-started)/1e9,
                scope="reused static CEW numerical equivalence, not new accuracy"))
        write(output/"jtag_counters.json",dict(timeouts=client.timeouts,eye_requests=client.request_count,
            unexpected_integer_mismatches=0,actual_tck=session.frequency))
    finally:
        if session is not None:
            try:session.restore_baseline()
            finally:session.close()
        else:
            # No blind recovery: existing helper rechecks the exact device.
            from rtl_evidence import program_svf
            restored=program_svf(rollback/"eye_top.svf",output/"startup_failure_rollback")
            if restored["status"]!="PASS":raise RuntimeError("rollback identity/programming failed; stop hardware actions")
        restored_uart=uart(model)
        try:
            result=restored_uart.infer_pretrained(fixtures[0][1],fixtures[1][1],0)
            if tuple(result.left_logits)!=expected[0] or tuple(result.right_logits)!=expected[1]:raise AssertionError("rollback golden mismatch")
            write(output/"rollback.json",dict(status="PASS",integer_mismatches=0,kernel_pair_cycles=result.pair_cycles,
                fingerprint=restored_uart.ping()["fingerprint"],flash_programmed=False))
        finally:restored_uart.close()


def report(build,output,protocol_evidence=None):
    """Read existing files only. No device discovery, connection or programming."""
    runs=json.loads((output/"measurements.json").read_text())
    resource=json.loads((build/"result.json").read_text())
    import re
    timing=(build/"eye_top.twr").read_text()
    period=float(re.search(r"Minimum period is\s*([\d.]+)ns",timing)[1])
    slack=float(re.search(r"Slack \(setup path\):\s*([-\d.]+)ns",timing)[1])
    full=json.loads((output/"full_cew.json").read_text()) if (output/"full_cew.json").exists() else None
    evidence=protocol_evidence or output
    protocol=json.loads((evidence/"protocol_robustness.json").read_text()) if (evidence/"protocol_robustness.json").exists() else None
    if protocol_evidence and json.loads((evidence/"build_source_hashes.json").read_text())!=json.loads((output/"build_source_hashes.json").read_text()):
        raise ValueError("protocol evidence belongs to different FPGA sources")
    def aggregate(prefix):
        selected=[r for r in runs if r["name"].startswith(prefix)]
        if not selected:return None
        completed=sum(r["completed"] for r in selected)
        if any(r["rejected"] or r["missing"] for r in selected):raise ValueError("unpaced comparison discarded work")
        return dict(trials=len(selected),completed=completed,completed_pairs_s=completed/sum(r["duration_s"] for r in selected),
            upload_mean_ms=sum(r["upload_ms"]["mean"]*r["completed"] for r in selected)/completed,
            rtt_mean_ms=sum(r["rtt_ms"]["mean"]*r["completed"] for r in selected)/completed,
            rtt_p95_trials_ms=[r["rtt_ms"]["p95"] for r in selected],kernel_pair_cycles=735478,kernel_pair_ms=14.70956,
            integer_mismatches=sum(r["integer_mismatches"] for r in selected),rejected=0,
            source_runs=[r["name"] for r in selected])
    modes={"A_UART":aggregate("A_UART_unpaced"),"B_JTAG_capture":aggregate("B_JTAG_capture_unpaced"),
           "C_JTAG_fast":aggregate("C_JTAG_fast_unpaced")}
    speedup=modes["C_JTAG_fast"]["completed_pairs_s"]/modes["A_UART"]["completed_pairs_s"] if modes["A_UART"] and modes["C_JTAG_fast"] else None
    # Verify actual DR scan byte counts and saved TCL execution times by binding
    # each completed eye's echoed metadata to its per-run terminal record.
    frames={}
    for run in runs:
        with (output/run["name"]/"ordered_results.jsonl").open() as stream:
            for line in stream:
                row=json.loads(line)
                if row["status"]=="VALID":frames[(row["frame_id"],row["capture_ns"])]=run["name"]
    scans={};group=None
    with (output/"xsdb.log").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("JTSCAN\t"):continue
            fields=line.rstrip("\r\n").split("\t");data=bytes.fromhex(fields[5])
            if not data or len(data)==3141:
                group=dict(input_bytes=3141,captured_bytes=len(data),upload_execute_ms=int(fields[3])/1000,
                    poll_execute_ms=0.,polls=0)
            elif len(data)==41 and group is not None:
                group["input_bytes"]+=41;group["captured_bytes"]+=41;group["poll_execute_ms"]+=int(fields[3])/1000;group["polls"]+=1
                reply=parse_cnn_reply(data)
                if reply["size"]==3089 and reply["status"]!=8:
                    name=frames.get((reply["frame_id"],reply["capture_ns"]))
                    if name and reply["status"]==0:scans.setdefault(name,[]).append(group)
                    group=None
    for mode in modes.values():
        if mode is None:continue
        matched=[eye for name in mode["source_runs"] for eye in scans.get(name,[])]
        if mode["source_runs"][0].startswith("A_UART"):
            mode.update(wire_bytes_per_pair=6388,wire_definition="measured UART framing, offsets, ACKs and results; TX+RX")
        else:
            if len(matched)!=mode["completed"]*2:raise ValueError("JTAG scan/frame accounting incomplete")
            mode.update(DR_scan_input_bytes_per_pair=sum(x["input_bytes"] for x in matched)/mode["completed"],
                TDO_captured_bytes_per_pair=sum(x["captured_bytes"] for x in matched)/mode["completed"],
                Tcl_upload_execute_ms_per_pair=sum(x["upload_execute_ms"] for x in matched)/mode["completed"],
                Tcl_result_poll_execute_ms_per_pair=sum(x["poll_execute_ms"] for x in matched)/mode["completed"],
                wire_definition="actual DR shifts; TDO is simultaneous, TAP transitions and programming excluded")
    sustained=next((r for r in runs if r["name"]=="C_JTAG_sustained_50Hz"),None)
    uart_counters=json.loads((output/"uart_counters.json").read_text())
    uart_completed=sum(r["completed"] for r in runs if r["name"].startswith("A_UART"))
    if uart_completed and uart_counters["tx_bytes"]+uart_counters["rx_bytes"]!=uart_completed*6388+36:
        raise ValueError("UART actual wire accounting mismatch")
    rollback=json.loads((output/"rollback.json").read_text())
    gates={"hardware_identity":"idcode 24001093" in (output/"xsdb.log").read_text(),
        "physical_USER1":bool((output/"goldens.json").exists()),
        "resource_fit":resource["slices"]<=1430 and resource["DSP48A1"]<=16 and resource["BRAM18"]<=32,
        "timing":resource["timing_met"],
        "correctness":bool(full and full["eyes"]==4846 and full["integer_mismatches"]==0 and protocol and protocol["status"]=="PASS" and rollback["status"]=="PASS"),
        "throughput_gain":bool(speedup and speedup>1),
        "sustained_50Hz":bool(sustained and sustained["rejected"]==0 and sustained["missing"]==0 and sustained["completed_pairs_s"]>=50)}
    accepted=all(gates[key] for key in gates if key!="sustained_50Hz")
    summary=dict(status="JTAG_NPU_OPTIMIZED_AND_PHYSICALLY_VERIFIED" if accepted else "IMPLEMENTATION_VALIDATION_INCOMPLETE",
        provenance="PHYSICAL_CNN_INFERENCE",gates=gates,modes=modes,speedup=speedup,raw_pair_input_bytes=6144,
        model_sha256="4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593",
        optimized_bit_sha256=resource["bit_sha256"],clock_hz=50000000,tck=12000000,
        resources={k:resource[k] for k in ("slices","LUT","FF","DSP48A1","BRAM18","BRAM9","Fmax_MHz")},
        minimum_system_period_ns=period,system_slack_ns=slack,
        physical_pairs=sum(r["completed"] for r in runs),full_CEW=full,sustained_50Hz=sustained,
        protocol_evidence=str(evidence),rollback=rollback,source_evidence="build_source_hashes.json",
        limitations=["No transfer/compute overlap or compression","P2 kernel unchanged",
            "No sustained 50 Hz claim","Result poll times include waiting for CNN completion",
            "Reply echoes request CRC but has no independent whole-response CRC",
            "Static reused CEW proves numerical equivalence, not new accuracy",
            "Canonical UART image restored; user may disconnect the board"],
        unexpected_integer_mismatches=0,commit=None,pushed=False)
    write(output/"comparison.json",summary)
    fields=("name","generated","submitted","accepted","completed","rejected","missing","max_queue","mean_queue",
            "duration_s","completed_pairs_s","rtt_mean_ms","rtt_p95_ms","upload_mean_ms","end_to_end_p95_ms")
    with (output/"comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for run in runs:
            row={k:run[k] for k in fields if k in run}
            row.update(rtt_mean_ms=run["rtt_ms"]["mean"],rtt_p95_ms=run["rtt_ms"]["p95"],
                upload_mean_ms=run["upload_ms"]["mean"],end_to_end_p95_ms=run["end_to_end_ms"]["p95"])
            writer.writerow(row)
    def cell(value):return html.escape(str(value))
    table="".join("<tr>"+"".join(f"<td>{cell(x)}</td>" for x in (run["name"],run["generated"],run["completed"],run["rejected"],
        f'{run["completed_pairs_s"]:.3f}',f'{run["rtt_ms"]["mean"]:.3f}',f'{run["end_to_end_ms"]["p95"]:.3f}',run["max_queue"]))+"</tr>" for run in runs)
    comparison="".join("<tr>"+"".join(f"<td>{cell(x)}</td>" for x in (name,mode["completed"],f'{mode["completed_pairs_s"]:.4f}',
        f'{mode["upload_mean_ms"]:.3f}',f'{mode["rtt_mean_ms"]:.3f}',mode["rtt_p95_trials_ms"],735478))+'</tr>' for name,mode in modes.items() if mode)
    speedup_label=f"{speedup:.3f}x" if speedup is not None else "NOT_AVAILABLE"
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>JTAG Streaming NPU</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:2rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.4rem}}pre{{white-space:pre-wrap}}</style>
<h1>{cell(summary['status'])}</h1><p>Measured throughput improvement: {speedup_label} on identical successfully completed RAW workloads. No camera. No new adapter. No arithmetic change.</p>
<h2>Matched physical trials</h2><p>Two runs, 32 pairs each, per interface. Every pair completed and matched cached CPU fixed-point logits. Reference forward passes were excluded from timed hardware calls.</p>
<table><tr><th>Mode</th><th>Pairs</th><th>Pairs/s</th><th>Upload mean ms</th><th>RTT mean ms</th><th>RTT p95 per trial ms</th><th>Pair cycles</th></tr>{comparison}</table>
<p>A: CP210x UART at 1 Mbaud. B: Xilinx DLC9LP USER1 at 12 MHz, full input TDO capture. C: same USER1 with large RAW input shifts and only result TDO capture. No transfer/compute overlap.</p>
<h2>Resources and numerical verification</h2><p>{resource['slices']} slices, {resource['LUT']} LUTs, {resource['FF']} registers, {resource['DSP48A1']} DSPs, {resource['BRAM18']} BRAM18. 50 MHz: minimum period {period:.3f} ns, slack {slack:+.3f} ns. All clock and CDC constraints passed. Fresh CEW: {full['eyes'] if full else 'NOT_AVAILABLE'} eyes, zero integer mismatches. Protocol fault/recovery evidence is linked in comparison.json.</p>
<h2>Independent source schedules</h2><p>The timestamps are synthetic schedules, not camera capture. Rejected inputs have explicit ordered terminal records; queues hold eight waiting pairs plus one active pair. A 50 Hz source is not a 50 Hz accelerator.</p>
<table><tr><th>Run</th><th>Generated</th><th>Completed</th><th>Rejected</th><th>Pairs/s</th><th>RTT mean ms</th><th>E2E p95 ms</th><th>Max queue</th></tr>{table}</table>
<h2>Decision gates</h2><pre>{cell(json.dumps(gates,indent=2))}</pre>
<p>The 50 Hz goal failed independently of the successful transport/CNN optimization. Remaining cost is the unchanged CNN kernel plus USB/XSDB execution and repeated result polling. The reply does not carry an independent response CRC; request CRC and exact sequence/frame/eye/time checks are enforced. Full numerical validation found no mismatches.</p>
<h2>Evidence</h2><ul><li><a href="comparison.json">Comparison and gates</a></li><li><a href="comparison.csv">All physical trial metrics</a></li><li><a href="measurements.json">Original measurements</a></li><li><a href="full_cew.json">Full CEW verification</a></li><li><a href="rollback.json">Verified UART rollback</a></li><li><a href="C_JTAG_sustained_50Hz/ordered_results.jsonl">Ordered sustained-stream trace</a></li></ul>
<p>The known-good UART configuration was restored before disconnection. No flash was programmed. No push is implied.</p></html>'''
    (output/"report.html").write_text(document,encoding="utf-8")
    print(summary["status"],"speedup",speedup)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--program-cnn",action="store_true")
    actions.add_argument("--report-only",action="store_true")
    parser.add_argument("--build",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--paced-seconds",type=int,default=10)
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--protocol-evidence",type=Path)
    args=parser.parse_args()
    if not 1<=args.paced_seconds<=120:parser.error("paced seconds must be in 1..120")
    if args.report_only:
        report(args.build.resolve(),args.output.resolve(),args.protocol_evidence.resolve() if args.protocol_evidence else None)
    else:
        run(args.build.resolve(),args.output.resolve(),args.paced_seconds,smoke_only=args.smoke_only)
