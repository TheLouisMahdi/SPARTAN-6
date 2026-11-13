"""Reproducible isolated JTG1 probe simulation/build; no implicit programming."""
import argparse
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from jtag_host import probe_packet

SCAN_CONSTRAINTS='''NET "tck" TNM_NET="scan_clock";
TIMESPEC "TS_scan_clock"=PERIOD "scan_clock" 83.333 ns HIGH 50%;
INST "stream/fifo/wmeta*" TNM="wptr_first";
INST "stream/fifo/rmeta*" TNM="rptr_first";
INST "stream/ack_meta" TNM="ack_first";
INST "stream/overflow_meta" TNM="overflow_first";
TIMESPEC "TS_wptr_cdc"=FROM "scan_clock" TO "wptr_first" 20 ns DATAPATHONLY;
TIMESPEC "TS_rptr_cdc"=FROM "board_clock" TO "rptr_first" 20 ns DATAPATHONLY;
TIMESPEC "TS_ack_cdc"=FROM "board_clock" TO "ack_first" TIG;
TIMESPEC "TS_overflow_cdc"=FROM "scan_clock" TO "overflow_first" TIG;
INST "stream/tx_shift*" TNM="capture_reply";
TIMESPEC "TS_stable_reply"=FROM "board_clock" TO "capture_reply" 60 ns DATAPATHONLY;
'''


def ensure_baseline(parent):
    """Prepare a verified on-disk rollback before any explicit programming."""
    record=parent/"baseline.json"
    if record.exists():
        metadata=json.loads(record.read_text())
        bit=parent/"rollback/eye_top.bit"
        if not bit.is_file() or hashlib.sha256(bit.read_bytes()).hexdigest()!=metadata["bit_sha256"]:
            raise ValueError("prepared rollback BIT changed")
        return
    from rtl_evidence import _check_pretrained_weights
    _check_pretrained_weights()
    artifact=json.loads((ROOT/"fpga/result.json").read_text())
    if artifact["status"]!="PASS" or not artifact["timing_met"] or artifact["pe_count"]!=2 or artifact["network"]!="pretrained":
        raise ValueError("verified canonical P2 rollback build is required")
    bit=ROOT/"fpga/eye_top.bit"
    if hashlib.sha256(bit.read_bytes()).hexdigest()!=artifact["bit_sha256"]:raise ValueError("canonical BIT changed")
    protected={}
    for name,value in artifact["source_sha256"].items():
        if hashlib.sha256((ROOT/"fpga"/name).read_bytes()).hexdigest()!=value:
            raise ValueError(f"canonical source changed since known-good build: {name}")
        protected["fpga/"+name]=value
    rollback=parent/"rollback";rollback.mkdir(parents=True,exist_ok=False)
    shutil.copy2(bit,rollback/"eye_top.bit");shutil.copy2(ROOT/"fpga/result.json",rollback/"result.json")
    (parent/"source_before.json").write_text(json.dumps(protected,indent=2))
    (parent/"status_before.txt").write_bytes(subprocess.check_output(["git","status","--short"],cwd=ROOT))
    record.write_text(json.dumps(dict(bit_sha256=artifact["bit_sha256"],clock_hz=50000000,baud=1000000,
        model_sha256=hashlib.sha256((ROOT/"assets/models/open-closed-eye-0001/model.onnx").read_bytes()).hexdigest()),indent=2))


def prepare_cnn(output):
    """Make an isolated variant. Never edit canonical RTL, ROMs, pins or BIT."""
    from rtl_evidence import _check_pretrained_weights
    _check_pretrained_weights()
    output.mkdir(parents=True,exist_ok=False)
    for folder in ("rtl","generated"):shutil.copytree(ROOT/"fpga"/folder,output/folder)
    for source in (ROOT/"fpga/experimental").glob("*.v"):shutil.copy2(source,output/"rtl"/source.name)
    top=output/"rtl/eye_top.v";text=top.read_text()
    for name,value in (("CLOCK_HZ",50000000),("BAUD",1000000),("PE_COUNT",2),("PRETRAINED",1)):
        if not re.search(r"parameter integer "+name+"="+str(value)+r"\b",text):
            raise ValueError(f"canonical P2 source requires {name}={value}")
    if text.count("endmodule")!=1:raise ValueError("canonical top structure changed")
    block='''    wire uart_input_we,uart_core_start,jtag_claim,jtag_input_we,jtag_core_start,uart_upload_active;
    wire [11:0] uart_input_address,jtag_input_address;
    wire [7:0] uart_input_data,jtag_input_data;
    reg jtag_lock=0;
    always @(posedge clk)begin
        if(reset)jtag_lock<=0;
        else if(jtag_claim)jtag_lock<=1;
        else if(!core_busy)jtag_lock<=0;
    end
    wire jtag_available=!(packet_valid||tx_packet_busy||core_busy||uart_input_we||uart_core_start||jtag_lock||uart_upload_active);
    assign input_we=!core_busy&&((jtag_claim&&jtag_input_we)||(!jtag_lock&&!jtag_claim&&uart_input_we));
    assign input_address=jtag_claim ? jtag_input_address : uart_input_address;
    assign input_data=jtag_claim ? jtag_input_data : uart_input_data;
    assign core_start=!core_busy&&((jtag_claim&&jtag_core_start)||(!jtag_lock&&!jtag_claim&&uart_core_start));
    jtag_cnn_port jtag_port(clk,jtag_available,core_busy,core_done,core_open,core_closed,core_cycles,
        jtag_claim,jtag_input_we,jtag_input_address,jtag_input_data,jtag_core_start);
'''
    replacements={'.packet_valid(packet_valid),':'.packet_valid(packet_valid&&!jtag_lock&&!jtag_claim),',
        '.input_we(input_we), .input_address(input_address),':'.input_we(uart_input_we), .input_address(uart_input_address),',
        '.input_data(input_data), .core_start(core_start),':'.input_data(uart_input_data), .core_start(uart_core_start),',
        '.last_duration_ms(last_duration_ms));':'.last_duration_ms(last_duration_ms), .pretrained_upload_active(uart_upload_active));'}
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError(f"canonical port structure changed: {old}")
        text=text.replace(old,new)
    position=text.index('    uart_rx #')
    text=text[:position]+block+'\n\n'+text[position:]
    text=text.replace('endmodule','\nendmodule')
    top.write_text(text)
    controller=output/"rtl/command_controller.v";text=controller.read_text()
    for old,new in {'input wire [31:0] last_duration_ms);':'input wire [31:0] last_duration_ms, output wire pretrained_upload_active);',
                    'reg [12:0] uploaded=0;':'reg [12:0] uploaded=0;\n    assign pretrained_upload_active=uploaded!=0;'}.items():
        if text.count(old)!=1:raise ValueError("controller structure changed")
        text=text.replace(old,new)
    controller.write_text(text)
    constraints=(ROOT/"fpga/board_pins.ucf").read_text()
    # Preserve the exact routed variant's constraint layout, including its
    # harmless blank separator lines, for source-hash reproducibility.
    constraints+='\n'+SCAN_CONSTRAINTS.rstrip('\n').replace('"tck"','"jtag_port/tck"').replace('"stream/','"jtag_port/')
    (output/"board_pins.ucf").write_text(constraints)
    sources="jtag_cnn_port.v "+" ".join(p.name for p in sorted((output/"rtl").glob("*.v")) if p.name!="jtag_cnn_port.v")
    (output/"build.tcl").write_text('''project new eye_top.xise
project set family Spartan6
project set device xc6slx9
project set package ftg256
project set speed -2
project set synthesis_tool {XST (VHDL/Verilog)}
'''+f'foreach name {{{sources}}} {{xfile add rtl/$name}}\n'+'''xfile add board_pins.ucf
project set top eye_top
project set {Unused IOB Pins} Float -process {Generate Programming File}
project set {FPGA Start-Up Clock} {JTAG Clock} -process {Generate Programming File}
project set {UserID Code (8 Digit Hexadecimal)} 4A544E31 -process {Generate Programming File}
foreach step {{Synthesize - XST} {Implement Design} {Generate Programming File}} {
 if {![process run $step]} {project close;error "Failed $step"}
}
project close
''')


def build_cnn(output):
    from rtl_evidence import _build_project
    prepare_cnn(output)
    result=_build_project(output,2,Path(r"C:\Xilinx\14.7\ISE_DS"),"pretrained",1000000)
    timing=(output/"eye_top.twr").read_text() if (output/"eye_top.twr").exists() else ""
    errors=re.findall(r"(\d+) timing errors detected",timing)
    result["timing_met"]=bool(errors and all(int(x)==0 for x in errors))
    if not result["timing_met"] and result["status"]=="PASS":result["status"]="FAIL_TIMING"
    result.update(scope="isolated JTAG CNN RAW variant, canonical default unchanged",usercode="4A544E31",
                  source_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ("rtl","generated") for p in (output/folder).glob("*") if p.is_file()})
    (output/"result.json").write_text(json.dumps(result,indent=2))


def simulate_cnn(output):
    from pretrained_eye import PretrainedEye,input_pixels
    import numpy as np
    prepare_cnn(output)
    model=PretrainedEye(precision="fixed");neutral=np.full((32,32,3),127,np.uint8)
    random=np.random.default_rng(417).integers(0,256,(32,32,3),np.uint8)
    cases=[(0xfffffffe,0,neutral),(0xffffffff,0,random),(0xffffffff,3,neutral),
           (0,1,neutral),(0,8,neutral),(0,9,neutral),(0,0,neutral)]
    vector=bytearray()
    from jtag_host import cnn_packet
    for i,(sequence,status,image) in enumerate(cases):
        frame,eye,capture=77+i,i%2,123456789+i*20000000
        packet=bytearray(cnn_packet(sequence,frame,capture,eye,input_pixels(image).tobytes(),
                                    model_id=0xdeadbeef if status==9 else 0x4daa1000))
        if status==1:packet[-2]^=1
        canonical=tuple(int(v) for v in model.predict(image,fixed=True))
        vector.extend(struct.pack("<HBIIBQii",len(packet),status,sequence,frame,eye,capture,canonical[1],canonical[0])+packet)
    (output/"cnn_vectors.hex").write_text("\n".join(f"{b:02x}" for b in vector)+"\n")
    binary=ROOT/"assets/tools/iverilog/ucrt64/bin";env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ["PATH"])
    with (output/"cnn_simulation.log").open("w") as log:
        for command in ([str(binary/"iverilog.exe"),"-g2001",f"-DVECTOR_BYTES={len(vector)}","-s","tb_cnn_candidate","-o","cnn_tb.vvp",
            "rtl/jtag_cnn_port.v","rtl/jtag_async_fifo.v","rtl/omz_cnn.v",str(ROOT/"research_ext/tb_jtag_cnn.v")],
            [str(binary/"vvp.exe"),"cnn_tb.vvp"]):
            subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    text=(output/"cnn_simulation.log").read_text();passed="CNN_JTAG_SIM_PASS" in text and "FAIL" not in text
    if not passed:raise AssertionError(text[-1000:])
    from fpga_host import encode_packet
    uart=encode_packet(0x20,1,b"\x00\x00\x37")+encode_packet(0x20,2,b"\xff\xff\x01")
    (output/"uart_ownership.hex").write_text("\n".join(f"{b:02x}" for b in uart)+"\n")
    with (output/"ownership.log").open("w") as log:
        for command in ([str(binary/"iverilog.exe"),"-g2001","-s","tb_jtag_ownership","-o","ownership.vvp",
            *[p.relative_to(output).as_posix() for p in (output/"rtl").glob("*.v")],str(ROOT/"research_ext/tb_jtag_ownership.v")],
            [str(binary/"vvp.exe"),"ownership.vvp"]):
            subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    text=(output/"ownership.log").read_text()
    if "JTAG_OWNERSHIP_PASS" not in text or "FAIL" in text:raise AssertionError(text[-1000:])
    (output/"simulation.json").write_text(json.dumps(dict(status="PASS",provenance="RTL_SIMULATION",eyes=3,full_top_UART_ownership="PASS")))


def simulate(output):
    output.mkdir(parents=True,exist_ok=False)
    data=bytes((i*73+19)%256 for i in range(6144))
    cases=[]
    for seq,n in zip((0xfffffffe,0xffffffff,0,1,2),(32,256,1024,3072,6144)):
        cases.append((probe_packet(seq,data[:n]),0,seq))
    cases.extend([(probe_packet(2,data[:32]),3,2),(probe_packet(4,data[:32]),3,4)])
    corrupt=bytearray(probe_packet(3,data[:32]));corrupt[-2]^=1
    cases.append((bytes(corrupt),1,3))
    cases.append((probe_packet(3,data[:32])[:-1],2,3))
    cases.append((probe_packet(3,data[:32]),5,3)) # TB adds three partial bits
    cases.extend([(probe_packet(3,data[:32]),0,3),(probe_packet(9,b"",3),0,9),
                  (probe_packet(10,b""),0,10)])
    def malformed(version,command,seq,size):
        body=struct.pack("<BBIH",version,command,seq,size)
        return bytes(16)+body+struct.pack("<H",binascii.crc_hqx(body,0xffff))+b"\0"
    cases.extend([(malformed(1,1,11,7000),2,11),(malformed(2,1,11,0),6,11)])
    vector=b"".join(struct.pack("<HBI",len(packet),status,seq)+packet for packet,status,seq in cases)
    (output/"probe_vectors.hex").write_text("\n".join(f"{b:02x}" for b in vector)+"\n")
    shutil.copytree(ROOT/"fpga/experimental",output/"experimental")
    (output/"rtl").mkdir()
    shutil.copy2(ROOT/"fpga/rtl/crc16_function.vh",output/"rtl")
    for name in ("tb_jtag_probe.v","tb_jtag_fifo.v"):
        shutil.copy2(ROOT/"research_ext"/name,output/name)
    binary=ROOT/"assets/tools/iverilog/ucrt64/bin"
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ["PATH"])
    logs=[]
    for half in (7,20,42,167,333):
        commands=[[str(binary/"iverilog.exe"),"-g2001",f"-DTCK_HALF={half}",f"-DVECTOR_BYTES={len(vector)}","-s","tb_jtag_probe","-o",f"probe{half}.vvp",
            "experimental/jtag_async_fifo.v","experimental/jtag_probe.v","tb_jtag_probe.v"],
            [str(binary/"vvp.exe"),f"probe{half}.vvp"]]
        with (output/f"probe{half}.log").open("w") as log:
            for cmd in commands:subprocess.run(cmd,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        text=(output/f"probe{half}.log").read_text()
        if "JTAG_PROBE_PASS" not in text or "FAIL" in text:raise AssertionError(text[-1000:])
        logs.append({"half_ns":half,"status":"PASS"})
    with (output/"fifo.log").open("w") as log:
        for cmd in ([str(binary/"iverilog.exe"),"-g2001","-s","tb_jtag_fifo","-o","fifo.vvp","experimental/jtag_async_fifo.v","tb_jtag_fifo.v"],
                    [str(binary/"vvp.exe"),"fifo.vvp"]):
            subprocess.run(cmd,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    if "JTAG_FIFO_PASS" not in (output/"fifo.log").read_text():raise AssertionError("FIFO failed")
    (output/"result.json").write_text(json.dumps({"status":"PASS","provenance":"RTL_SIMULATION","probe":logs,"FIFO":"PASS"},indent=2))


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/"fpga/experimental",output/"experimental")
    (output/"rtl").mkdir();shutil.copy2(ROOT/"fpga/rtl/crc16_function.vh",output/"rtl")
    constraints='''NET "clk" LOC="T8" | IOSTANDARD=LVCMOS33;
NET "clk" TNM_NET="board_clock";
TIMESPEC "TS_board_clock"=PERIOD "board_clock" 20 ns HIGH 50%;
NET "tck" TNM_NET="scan_clock";
TIMESPEC "TS_scan_clock"=PERIOD "scan_clock" 83.333 ns HIGH 50%;
NET "led<0>" LOC="P4" | IOSTANDARD=LVCMOS33;
NET "led<1>" LOC="N5" | IOSTANDARD=LVCMOS33;
NET "led<2>" LOC="P5" | IOSTANDARD=LVCMOS33;
NET "led<3>" LOC="M6" | IOSTANDARD=LVCMOS33;
INST "stream/fifo/wmeta*" TNM="wptr_first";
INST "stream/fifo/rmeta*" TNM="rptr_first";
INST "stream/ack_meta" TNM="ack_first";
INST "stream/overflow_meta" TNM="overflow_first";
TIMESPEC "TS_wptr_cdc"=FROM "scan_clock" TO "wptr_first" 20 ns DATAPATHONLY;
TIMESPEC "TS_rptr_cdc"=FROM "board_clock" TO "rptr_first" 20 ns DATAPATHONLY;
TIMESPEC "TS_ack_cdc"=FROM "board_clock" TO "ack_first" TIG;
TIMESPEC "TS_overflow_cdc"=FROM "scan_clock" TO "overflow_first" TIG;
INST "stream/tx_shift*" TNM="capture_reply";
TIMESPEC "TS_stable_reply"=FROM "board_clock" TO "capture_reply" 60 ns DATAPATHONLY;
'''
    (output/"probe.ucf").write_text(constraints)
    tcl='''project new jtag_probe.xise
project set family Spartan6
project set device xc6slx9
project set package ftg256
project set speed -2
project set synthesis_tool {XST (VHDL/Verilog)}
xfile add experimental/jtag_probe.v
xfile add experimental/jtag_async_fifo.v
xfile add probe.ucf
project set top jtag_probe
project set {Unused IOB Pins} Float -process {Generate Programming File}
project set {FPGA Start-Up Clock} {JTAG Clock} -process {Generate Programming File}
project set {UserID Code (8 Digit Hexadecimal)} 4A544731 -process {Generate Programming File}
foreach step {{Synthesize - XST} {Implement Design} {Generate Programming File}} {
    if {![process run $step]} {project close;error "Failed $step"}
}
project close
'''
    (output/"build.tcl").write_text(tcl)
    (output/"build.cmd").write_text('@echo off\ncall C:\\Xilinx\\14.7\\ISE_DS\\settings64.bat\nxtclsh build.tcl > build.log 2>&1\nexit /b %errorlevel%\n')
    process=subprocess.run([os.environ["COMSPEC"],"/d","/c",str(output/"build.cmd")],cwd=output,timeout=1800)
    report=next(output.glob("*_map.mrp"),output/"jtag_probe.mrp")
    mapped=report.read_text(errors="replace") if report.exists() else ""
    counts={}
    for key,label in (("slices","occupied Slices"),("LUTs","Slice LUTs"),("FF","Slice Registers"),
                      ("DSP","DSP48A1s"),("BRAM18","RAMB16BWERs"),("BRAM9","RAMB8BWERs")):
        match=re.search(r"Number of "+label+r":\s*([\d,]+)",mapped)
        counts[key]=int(match[1].replace(",","")) if match else None
    timing=output/"jtag_probe.twr"
    text=timing.read_text(errors="replace") if timing.exists() else ""
    match=re.search(r"(\d+) timing errors detected",text)
    result={"provenance":"ISE_IMPLEMENTATION","status":"PASS" if process.returncode==0 and match and int(match[1])==0 else "FAIL",
            "resources":counts,"timing_met":bool(match and int(match[1])==0),"returncode":process.returncode,
            "bit_sha256":hashlib.sha256((output/"jtag_probe.bit").read_bytes()).hexdigest() if (output/"jtag_probe.bit").exists() else None}
    result["source_sha256"]={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*output.glob("experimental/*.v"),output/"rtl/crc16_function.vh",output/"probe.ucf"]}
    (output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


def program_probe(build_path):
    """Explicit volatile programming experiment, always restoring baseline in finally."""
    from jtag_host import JtagSession,parse_reply
    from rtl_evidence import program_svf
    ensure_baseline(build_path.parent)
    build_result=json.loads((build_path/"result.json").read_text())
    if build_result["status"]!="PASS" or not build_result["timing_met"]:
        raise ValueError("A routed, timing-passing probe is required")
    if hashlib.sha256((build_path/"jtag_probe.bit").read_bytes()).hexdigest()!=build_result["bit_sha256"]:
        raise ValueError("probe BIT changed")
    for name in ("jtag_probe.v","jtag_async_fifo.v"):
        if (build_path/"experimental"/name).read_bytes()!=(ROOT/"fpga/experimental"/name).read_bytes():
            raise ValueError("probe source changed after build")
    if (build_path/"rtl/crc16_function.vh").read_bytes()!=(ROOT/"fpga/rtl/crc16_function.vh").read_bytes():
        raise ValueError("CRC source changed")
    timing=(build_path/"jtag_probe.twr").read_text()
    if "TS_board_clock" not in timing or "TS_scan_clock" not in timing:
        raise ValueError("missing clock constraints")
    parent=build_path.parent
    rollback=parent/"rollback"
    baseline=json.loads((parent/"baseline.json").read_text())
    if hashlib.sha256((rollback/"eye_top.bit").read_bytes()).hexdigest()!=baseline["bit_sha256"]:
        raise ValueError("known-good rollback BIT unavailable")
    original=json.loads((parent/"source_before.json").read_text())
    for name,digest in original.items():
        if name.startswith(("fpga/rtl/","fpga/generated/")) and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("canonical FPGA sources changed")
    def svf(folder,top):
        (folder/"generate.cmd").write_text('setMode -bscan\nsetCable -port svf -file '+top+'.svf\naddDevice -p 1 -file '+top+'.bit\nprogram -p 1\nquit\n')
        (folder/"generate_svf.cmd").write_text('@echo off\ncall C:\\Xilinx\\14.7\\ISE_DS\\settings64.bat\nimpact -batch generate.cmd > generate_svf.log 2>&1\nexit /b %errorlevel%\n')
        subprocess.run([os.environ["COMSPEC"],"/d","/c",str(folder/"generate_svf.cmd")],cwd=folder,check=True,timeout=120)
    svf(rollback,"eye_top") # Prepare rollback before touching the physical configuration.
    svf(build_path,"jtag_probe")
    output=parent/("physical_probe_"+uuid.uuid4().hex[:8]);output.mkdir(exist_ok=False)
    print("PHYSICAL_OUTPUT",output,flush=True)
    records=[]
    result={"status":"RUNNING","provenance":"PHYSICAL_JTAG_TRANSFER","CNN_integration":False}
    changed=False
    try:
        changed=True # A failed programming return still requires rollback.
        # One XSDB owner retains the positively identified target across the
        # explicit programming stage and subsequent USER scans. Fresh enumeration
        # after configuring this probe failed on this hw_server; records retained.
        session=JtagSession(output/"xsdb.log",experimental_probe=True,program_probe_svf=build_path/"jtag_probe.svf")
        try:
            session.reset()
            hello=parse_reply(session.scan(bytes(16))["data"])
            if hello["status"]!=9:raise ValueError("wrong USER1 startup state")
            result["USER1_signature"]=True
            sequence=0xfffffffb
            first=session.transact(sequence,b"",command=3,capture=True)
            if first["status"]!=0:raise ValueError("initial reset transaction failed")
            data=bytes((i*73+19)%256 for i in range(6144))
            for rate in (750000,1500000,3000000,6000000,12000000):
                session.set_frequency(rate)
                for n in (32,256,1024,3072,6144):
                    for capture in (True,False):
                        for repeat in range(3):
                            sequence=(sequence+1)&0xffffffff
                            measured=session.transact(sequence,data[:n],capture=capture)
                            if measured["status"]!=0:raise ValueError(f"probe rejected valid payload: {measured}")
                            measured.update(payload_bytes=n,repeat=repeat,sequence=sequence,
                                useful_bits_s=n*8000/measured["wall_ms"],
                                upload_useful_bits_s=n*8000/measured["upload"]["wall_ms"],
                                source_sha256=hashlib.sha256(data[:n]).hexdigest())
                            records.append(measured)
                (output/"measurements.json").write_text(json.dumps(records,indent=2))
                print("PHYSICAL_USER1_TCK_PASS",rate,flush=True)
            # Duplicate and gap must not be accepted; valid next identity recovers.
            duplicate=session.transact(sequence,b"x")
            if duplicate["status"]!=3:raise ValueError("duplicate accepted")
            gap=session.transact((sequence+2)&0xffffffff,b"x")
            if gap["status"]!=3:raise ValueError("sequence gap accepted")
            def terminal(expected, packet, status):
                session.scan(packet,capture=False)
                deadline=time.perf_counter()+3
                while time.perf_counter()<deadline:
                    reply=parse_reply(session.scan(bytes(16))["data"])
                    if reply["sequence"]==expected and reply["status"]==status:return reply
                raise TimeoutError(f"expected explicit status {status}")
            next_seq=(sequence+1)&0xffffffff
            corrupt=bytearray(probe_packet(next_seq,b"crc-test"));corrupt[-2]^=1
            terminal(next_seq,bytes(corrupt),1)
            terminal(next_seq,probe_packet(next_seq,b"truncated")[:-1],2)
            recovery=session.transact(next_seq,b"recovery",capture=True)
            if recovery["status"]!=0:raise ValueError("error recovery failed")
            session.reset()
            if parse_reply(session.scan(bytes(16))["data"])["status"]!=9:raise ValueError("TAP reset recovery failed")
            result.update(status="PASS",transactions=len(records),errors=0,USER1_verified=True,
                CRC_failure_rejected=True,truncation_rejected=True,duplicate_rejected=True,
                gap_rejected=True,reset_recovery=True,sequence_wrap_verified=True,
                frame_eye_association="NOT_APPLICABLE_PROBE_HAS_NO_CNN_FRAME_PROTOCOL")
        finally:
            session.close()
    except Exception as exc:
        result.update(status="FAIL",error=repr(exc))
        raise
    finally:
        (output/"result.json").write_text(json.dumps(result,indent=2))
        if changed:
            restored=program_svf(rollback/"eye_top.svf",output/"rollback_program")
            if restored["status"]!="PASS":raise RuntimeError("baseline rollback failed; stop all hardware interaction")
            from fpga_host import FPGAClient
            from pretrained_eye import PretrainedEye
            from serial.tools import list_ports
            import numpy as np
            ports=[p for p in list_ports.comports() if p.vid==0x10c4 and p.pid==0xea60 and p.serial_number=="04D5A2633F36EF11B7ABFF13A04C7D1C"]
            if len(ports)!=1:raise RuntimeError("registered rollback UART not found")
            client=FPGAClient(ports[0].device,1000000)
            try:
                model=PretrainedEye(precision="fixed");identity=client.load_model(model)
                fixtures=[np.full((32,32,3),v,np.uint8) for v in (0,127,255)]
                fixtures.append(np.random.default_rng(417).integers(0,256,(32,32,3),np.uint8))
                checks=[]
                for i,image in enumerate(fixtures):
                    golden=tuple(int(v) for v in model.predict(image,fixed=True))
                    actual=client.infer_pretrained(image,image,i)
                    if actual.left_logits!=golden or actual.right_logits!=golden:raise ValueError("rollback CNN mismatch")
                    checks.append({"frame_id":i,"logits":golden,"cycles":actual.pair_cycles})
                (output/"rollback.json").write_text(json.dumps({"status":"PASS","identity":identity,
                    "golden_eyes":8,"integer_mismatches":0,"cases":checks,"flash_programmed":False},indent=2))
            finally:client.close()


def coexistence(output):
    """Area/timing gate for verified transport beside unchanged CNN, no handoff yet."""
    from rtl_evidence import _build_project,_check_pretrained_weights
    _check_pretrained_weights()
    output.mkdir(parents=True,exist_ok=False)
    for folder in ("rtl","generated"):shutil.copytree(ROOT/"fpga"/folder,output/folder)
    for path in (ROOT/"fpga/experimental").glob("*.v"):shutil.copy2(path,output/"rtl"/path.name)
    top=output/"rtl/eye_top.v"
    text=top.read_text()
    if text.count("endmodule")!=1:raise ValueError("unexpected canonical top structure")
    top.write_text(text.replace("endmodule","wire [3:0] unused_jtag_led;\n    jtag_probe jtag_preflight(clk,unused_jtag_led);\nendmodule"))
    constraints=(ROOT/"fpga/board_pins.ucf").read_text()
    constraints+='\n'+SCAN_CONSTRAINTS.rstrip('\n').replace('NET "tck"','NET "jtag_preflight/tck"').replace('"stream/','"jtag_preflight/stream/')
    (output/"board_pins.ucf").write_text(constraints)
    sources=" ".join(p.name for p in sorted((output/"rtl").glob("*.v")))
    (output/"build.tcl").write_text('''project new eye_top.xise
project set family Spartan6
project set device xc6slx9
project set package ftg256
project set speed -2
project set synthesis_tool {XST (VHDL/Verilog)}
'''+f'foreach name {{{sources}}} {{xfile add rtl/$name}}\n'+'''xfile add board_pins.ucf
project set top eye_top
project set {Unused IOB Pins} Float -process {Generate Programming File}
project set {FPGA Start-Up Clock} {JTAG Clock} -process {Generate Programming File}
project set {UserID Code (8 Digit Hexadecimal)} 4A544732 -process {Generate Programming File}
foreach step {{Synthesize - XST} {Implement Design} {Generate Programming File}} {
 if {![process run $step]} {project close;error "Failed $step"}
}
project close
''')
    # Non-invasive elaboration. The dedicated primitive is inactive in this
    # resource preflight simulation; CNN/UART arithmetic is covered separately.
    (output/"bscan_stub.v").write_text('''module BSCAN_SPARTAN6 #(parameter JTAG_CHAIN=1)(output CAPTURE,DRCK,RESET,RUNTEST,SEL,SHIFT,TCK,TDI,TMS,UPDATE,input TDO);
assign {CAPTURE,DRCK,RUNTEST,SEL,SHIFT,TCK,TDI,TMS,UPDATE}=0;assign RESET=1;endmodule
module BUFG(input I,output O);assign O=I;endmodule
''')
    binary=ROOT/"assets/tools/iverilog/ucrt64/bin"
    env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ["PATH"])
    with (output/"elaboration.log").open("w") as log:
        subprocess.run([str(binary/"iverilog.exe"),"-g2001","-s","eye_top","-o","coexist.vvp",
                        "bscan_stub.v",*[f"rtl/{name}" for name in sources.split()]],cwd=output,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    result=_build_project(output,2,Path(r"C:\Xilinx\14.7\ISE_DS"),"pretrained",1000000)
    result.update(scope="co-residency resource/timing preflight only; no JTAG-to-CNN handoff",
                  physical_candidate="NOT_PROGRAMMED",source_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ("rtl","generated") for p in (output/folder).glob("*") if p.is_file()})
    (output/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("simulate","build","program-probe","coexistence","prepare-cnn","build-cnn","simulate-cnn"));parser.add_argument("output",type=Path)
    args=parser.parse_args();globals()[args.operation.replace("-","_")](args.output.resolve())
