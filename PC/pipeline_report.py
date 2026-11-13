"""Reports from recorded pipeline intervals; no inferred time is a measurement."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from experiment import latency, write_json


def intersection(a,b,c,d):
    return max(0,min(b,d)-max(a,c))


def summarize(directory, metadata, frames_path=None):
    directory=Path(directory)
    events=[json.loads(line) for line in (directory/'trace.jsonl').open()]
    prep=[e for e in events if e['kind']=='prepared']
    inferred=[e for e in events if e['kind']=='inference']
    published=[e for e in events if e['kind']=='published']
    captures=[e for e in events if e['kind']=='capture']
    begin=(captures[0]['capture_ns'] if captures else metadata['start_ns'])+int(metadata.get('warmup_seconds',0)*1e9)
    # Configuration/rollback can take seconds. Inference throughput uses only
    # recorded data-stage events, never firmware setup or session cleanup.
    endpoints=[e[key] for e in events for key in ('capture_ns','prep_end_ns','fpga_end_ns','result_end_ns') if key in e]
    end=max(endpoints) if endpoints else metadata['end_ns']
    duration=max(0,end-begin)/1e9
    selected=[e for e in published if e['capture_ns']>=begin]
    ids={e['frame_id'] for e in selected}
    hw=[e for e in inferred if e['frame_id'] in ids and e['cycles'] is not None]
    good=[e for e in selected if e['reason'] is None and all(v['state']!='UNAVAILABLE' for v in e['eyes'].values())]
    pselected=[e for e in prep if e['capture_ns']>=begin]
    overlap=[]; bounded=[]
    for p in pselected:
        for f in hw:
            if p['frame_id']<=f['frame_id']: continue
            amount=intersection(p['prep_start_ns'],p['prep_end_ns'],f['fpga_start_ns'],f['fpga_end_ns'])
            if amount: overlap.append(dict(cpu_frame=p['frame_id'],fpga_frame=f['frame_id'],ns=amount))
            for lower,upper,kernel in f.get('kernel_envelopes') or []:
                # Kernel has duration D but its host-clock start is unknown in
                # [lower,upper-D]. Minimum single-interval intersection occurs
                # at an endpoint. This is a conservative bound, not direct HW timestamps.
                if upper-lower<kernel: raise ValueError('Kernel does not fit host envelope')
                bound=min(intersection(p['prep_start_ns'],p['prep_end_ns'],lower,lower+kernel),
                          intersection(p['prep_start_ns'],p['prep_end_ns'],upper-kernel,upper))
                if bound: bounded.append(dict(cpu_frame=p['frame_id'],fpga_frame=f['frame_id'],ns=bound))
    total_request=sum(e['fpga_end_ns']-e['fpga_start_ns'] for e in hw)
    overlap_total=sum(v['ns'] for v in overlap)
    counts={kind:sum(e['kind']==kind for e in events) for kind in sorted({e['kind'] for e in events})}
    summary=dict(status=metadata['status'],provenance=metadata.get('provenance','NOT_AVAILABLE'),scope=metadata['source'],
        model_sha256=metadata.get('model_sha256'),bitstream_sha256=metadata.get('bitstream_sha256_local_artifact'),baud=metadata.get('baud'),verify=metadata.get('verify'),
        pipeline_mode=metadata['mode'],measurement_seconds=duration,warmup_seconds=metadata.get('warmup_seconds',0),counts=counts,
        measurement_window_definition='first capture plus warmup through last data-stage event; firmware setup and rollback excluded',
        unpublished_completed_frame_ids=sorted({e['frame_id'] for e in events if e['kind']=='fpga_completed'}-{e['frame_id'] for e in published}),
        invalid_eye_pairs=sum(e['reason']=='missing_eye' for e in selected),stale_frames=sum(e['reason'] in ('stale_input','stale_result') for e in selected),
        dropped_capture_frames=counts.get('dropped_capture',0),dropped_preparation_frames=counts.get('dropped_preparation',0),
        frame_id_errors=sum(e['kind']=='worker_failure' and ('frame' in e.get('error','').lower()) for e in events),
        processed_fps=len(hw)/duration if duration>0 else None,valid_result_fps=len(good)/duration if duration>0 else None,
        capture_fps=sum(e['capture_ns']>=begin for e in captures)/duration if duration>0 else None,
        throughput_eye_samples_per_second=2*len(hw)/duration if duration>0 else None,
        preparation=latency([(e['prep_end_ns']-e['prep_start_ns'])/1e6 for e in pselected]),
        landmarks=latency([(e['landmark_end_ns']-e['landmark_start_ns'])/1e6 for e in events if e['kind']=='preparation_components' and e['capture_ns']>=begin]),
        roi_extraction=latency([(e['roi_end_ns']-e['landmark_end_ns'])/1e6 for e in events if e['kind']=='preparation_components' and e['capture_ns']>=begin]),
        fpga_kernel_pair=latency([(e['cycles'][0]+e['cycles'][1])/50000 for e in hw]),
        fpga_controller_pair=latency([e['cycles'][2]/50000 for e in hw]),
        uart_roundtrip=latency([e['uart_roundtrip_ms'] for e in hw]),
        uart_upload=latency([e['uart_upload_ms'] for e in hw if e.get('uart_upload_ms') is not None]),
        end_to_end=latency([(e['result_end_ns']-e['capture_ns'])/1e6 for e in selected]),
        result_processing=latency([(e['result_end_ns']-e['result_start_ns'])/1e6 for e in selected]),
        pipeline_overlap_ms=overlap_total/1e6,pipeline_overlap_fraction=overlap_total/total_request if total_request else None,
        overlap_definition='Union of serial CPU preparation spans against nonoverlapping request intervals for older frames; includes dropped preparations; excludes same-frame work',
        kernel_overlap_lower_bound_ms=sum(e['ns'] for e in bounded)/1e6,kernel_overlap_provenance='MODELLED_ESTIMATE: conservative lower bound using measured command envelopes and cycle-derived duration; not directly synchronized FPGA timestamps',
        physical_overlap_supported=bool(metadata.get('provenance')=='PHYSICAL_MEASUREMENT' and hw and overlap),overlap_examples=overlap[:40],kernel_overlap_bound_examples=bounded[:40],
        inference_frame_ids=[e['frame_id'] for e in hw],published_frame_ids=[e['frame_id'] for e in selected],
        crc_errors=metadata.get('integrity',{}).get('crc_errors'),timeouts=metadata.get('integrity',{}).get('timeouts'),
        sequence_errors=metadata.get('integrity',{}).get('sequence_errors'),
        temporal_source='CPU BinocularTimer; frame gaps invalidate continuity; onboard equivalence not claimed for this mode',
        limitations=['Live trials have different camera images; no clinical inference.', 'Replay of eye patches excludes camera/MediaPipe.','Kernel duration is cycle-derived at verified project 50 MHz; host interval location is bounded.'])
    write_json(directory/'summary.json',summary)
    frames_path=Path(frames_path) if frames_path else directory/'frames.csv'
    fields=['frame_id','capture_ns','timestamp_ms','left_state','right_state','validity','reason','result_start_ns','result_end_ns','blink_count','blink_event']
    if not frames_path.exists():
      with frames_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for e in published:
            writer.writerow(dict(frame_id=e['frame_id'],capture_ns=e['capture_ns'],timestamp_ms=(e['capture_ns']-metadata['start_ns'])/1e6,
                left_state=e['eyes']['left']['state'],right_state=e['eyes']['right']['state'],validity='valid' if e['reason'] is None else 'invalid',reason=e['reason'],
                result_start_ns=e['result_start_ns'],result_end_ns=e['result_end_ns'],blink_count=e['blink_count'],blink_event=e['blink_event']))
    # Real timeline from up to the first 12 submitted frames; no conceptual bars.
    chosen=[e for e in inferred if e['cycles']][:12]; prepmap={e['frame_id']:e for e in prep}
    svg=[]
    if chosen:
        origin=min(prepmap[e['frame_id']]['prep_start_ns'] for e in chosen)
        extent=max(e['fpga_end_ns'] for e in chosen)-origin
        for n,e in enumerate(chosen):
            p=prepmap[e['frame_id']]
            for row,a,b,color in [(0,p['prep_start_ns'],p['prep_end_ns'],'#167d9a'),(1,e['fpga_start_ns'],e['fpga_end_ns'],'#c17520')]:
                x=100+900*(a-origin)/max(1,extent); width=max(.5,900*(b-a)/max(1,extent))
                svg.append(f'<rect x="{x}" y="{n*40+row*15}" width="{width}" height="12" fill="{color}"><title>Frame {e["frame_id"]}; {"CPU prep" if row==0 else "FPGA request"}; {(b-a)/1e6:.3f} ms</title></rect>')
            svg.append(f'<text x="0" y="{n*40+13}">Frame {e["frame_id"]}</text>')
    (directory/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Spartan-6 pipeline</title><style>body{font:16px system-ui;margin:30px}pre{white-space:pre-wrap}svg{width:100%;max-width:1100px}</style><h1>Spartan-6 pipeline</h1><p>Blue: CPU preparation. Orange: host FPGA request, including UART. Kernel placement is bounded separately.</p><svg viewBox="0 0 1050 '+str(max(40,len(chosen)*40))+'">'+''.join(svg)+'</svg><pre>'+html.escape(json.dumps(dict(session=metadata,summary=summary),indent=2))+'</pre>',encoding='utf-8')
    return summary


def compare(paths, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    runs=[dict(path=str(Path(p).resolve()),**json.loads((Path(p)/'summary.json').read_text())) for p in paths]
    sequential=[r for r in runs if r['pipeline_mode']=='sequential']; overlapped=[r for r in runs if r['pipeline_mode']=='overlap']
    usable=bool(sequential and overlapped and all(r['status']=='PASS' and r['processed_fps'] and not r.get('verify') and
        all(r.get(k)==runs[0].get(k) for k in ('scope','model_sha256','bitstream_sha256','baud')) for r in runs))
    sessions=[json.loads((Path(p)/'session.json').read_text()) for p in paths]
    source_equal=all(s.get('source_sha256')==sessions[0].get('source_sha256') for s in sessions) if sessions else False
    usable=usable and source_equal
    exact=None
    if usable and runs[0]['scope']=='ROI_REPLAY_ONLY':
        usable=all(all(s.get(k)==sessions[0].get(k) for k in ('replay_manifest_sha256','replay_count','replay_fps')) for s in sessions)
        data=[]
        for path in paths:
            rows=[json.loads(line) for line in (Path(path)/'trace.jsonl').open()]
            data.append(({e['frame_id']:e['raw_logits'] for e in rows if e['kind']=='published' and e['raw_logits'] is not None},
                         {e['frame_id']:e['input_sha256'] for e in rows if e['kind']=='prepared'}))
        common=set.intersection(*(set(d[0]) for d in data)) if data else set()
        mismatches=[fid for fid in sorted(common) if any(d[0][fid]!=data[0][0][fid] or d[1][fid]!=data[0][1][fid] for d in data)]
        exact=dict(common_pair_frames=len(common),mismatching_frame_ids=mismatches)
        usable=usable and not mismatches and bool(common)
    ratio=float(np.mean([r['processed_fps'] for r in overlapped])/np.mean([r['processed_fps'] for r in sequential])) if usable else None
    result=dict(status='PASS' if usable else 'NOT_AVAILABLE',provenance='PHYSICAL_MEASUREMENT' if sessions and all(s.get('provenance')=='PHYSICAL_MEASUREMENT' for s in sessions) else 'NOT_AVAILABLE',runs=runs,throughput_ratio=ratio,
        algorithm_source_hashes_equal=source_equal,common_input_exact_comparison=exact,
        comparison_scope=runs[0]['scope'] if runs else None,limitations='Live inputs are uncontrolled between trials; ratio is descriptive, not a statistically rigorous speedup. See sample IDs, drops and per-run scope. ROI replay excludes camera/MediaPipe.')
    write_json(output/'comparison.json',result)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Pipeline comparison</title><h1>Pipeline comparison</h1><pre style="white-space:pre-wrap">'+html.escape(json.dumps(result,indent=2))+'</pre>')
    return result
