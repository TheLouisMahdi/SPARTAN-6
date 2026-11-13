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
    export_flow_diagnostic(directory, metadata, summary, events)
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


def export_flow_diagnostic(directory, metadata, summary, events):
    """Build one bounded, evidence-labelled graph and timeline from existing trace events."""
    directory = Path(directory)
    model_method = metadata.get('model_method')
    pretrained = model_method == 'omz_bgr32'
    def node(identifier, name, path, symbol, owner, input_spec, output_spec, transform):
        return dict(id=identifier, name=name, source_file=path, source_symbol=symbol,
                    owner=owner, input=input_spec, output=output_spec, transform=transform)
    nodes = [
        node('capture','Frame acquisition','PC/capture.py','capture_frames','CPU: camera-capture',
             'webcam BGR uint8 HWC','frame_id, capture_ns, BGR uint8 HWC','latest-frame bounded capture queue'),
        node('landmarks','Face landmarks','PC/live_pipeline.py','run_streaming.Prepare.__call__','CPU: eye-prepare',
             'BGR uint8 HWC','MediaPipe normalized landmarks','BGR to RGB, FaceLandmarker.detect_for_video'),
        node('roi','Eye crop','PC/capture.py','extract_eye_roi','CPU: eye-prepare',
             'BGR frame and eye-corner landmarks','left/right bounded eye ROIs','landmark geometry; original BGR crop retained for pretrained'),
        node('tensor','Tensor preparation','PC/pretrained_eye.py' if pretrained else 'PC/encoders',
             'input_pixels' if pretrained else 'METHODS[model.method].encode','CPU: eye-prepare',
             'left/right ROI','uint8 CHW [3,32,32]' if pretrained else 'uint8 encoded [16,32]',
             'resize BGR and transpose HWC to CHW' if pretrained else 'encode grayscale eye crop'),
        node('queue','Prepared queue','PC/stream_pipeline.py','StreamingPipeline._prepare','CPU: eye-prepare',
             'PreparedFrame','bounded PreparedFrame queue','overwrite latest when full; preserve original IDs'),
        node('uart_upload','UART upload','PC/fpga_host.py','FPGAClient.infer_pretrained' if pretrained else 'FPGAClient.infer',
             'CPU: eye-fpga','tensor bytes','CRC framed UART packets','blocking request-response, single UART owner'),
        node('packet_rx','Packet reception','FPGA/rtl/packet_rx.v','packet_rx','FPGA',
             'UART bytes','CRC checked payload','framing and CRC validation'),
        node('controller','CNN dispatch','FPGA/rtl/command_controller.v','command_controller','FPGA',
             'upload chunks and inference command','memory writes, CNN start','three upload chunks per eye for pretrained'),
        node('cnn','CNN inference','FPGA/rtl/omz_cnn.v' if pretrained else 'FPGA/rtl/eye_cnn.v',
             'omz_cnn' if pretrained else 'eye_cnn','FPGA','fixed-point input tensor',
             'two signed INT32 native logits and cycles','fixed-point CNN, one eye per invocation for pretrained'),
        node('packet_tx','FPGA response','FPGA/rtl/packet_tx.v','packet_tx','FPGA',
             'CNN logits, cycles, status','CRC framed UART reply','serialize response'),
        node('decode','Host decode','PC/fpga_host.py','FPGAClient.infer_pretrained' if pretrained else 'FPGAClient.infer',
             'CPU: eye-fpga','native logits and cycles','canonical OPEN/CLOSED logits','native [CLOSED,OPEN] mapped to [OPEN,CLOSED] for pretrained'),
        node('publish','Freshness and publication','PC/stream_pipeline.py','StreamingPipeline._result','CPU: eye-result',
             'InferenceResult','PublishedFrame','reject stale or missing results; preserve raw logits'),
        node('blink','Blink tracking','PC/blink_timing.py','BinocularTimer.update','CPU: eye-result',
             'two valid CLOSED/OPEN observations and capture timestamp','blink event and counters',
             'capture-clock binocular transition and gap checking'),
        node('gui','Preview','PC/live_pipeline.py','render_stream_preview','CPU: caller GUI',
             'PublishedFrame','annotated BGR image','expire displayed states after maximum age'),
        node('report','Reports','PC/pipeline_report.py','summarize','CPU: caller',
             'trace.jsonl + session metadata','summary.json, flow_diagnostic.json','post-session aggregation without inference')]
    links = [('capture','landmarks'),('landmarks','roi'),('roi','tensor'),('tensor','queue'),
             ('queue','uart_upload'),('uart_upload','packet_rx'),('packet_rx','controller'),
             ('controller','cnn'),('cnn','packet_tx'),('packet_tx','decode'),('decode','publish'),
             ('publish','blink'),('publish','gui'),('capture','report'),('publish','report')]
    if metadata.get('source') == 'ROI_REPLAY_ONLY':
        nodes = [n for n in nodes if n['id'] not in ('capture','landmarks','roi')]
        nodes.insert(0,node('replay','ROI replay','PC/live_pipeline.py','run_streaming.source','CPU: roi-source',
                            'registered eye samples','synthetic frame ID, capture timestamp, ROI pair',
                            'synthetic replay, not physical camera measurement'))
        links = [('replay','tensor') if a=='capture' and b=='landmarks' else (a,b) for a,b in links
                 if a not in ('capture','landmarks','roi') and b not in ('capture','landmarks','roi')]
        links.insert(0,('replay','tensor'))
    frames = {}
    components = {}
    global_events = []
    def frame(fid):
        return frames.setdefault(fid, {'frame_id':fid, 'capture_ns':None, 'events':{}, 'drops':[]})
    for event in events:
        kind, fid = event.get('kind'), event.get('frame_id')
        if kind == 'preparation_components':
            components[event['capture_ns']] = event
        elif isinstance(fid,int):
            record = frame(fid)
            if kind in ('dropped_capture','dropped_preparation','dropped_display','dropped_timestamp','missing_frame','display_expired'):
                record['drops'].append(event)
            else:
                record['events'][kind] = event
            if kind == 'capture': record['capture_ns'] = event.get('capture_ns')
            elif kind in ('prepared','published') and record['capture_ns'] is None:
                record['capture_ns'] = event.get('capture_ns')
        else:
            global_events.append(event)
    def field(value, source='observed'):
        return {'value':value, 'provenance':source if value is not None else 'unavailable'}
    def ms(start,end):
        return field(round((end-start)/1e6,6),'derived') if isinstance(start,int) and isinstance(end,int) and end>=start else field(None)
    records=[]
    previous_published_ns=None
    for fid,record in sorted(frames.items()):
        e=record['events']; prep=e.get('prepared',{}); inf=e.get('inference',{}); pub=e.get('published',{})
        comp=components.get(record['capture_ns'],{})
        capture_ns=record['capture_ns']
        eyes=pub.get('eyes')
        raw=pub.get('raw_logits')
        reason=pub.get('reason') or inf.get('reason')
        tags=[]
        if prep and prep.get('valid_eyes',2)!=2: tags.append('missing_eye_roi')
        if reason: tags.append(reason)
        if raw is None and inf.get('cycles') is None and inf: tags.append('no_fpga_result')
        if raw is not None and any(pair is not None and len(pair)==2 and pair[0]==pair[1] for pair in raw):
            tags.append('equal_logits')
        if any(x['kind']=='display_expired' for x in record['drops']): tags.append('gui_display_expired')
        if any(x['kind'].startswith('dropped_') for x in record['drops']): tags.append('queue_or_timestamp_drop')
        if pub and previous_published_ns is not None and capture_ns is not None:
            if (capture_ns//1000000 - previous_published_ns//1000000)>250: tags.append('temporal_gap_gt_250ms')
        if pub and capture_ns is not None: previous_published_ns=capture_ns
        native=None
        if pretrained and raw is not None:
            native=[[pair[1],pair[0]] for pair in raw]
        stamps={key:field(value) for key,value in {
            'capture_ns':capture_ns, 'landmark_start_ns':comp.get('landmark_start_ns'),
            'landmark_end_ns':comp.get('landmark_end_ns'),'roi_end_ns':comp.get('roi_end_ns'),
            'prep_start_ns':prep.get('prep_start_ns'),'prep_end_ns':prep.get('prep_end_ns'),
            'fpga_start_ns':inf.get('fpga_start_ns'),'fpga_end_ns':inf.get('fpga_end_ns'),
            'result_start_ns':pub.get('result_start_ns'),'result_end_ns':pub.get('result_end_ns')}.items()}
        measurements={
            'landmarks_ms':ms(comp.get('landmark_start_ns'),comp.get('landmark_end_ns')),
            'roi_extraction_ms':ms(comp.get('landmark_end_ns'),comp.get('roi_end_ns')),
            'preparation_ms':ms(prep.get('prep_start_ns'),prep.get('prep_end_ns')),
            'uart_upload_request_group_ms':field(inf.get('uart_upload_ms')),
            'uart_transaction_total_ms':field(inf.get('uart_roundtrip_ms')),
            'uart_response_only_ms':field(None),
            'fpga_pair_cycles':field(inf.get('cycles')),
            'end_to_end_ms':ms(capture_ns,pub.get('result_end_ns')),
            'frame_age_at_inference_start_ms':ms(capture_ns,inf.get('fpga_start_ns')),
            'frame_age_at_publication_ms':ms(capture_ns,pub.get('result_end_ns')),
        }
        tensor=prep.get('tensor_metadata')
        roi_valid=({side:side in (prep.get('tensor_metadata') or {}) for side in ('left','right')} if prep else None)
        decision_valid=({side:data.get('state') in ('OPEN','CLOSED') for side,data in eyes.items()} if eyes else None)
        records.append(dict(frame_id=fid,capture_timestamp=field(capture_ns),
              source_ids=field(e.get('capture',{}).get('source_ids')),timestamps=stamps,measurements=measurements,
              roi_validity=field(roi_valid,'derived'),decision_validity=field(decision_valid,'derived'),
              bounding_boxes=field(prep.get('boxes')),tensor_metadata=field(tensor),
              input_sha256=field(prep.get('input_sha256')),approved_tensor_samples=prep.get('approved_tensor_samples'),
              raw_logits_canonical=field(raw),raw_logits_native=field(native,'derived_class_permutation'),
              predictions=field(eyes),validity_reason=field(reason),validity_tags=tags,
              blink_count=field(pub.get('blink_count')),blink_event=field(pub.get('blink_event')),
              dropped_events=[dict(kind=x['kind'],at_ns=x.get('at_ns'),reason=x.get('reason'),replaced_by=x.get('replaced_by')) for x in record['drops']]))
    anomaly=lambda r: bool(r['validity_tags']) or r['approved_tensor_samples'] is not None or (r['predictions']['value'] is not None and any(v['state']=='UNAVAILABLE' for v in r['predictions']['value'].values()))
    approved=[r['frame_id'] for r in records if r['approved_tensor_samples'] is not None]
    priority=[r['frame_id'] for r in records if anomaly(r)]
    boundary=[r['frame_id'] for r in records[:32]+records[-32:]]
    ordered_ids=list(dict.fromkeys(approved+priority+boundary))
    selected_ids=set(ordered_ids[:256])
    selected=[r for r in records if r['frame_id'] in selected_ids]
    count_tags={key:sum(key in r['validity_tags'] for r in records) for key in sorted({tag for r in records for tag in r['validity_tags']})}
    missing=['Exact physical exposure timestamp is not available from webcam API',
             'UART one-way TX and RX timestamps are not separately measured',
             'Host and FPGA clocks are not synchronized for kernel start',
             'No operator-labelled closed-eye ground truth is present']
    if any(r['measurements']['landmarks_ms']['value'] is None for r in records):
        missing.append('Landmark stage durations are absent for some frames or ROI-only replay')
    diagnostic=dict(schema_version=1,source_commit=metadata.get('source_revision'),
          session=dict(source=metadata.get('source'),provenance=metadata.get('provenance'),
                       pipeline_mode=metadata.get('mode'),model_method=model_method,baud=metadata.get('baud'),
                       model_sha256=metadata.get('model_sha256'),bitstream_sha256=metadata.get('bitstream_sha256_local_artifact'),
                       status=metadata.get('status')),
          architecture=dict(nodes=nodes,edges=[{'from':a,'to':b,'relationship':'same-frame dependency'} for a,b in links],
                execution_order=[n['id'] for n in nodes],
                concurrency='sequential completion barrier' if metadata.get('mode')=='sequential' else 'inter-frame overlap with bounded queues',
                binocular_association='both eyes from the same original frame ID and capture timestamp'),
          runtime=dict(total_frames=len(records),selected_frames=len(selected),omitted_frames=len(records)-len(selected),
                 selection_policy='approved first; then anomaly frames; then first/last 32 boundary frames; at most 256, returned sorted by frame ID',
                 frames=selected,global_events=global_events[:32],summary_counts=summary.get('counts')),
          findings=dict(observations=[dict(code=k,frame_count=v,evidence='recorded trace events and derived tags',confidence='trace-supported') for k,v in count_tags.items()],
                   hypotheses=[dict(id='H1',text='Latency or queue saturation may reduce observable blink transitions',confidence='unverified',
                                    evidence='Requires controlled sequential/overlap physical comparisons')],
                   missing_measurements=missing,
                   next_experiments=['Offline JSON and mock-UART contract tests',
                                     'Later controlled sequential/overlap physical runs with matched setup',
                                     'Later operator-labelled OPEN/CLOSED eye samples, paired CPU/FPGA logits']),
          privacy=dict(raw_frames_recorded=False,approved_tensor_capture='opt-in up to four frame IDs',cloud_upload=False))
    (directory/'flow_diagnostic.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return diagnostic
