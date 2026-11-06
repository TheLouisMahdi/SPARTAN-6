"""Deterministic compact-model recipe; validation selects, TRAIN calibrates, TEST evaluates.

Requires the approved CEW manifest and source files. No camera or devices opened.
Do not redistribute source images without the dataset owner's permission.
"""
from pathlib import Path
import argparse,csv,hashlib,json,random,sys,re
from types import SimpleNamespace
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Frozen CEW photo-group TinyEye training; authorized local dataset required')
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args_cli=parser.parse_args()
OUT=args_cli.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
DATA=args_cli.dataset.resolve()
sys.path.insert(0,str(ROOT/'python'))
from research import fit_one,metrics
from live_pipeline import read_parameters
from integer_model import infer
torch.set_num_threads(4)
provenance=json.loads((DATA/'provenance.json').read_text())
assert provenance['approved_local_use'] is True
rows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')))
groups={};seen=set()
for row in rows:
    assert row['source_type']=='eye_patch'
    path=Path(row['source_path']);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==row['source_sha256'] and sha not in seen;seen.add(sha)
    key=path.name.split('_face_')[0]
    groups.setdefault((row['label'],key),[]).append(row)
partitions={s:[] for s in ('train','val','test')};rng=random.Random(20261106)
for label in ('OPEN','CLOSED'):
    keys=sorted(k for k in groups if k[0]==label);rng.shuffle(keys)
    ntest=round(.2*len(keys));nval=round(.15*len(keys))
    for i,key in enumerate(keys):
        split='test' if i<ntest else 'val' if i<ntest+nval else 'train'
        for row in groups[key]:partitions[split].append(dict(row,photo_group=key[1],split=split))
(OUT/'frozen_split.json').write_text(json.dumps(dict(seed=20261106,grouping='original photo before _face_; all cropped eyes/faces stay together',
    limitations='subject identities unavailable; no unseen-subject generalization claim',partitions=partitions),indent=2))
pixels={}
for split,items in partitions.items():
    values=[]
    for row in items:
        gray=cv2.cvtColor(cv2.imread(row['source_path']),cv2.COLOR_BGR2GRAY)
        roi=cv2.resize(gray,(32,16),interpolation=cv2.INTER_AREA)
        values.append((roi//64)*np.uint8(85))
    pixels[split]=np.asarray(values,np.uint8)[:,None]
args=SimpleNamespace(resume=False,epochs=12,batch_size=128,learning_rate=.001)
target=OUT/'compact_model'
fit_one(target,'grayscale2',42,pixels,partitions,args,torch.device('cpu'))
params=read_parameters(target/'int8_parameters.npz');evaluations={}
for split in ('val','test'):
    outputs=np.concatenate([infer(pixels[split][i:i+128],params) for i in range(0,len(pixels[split]),128)])
    labels=np.asarray([0 if r['label']=='OPEN' else 1 for r in partitions[split]])
    evaluations[split]=metrics(labels,outputs.argmax(axis=1))
record=dict(status='TRAINED_OFFLINE_PHYSICAL_PENDING',architecture='tiny-eye-v1',parameters=1130,
    encoder='grayscale2-v1',input=[1,16,32],bytes_per_eye=128,classes=['OPEN','CLOSED'],
    checkpoint_sha256=hashlib.sha256((target/'int8_parameters.npz').read_bytes()).hexdigest(),
    split_sha256=hashlib.sha256((OUT/'frozen_split.json').read_bytes()).hexdigest(),
    split_counts={s:len(v) for s,v in partitions.items()},evaluation=evaluations,
    selection='12 epochs, one seed, validation-only selection, TRAIN-only quantization calibration',
    limitations=['Within-CEW photo-group holdout; subjects unknown','Not live camera accuracy','No TinyEye physical measurement'])
(OUT/'model.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
