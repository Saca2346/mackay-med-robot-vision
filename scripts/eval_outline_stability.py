"""Replay outline stability against explicitly labelled approximate visual references.

This evaluates display geometry only, never identity/PICK. A constant reference
requires a stationary clip; accuracy numbers are proxy measurements unless the
reference has independently reviewed ground truth provenance.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bbox_precision_eval import detector, boxes
from src.box_geometry import poly_iou
from src.box_outline import refine_box_outline
from src.box_outline_track import OutlineTracker, align_corners


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--weights',default='results/bbox_precision_20261018/model/y26_webcam_ft_544x960_openvino_model')
    p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    ref=json.loads(Path(args.reference).read_text())
    targets={name:np.asarray(poly,np.float32) for name,poly in ref['targets'].items()}
    trackers={name:OutlineTracker() for name in targets}
    model=detector(args.weights,[544,960])
    cap=cv2.VideoCapture(args.video)
    timing=Path(args.video).with_suffix('.times.csv')
    times=[]
    if timing.exists():
        with timing.open() as f: times=[float(r['wall_seconds']) for r in csv.DictReader(f)]
    fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else cap.get(cv2.CAP_PROP_FPS)
    writer=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,590))
    assert cap.isOpened() and writer.isOpened()
    records=[];previous={};index=0
    try:
        with (out/'frames.jsonl').open('w') as log:
            while True:
                ok,frame=cap.read()
                if not ok or (args.max_frames and index>=args.max_frames):break
                found,ms=boxes(model,frame)
                before,after=frame.copy(),frame.copy()
                for name,truth in targets.items():
                    choice=max(range(len(found)),key=lambda k:poly_iou(found[k],truth)) if found else None
                    if choice is None or poly_iou(found[choice],truth)<.45:
                        trackers[name].reset();previous.pop(name,None)
                        row={'frame':index,'target':name,'missing':True}
                    else:
                        raw=found[choice];neighbours=[q for i,q in enumerate(found) if i!=choice]
                        refined=refine_box_outline(frame,raw,neighbours)
                        tick=time.perf_counter();stable=trackers[name].update(frame,refined,name,neighbours,observed_poly=raw)
                        cost=(time.perf_counter()-tick)*1000
                        row={'frame':index,'target':name,'missing':False,'mode':trackers[name].mode,
                             'raw_iou':poly_iou(raw,truth),'refined_iou':poly_iou(refined,truth),
                             'stable_iou':poly_iou(stable,truth),'stabilizer_ms':cost,'detector_ms':ms,
                             'raw':raw.tolist(),'refined':refined.tolist(),'stable':stable.tolist()}
                        if name in previous:
                            for label,poly in [('raw',raw),('refined',refined),('stable',stable)]:
                                old=previous[name][label]
                                row[label+'_step_px']=float(np.linalg.norm(align_corners(poly,old)-old,axis=1).mean())
                        previous[name]={'raw':raw,'refined':refined,'stable':stable}
                        cv2.polylines(before,[np.rint(refined).astype(np.int32)],True,(0,160,255),3)
                        cv2.polylines(after,[np.rint(stable).astype(np.int32)],True,(0,210,0),3)
                    records.append(row);log.write(json.dumps(row)+'\n')
                vis=np.hstack([cv2.resize(before,(960,540)),cv2.resize(after,(960,540))])
                vis=cv2.copyMakeBorder(vis,50,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
                seconds=times[index]-times[0] if index<len(times) else index/fps
                cv2.putText(vis,f'Existing frame-by-frame outline | {seconds:.1f}s | geometry only',(10,30),0,.7,(255,255,255),1)
                cv2.putText(vis,'Motion-checked stable outline | identity not evaluated',(970,30),0,.7,(255,255,255),1)
                writer.write(vis)
                if index in (0,10,100):cv2.imwrite(str(out/f'comparison_{index:04d}.jpg'),vis)
                index+=1
                if index%100==0:print(f'{index} frames',flush=True)
    finally:
        cap.release();writer.release()
    summary={'video':args.video,'weights':args.weights,'frames':index,'reference_provenance':ref.get('provenance'),
             'identity_evaluated':False,'targets':{}}
    for name in targets:
        rows=[r for r in records if r['target']==name and not r['missing']]
        metrics={'found_frames':len(rows),'missing_frames':index-len(rows)}
        for metric in ('raw_iou','refined_iou','stable_iou','raw_step_px','refined_step_px','stable_step_px','stabilizer_ms'):
            vals=[r[metric] for r in rows if metric in r]
            metrics[metric]={'median':float(np.median(vals)),'p95':float(np.percentile(vals,95))} if vals else None
        steps=[r['stable_step_px'] for r in rows if 'stable_step_px' in r]
        metrics['exactly_stationary_fraction']=float(np.mean(np.asarray(steps)==0)) if steps else None
        metrics['modes']={mode:sum(r['mode']==mode for r in rows) for mode in sorted({r['mode'] for r in rows})}
        summary['targets'][name]=metrics
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
