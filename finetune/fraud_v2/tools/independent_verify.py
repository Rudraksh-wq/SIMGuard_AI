import json, sys, math, hashlib, csv
import numpy as np, torch, laya
sys.path.insert(0, 'D:/safe/finetune/fraud_v2')
from fraud_common import QUESTION_ID, QUESTION
D='D:/safe/finetune/fraud_v2/'
FT=D+'runs/20261113_171802_a1_6ep/final'
gate=json.load(open(FT+'/gate.json'))
def rd(n): return [json.loads(l) for l in open(D+f'data/{n}.jsonl')]
def lab(r): return int(round(r['gold'][QUESTION_ID]['probabilities']['true']))
sets={n:rd(n) for n in ['train','val','test','hard_dev','hard_test']}
out={}
# leakage
tr={r['state'] for r in sets['train']}
out['leakage']={n:sum(r['state'] in tr for r in sets[n]) for n in ['val','test','hard_dev','hard_test']}
out['dup_within_test']=len(sets['test'])-len({r['state'] for r in sets['test']})
for n in ['test','hard_test']:
    out.setdefault('hard_vs_test_overlap',0)
out['hard_test_vs_hard_dev_overlap']=len({r['state'] for r in sets['hard_test']}&{r['state'] for r in sets['hard_dev']})
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [(c-h)/d,(c+h)/d]
def run(path,rows):
    a=laya.load(path,device='cuda'); q={QUESTION_ID:dict(QUESTION)}
    ps=[]
    for i in range(0,len(rows),64):
        r=a.predict_batch([x['state'] for x in rows[i:i+64]],q,batch_size=64)
        ps+= [x['answers'][QUESTION_ID]['noul'] for x in r]
    del a; torch.cuda.empty_cache(); return np.array(ps)
res={}
for mname,path in [('base','convaiinnovations/laya'),('fine_tuned',FT)]:
    for sname in ['test','hard_test']:
        rows=sets[sname]; y=np.array([lab(r) for r in rows]); p=run(path,rows)
        pred=(p>0.5).astype(int); k=int((pred==y).sum()); n=len(y)
        tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
        fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
        ll=float(-np.mean(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))))
        m={'n':n,'acc':k/n,'acc_ci95':wilson(k,n),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'recall_fraud':tp/max(tp+fn,1),'recall_legit':tn/max(tn+fp,1),
           'precision_fraud':tp/max(tp+fp,1),'log_loss':ll,'fraud_rate':float(y.mean()),'majority_acc':max(y.mean(),1-y.mean()),'p_mean':float(p.mean())}
        if mname=='fine_tuned':
            allow=p<=gate['lo']; block=p>=gate['hi']; noul=~(allow|block)
            m['gate']={'block':int(block.sum()),'allow':int(allow.sum()),'human':int(noul.sum()),
              'fraud_leaked':int((allow&(y==1)).sum()),'false_block':int((block&(y==0)).sum()),
              'auto_decided_acc':float(((block&(y==1))|(allow&(y==0))).sum()/max((block|allow).sum(),1))}
            if sname=='hard_test':
                cats={}
                for c in sorted({r['meta']['category'] for r in rows}):
                    idx=np.array([r['meta']['category']==c for r in rows]); cats[c]=[int(idx.sum()),float((pred[idx]==y[idx]).mean())]
                m['by_cat']=cats
        else:
            if sname=='hard_test':
                cats={}
                for c in sorted({r['meta']['category'] for r in rows}):
                    idx=np.array([r['meta']['category']==c for r in rows]); cats[c]=[int(idx.sum()),float((pred[idx]==y[idx]).mean())]
                m['by_cat']=cats
        if mname=='fine_tuned' and sname=='hard_test':
            ref={r['case_id']:float(r['p_fraud']) for r in csv.DictReader(open(D+'runs/20261113_171802_a1_6ep/hard_test_predictions.csv'))}
            d=[abs(ref[r['meta']['case_id']]-pp) for r,pp in zip(rows,p)]
            m['max_abs_diff_vs_logged_csv']=max(d)
        res[f'{mname}/{sname}']=m
        np.save(f'{sys.argv[1]}/{mname}_{sname}_p.npy',p)
out['results']=res
json.dump(out,open(sys.argv[1]+'/verify.json','w'),indent=1)
print(json.dumps(out,indent=1))
