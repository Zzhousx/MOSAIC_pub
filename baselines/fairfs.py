"""FairFS-DCN author pretrain/eval and uniform evaluation; no author retraining stage."""
from pathlib import Path
import argparse,csv,json,os,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1];os.chdir(R)
sys.path.insert(0,str(R/'scripts'))
import experiment_utils as experiments
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['ml1m20','avazu','criteo']);p.add_argument('--fair-python',default=os.environ.get('FAIRFS_PYTHON',sys.executable));p.add_argument('--select-norm',action='store_true');p.add_argument('--dry',action='store_true');a=p.parse_args()
experiments.DRY=a.dry
run=experiments.run;evaluate=experiments.evaluate_ranking;cfg=experiments.CFG;ds=a.dataset
work=R/'third_party/FuxiCTR/model_zoo/DCN/DCN_torch'
if not work.exists() and not a.dry:raise SystemExit('Run baselines/setup_fairfs.py first')
export=[sys.executable,'baselines/export_fuxictr.py','--name',ds,'--fuxictr','third_party/FuxiCTR','--model','DCN',
        '--emb_dim','32','--batch_size','10000','--epochs','10','--min_count','1']
source=cfg['datasets'][ds]['source']
export += ['--npz',source['npz']] if 'npz' in source else ['--cache',str(Path(source['data_path'])/source['erase_dataset']/'mosaic_cache')]
mapping=R/f'third_party/FuxiCTR/data/mosaic_{ds}/field_names.json'
if not mapping.exists():run(export)


def rank(seed,norm,name):
    expid=f'FairFS_DCN_{ds}_s{seed}'
    fixed=[a.fair_python,'run_expid_embigNorm_select.py','--expid',expid,'--normk',str(float(norm)),
           '--interp',str(cfg['fairfs'][ds]['interp']),'--baseline','smooth','--gpu','0']
    out=R/f'results/rankings/{ds}_{name}_seed{seed}.csv';out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        run(fixed+['--stage','pretrain'],work);run(fixed+['--stage','eval'],work)
        author=work/f'checkpoints/mosaic_{ds}/feature_importance_result_k={float(norm)}_baseline=smooth_itp={cfg["fairfs"][ds]["interp"]}.csv'
        if not a.dry:shutil.copy2(author,out)
    if a.dry:
        print(f'# Evaluate preserved ranking: {out}')
    else:evaluate(ds,name,out,seed)
    return out


norm=cfg['fairfs'][ds]['normk']
if a.select_norm:
    if ds!='ml1m20':raise SystemExit('Author settings are fixed for Avazu/Criteo')
    candidates={}
    for value in cfg['fairfs_validation_grid']:
        name='FairFSk'+str(value).replace('.','p')
        rank(2027,value,name);candidates[name]=value
    if not a.dry:
        # Read each candidate's own protocol-checked cache; never merge bare keys across datasets/models.
        scores={}
        base=json.loads((R/'results/ml1m20_deepfm_s0_v2.json').read_text());idx={f:i for i,f in enumerate(base['fields'])}
        for name in candidates:
            path=R/f'results/ml1m20_deepfm_{name}_seed2027.json';j=json.loads(path.read_text());cache=json.loads(Path(str(path)+'.cache.json').read_text())
            assert cache['__config__']==base['protocol']
            vals=[cache[f'{sum(1<<idx[f] for f in fields)}|{s}|full']['val_auc'] for fields in j['subsets'].values() for s in [101,102,103]]
            scores[name]=sum(vals)/len(vals)
        best=max(scores,key=lambda name:(scores[name],name));norm=candidates[best]
        (R/'results/ml1m20_fairfs_normk_selection.json').write_text(json.dumps(dict(criterion='mean validation AUC only',candidates=scores,selected=best),indent=2))
        # The selected2027 ranking is already available; do not retrain its scoring model again.
        shutil.copy2(R/f'results/rankings/ml1m20_{best}_seed2027.csv',R/'results/rankings/ml1m20_FairFS_seed2027.csv')
for seed in [2027,2028,2029]:rank(seed,norm,'FairFS')
