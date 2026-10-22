"""Reproduce the experiments present in the current paper; one subprocess at a time."""
from pathlib import Path
import argparse,csv,glob,json,os,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'baselines'))
from erase_rank import ranking_from_npy
CFG=json.loads((ROOT/'configs/paper.json').read_text())
PY=sys.executable
DRY=False


def source(ds):
    s=CFG['datasets'][ds]['source']
    return ['--npz',s['npz']] if 'npz' in s else ['--erase_root','third_party/ERASE','--dataset',s['erase_dataset'],'--data_path',s['data_path']]


def command(stage,ds,model,seed,out,**overrides):
    d=CFG['datasets'][ds]
    opts=dict(model=model,seed=seed,budgets=d['budgets'],batch_size=d['batch_size'],epochs=d['epochs'],
              shared_min_epochs=d['shared_min_epochs'],est_rows=500000,out=out)
    opts.update(overrides)
    argv=[PY,'mosaic/mosaic_gpu.py',stage]+source(ds)
    for k,v in opts.items():argv += ['--'+k,str(v)]
    if d['sparse_emb']:argv.append('--sparse_emb')
    return argv


def run(argv,cwd=ROOT):
    print('$ '+subprocess.list2cmdline(list(map(str,argv))),flush=True)
    if not DRY:
        env=dict(os.environ,SHARK_MAX_SAMPLES='100000')
        subprocess.run(list(map(str,argv)),cwd=cwd,env=env,check=True)


def execute(stage,ds,model,seed,out,**kw):
    if Path(out).exists():return
    run(command(stage,ds,model,seed,out,**kw))


def evaluate_ranking(ds,name,path,seed):
    if str(path).endswith('.csv'):
        rows=list(csv.DictReader(Path(path).open(encoding='utf-8')))
        rows.sort(key=lambda row:-float(row['feature_weight']))
        names=[r['feature_name'][2:] if r['feature_name'].startswith('f_') and r['feature_name'][2:3].isdigit() else r['feature_name'] for r in rows]
        ranking=','.join(names)
    else:ranking=Path(path).read_text().strip() if Path(path).exists() else '<ranking produced by command above>'
    for model in (['deepfm'] if ds=='ml1m40c' else ['deepfm','dcnv2']):
        execute('eval_ranking',ds,model,0,f'results/{ds}_{model}_{name}_seed{seed}.json',ranking=ranking,name=name)


def baseline(ds,method,seed):
    out=Path(f'results/rankings/{ds}_{method}_seed{seed}.txt');out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        if method=='mRMR':
            run([PY,'baselines/mrmr_rank.py','--config','configs/paper.json','--ds',ds,'--seed',seed,'--out',out])
        else:
            e=CFG['erase'][ds];start=time.time();base=ROOT/'third_party/ERASE'
            run([PY,'fs_run.py','--dataset',e['dataset'],'--model','mlp','--fs',method,'--train_or_search','True',
                 '--retrain','False','--data_path',str((ROOT/e['data_path']).resolve())+os.sep,'--num_workers','0',
                 '--epoch',e['epoch'],'--seed',seed],base)
            if not DRY:
                folders=[p for p in base.glob(f'checkpoints/mlp_{method}_no_selection_{e["dataset"]}_*') if p.stat().st_mtime>=start-5]
                assert folders,'ERASE ranking output not found'
                chosen=max(folders,key=lambda p:p.stat().st_mtime)
                out.write_text(','.join(ranking_from_npy(chosen/'feature_rank.npy'))+'\n')
    evaluate_ranking(ds,{'autofield':'AutoField','lpfs':'LPFS','sfs':'SFS','shark':'SHARK'}.get(method,method),out,seed)


