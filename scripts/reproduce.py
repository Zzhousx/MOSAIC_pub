"""Run only the experiments reported in the paper, one subprocess at a time."""
from pathlib import Path
import argparse,json,os,sys
import experiment_utils as experiments
from experiment_utils import ROOT,CFG,PY,source,run,execute,baseline
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=['main','controls','mask-training','sensitivity','copies','scale','serving','agreement','timestamp','baselines','figures'])
p.add_argument('--dataset',choices=['ml1m20','avazu','criteo','ml1m40c']);p.add_argument('--dry',action='store_true')
ARGS=p.parse_args()
experiments.DRY=ARGS.dry
datasets=[ARGS.dataset] if ARGS.dataset else ['ml1m20','avazu','criteo']
Path('results').mkdir(exist_ok=True)
if ARGS.stage=='figures':
    run([PY,'figures/make_figures.py']);run([PY,'figures/make_toy.py'])
elif ARGS.stage=='copies':
    run([PY,'scripts/make_copies.py','--npz','data/ml1m20.npz','--factor','2','--rate','.1','--seed','2031','--out','data/ml1m40c.npz'])
    run([PY,'scripts/export_erase_ml1m.py','--npz','data/ml1m40c.npz','--out','erase_data_ml1m40c'])
elif ARGS.stage=='scale':
    for factor in [1,2,4,8]:
        F=39*factor;npz=f'data/criteo_scale_F{F}.npz';out=f'results/scale_criteo_F{F}.json'
        if Path(out).exists():continue
        cache=Path(CFG['datasets']['criteo']['source']['data_path'])/'criteo/mosaic_cache'
        run([PY,'scripts/make_copies.py','--cache',cache,'--rows','2000000,500000,100000','--factor',factor,'--rate','.1','--seed','2035','--out',npz])
        run([PY,'mosaic/mosaic_gpu.py','costonly','--npz',npz,'--model','deepfm','--budgets','auto','--batch_size','4096','--epochs','1',
             '--shared_min_epochs','1','--shared_max_epochs','1','--sparse_emb','--seed','0','--out',out])
elif ARGS.stage=='timestamp':
    example=json.loads((ROOT/'figures/results/ml1m20_deepfm_sets_timestamp.json').read_text())
    sets=';'.join(n+'='+','.join(v['fields']) for n,v in example['sets'].items())
    execute('eval_sets','ml1m20','deepfm',0,'results/ml1m20_deepfm_sets_timestamp.json',sets=sets)
else:
    for ds in datasets:
        if ARGS.stage=='main':
            for model in (['deepfm'] if ds=='ml1m40c' else ['deepfm','dcnv2']):
                seeds=[0,1,2] if ds=='ml1m40c' else CFG['main_seeds'][model]
                for seed in seeds:execute('run',ds,model,seed,f'results/{ds}_{model}_s{seed}_v2.json')
        elif ARGS.stage=='controls':
            for model in (['deepfm','dcnv2'] if ds=='ml1m20' else ['deepfm']):
                for seed in CFG['control_seeds']:
                    execute('ladder',ds,model,seed,f'results/{ds}_{model}_s{seed}_ladder.json',run_json=f'results/{ds}_{model}_s{seed}_v2.json')
        elif ARGS.stage=='mask-training':
            for seed in CFG['control_seeds']:
                for scheme in ['budget','half','uniform']:
                    execute('trainablate',ds,'deepfm',seed,f'results/{ds}_deepfm_s{seed}_tm-{scheme}.json',train_masks=scheme)
        elif ARGS.stage=='sensitivity':
            if ds not in ['ml1m20','avazu']:continue
            for seed in CFG['control_seeds']:
                execute('sens',ds,'deepfm',seed,f'results/{ds}_deepfm_s{seed}_sens.json',run_json=f'results/{ds}_deepfm_s{seed}_v2.json')
        elif ARGS.stage=='agreement':
            if ds!='ml1m20':continue
            for seed in CFG['control_seeds']:execute('agreement',ds,'deepfm',seed,f'results/{ds}_deepfm_s{seed}_agreement.json')
        elif ARGS.stage=='serving':
            run([PY,'scripts/serving_cost.py']+source(ds)+['--results','results','--out',f'results/{ds}_serving.json'])
        elif ARGS.stage=='baselines':
            for method in ['autofield','lpfs','sfs','shark','mRMR']:
                for seed in CFG['baseline_seeds']:baseline(ds,method,seed)
            if ds!='ml1m40c':
                fair=[PY,'baselines/fairfs.py','--dataset',ds]
                if ARGS.dry:fair.append('--dry')
                run(fair)
