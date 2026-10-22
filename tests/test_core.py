"""CPU mathematical checks; no GPU experiments or dataset training."""
from pathlib import Path
import importlib.util,itertools,json,subprocess,sys,unittest
import numpy as np

R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'mosaic'))
from mosaic_lib import auc_np,logloss_logits,to_mask,bits,mask_vec,quad_value,select_subsets,budget_surrogate


class CoreTests(unittest.TestCase):
    def test_auc_ties(self):
        y=np.array([0,1,0,1]);z=np.array([.1,.7,.7,.9])
        expected=np.mean([(a>b)+.5*(a==b) for a in z[y==1] for b in z[y==0]])
        self.assertAlmostEqual(auc_np(y,z),expected)

    def test_logloss(self):
        self.assertAlmostEqual(logloss_logits([0,1],[0,0]),np.log(2))

    def test_pairwise_moments_and_recovery(self):
        F=4;rng=np.random.default_rng(17);a=rng.normal(size=F);b=rng.normal(size=(F,F));b=(b+b.T)/2;np.fill_diagonal(b,0)
        masks=np.array(list(itertools.product([0,1],repeat=F)),dtype=float);est={}
        values=quad_value(masks,a,b)
        for p in [.2,.5,.8]:
            prob=np.prod(np.where(masks,p,1-p),axis=1);z=(masks-p)/(p*(1-p))
            g=(prob*values)@z;H=z.T@((prob*values)[:,None]*z);np.fill_diagonal(H,0)
            np.testing.assert_allclose(g,a+p*b.sum(1),atol=1e-12)
            np.testing.assert_allclose(H,b,atol=1e-12)
            est[str(p)]={'g':g.tolist(),'H':H.tolist()}
        aa,bb=budget_surrogate(est,F,2)
        np.testing.assert_allclose(aa,a,atol=1e-12);np.testing.assert_allclose(bb,b,atol=1e-12)

    def test_exact_subset_and_swap_gain(self):
        rng=np.random.default_rng(2);F=7;k=3;a=rng.normal(size=F);b=rng.normal(size=(F,F));b=(b+b.T)/2;np.fill_diagonal(b,0)
        masks=[to_mask(c) for c in itertools.combinations(range(F),k)]
        best=max(quad_value(mask_vec(m,F)[None],a,b)[0] for m in masks)
        chosen=select_subsets(a,b,F,k,top=1)[0]
        self.assertAlmostEqual(quad_value(mask_vec(chosen,F)[None],a,b)[0],best)
        x=mask_vec(chosen,F);s=b@x
        for i in bits(chosen,F):
            for j in set(range(F))-set(bits(chosen,F)):
                y=x.copy();y[i]=0;y[j]=1
                self.assertAlmostEqual(quad_value(y[None],a,b)[0]-quad_value(x[None],a,b)[0],a[j]+s[j]-b[i,j]-a[i]-s[i])

    def test_paper_fixture_scope(self):
        files=list((R/'figures/results').glob('*.json'));self.assertEqual(len(files),210)
        for path in files:
            j=json.loads(path.read_text())
            if j.get('kind')=='run':
                self.assertIn('MOSAIC',j['evals'])
                from mosaic_lib import RUNG_FULL,RUNG_PAIR,RUNG_LOCAL,RUNG_V1,RUNG_AGNOSTIC
                self.assertLessEqual(set(j['evals']),{RUNG_FULL,RUNG_PAIR,RUNG_LOCAL,RUNG_V1,RUNG_AGNOSTIC,'Permutation'})
                for k,fields in j['evals']['MOSAIC']['subsets'].items():self.assertEqual(len(fields),int(k))
            if j.get('kind')=='sens':
                self.assertEqual(set(j['variants']),{'default','sigma=0.1','sigma=0.5','sigma=1.0','candidates=1','candidates=5','rows=0.5'})

    def test_serving_entrypoint_arguments(self):
        import shlex
        output=subprocess.check_output([sys.executable,'scripts/reproduce.py','serving','--dataset','ml1m20','--dry'],cwd=R,text=True)
        tokens=shlex.split(output.replace('\\','/'))
        self.assertEqual(tokens[tokens.index('--name')+1],'ml1m20')
        self.assertEqual(tokens[tokens.index('--budgets')+1],'2,4,6,8,10,12')


if __name__=='__main__':unittest.main()
