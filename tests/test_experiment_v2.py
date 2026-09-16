import sys
from pathlib import Path
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_v2_experiment import sample_training, fit_scores, validate_training

class ExperimentTests(unittest.TestCase):
    def test_fixed_size_sampling_ignores_outcomes(self):
        t=np.r_[np.zeros(200,dtype=int),np.ones(800,dtype=int)];pool=np.arange(900)
        for f in (.85,.5,1/3,.2):
            ix=sample_training(pool,t,120,f,2027)
            self.assertEqual(len(ix),120)
            self.assertEqual(len(set(ix)),120)
            self.assertEqual(t[ix].sum(),round(120*f))
            np.testing.assert_array_equal(ix,sample_training(pool,t,120,f,2027))
            self.assertFalse((ix>=900).any())
        with self.assertRaises(ValueError):sample_training(pool,t,700,.2,2027)
        with self.assertRaises(ValueError):sample_training(pool,t,None,.2,2027)

    def test_effect_models_have_finite_predictions_and_oof_coverage(self):
        rng=np.random.default_rng(71);X=rng.normal(size=(1200,4));t=rng.integers(0,2,1200)
        prob=.1+.1*(X[:,0]>0)+.2*t*(X[:,1]>0)
        y=rng.binomial(1,prob)
        with threadpool_limits(1):scores,probs,audit=fit_scores(X[:1000],t[:1000],y[:1000],X[1000:],19)
        self.assertEqual(sum(a['heldout_n'] for a in audit),1000)
        for s in scores.values():self.assertTrue(np.isfinite(s).all());self.assertEqual(s.shape,(200,))
        self.assertEqual(len(probs),2)
        with self.assertRaises(ValueError):validate_training([0,1],np.array([0,0]))

if __name__=='__main__':unittest.main()
