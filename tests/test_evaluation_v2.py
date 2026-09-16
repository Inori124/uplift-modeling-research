import sys
import unittest
from pathlib import Path
from itertools import permutations
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation_v2 import evaluate, paired_bootstrap, _prepare, _calculate

class EvaluationTests(unittest.TestCase):
    def test_hand_calculation(self):
        r = evaluate([.9,.8,.1,0], [1,0,1,0], [1,0,0,0], p=.5)
        m = r['metrics']
        self.assertAlmostEqual(m['overall_ate'], .5)
        self.assertAlmostEqual(m['auuc'], .4375)
        self.assertAlmostEqual(m['qini_area'], .1875)
        self.assertAlmostEqual(m['top20_policy_gain'], .4)
        self.assertAlmostEqual(m['top20_selected_ate'], 2.)  # finite-sample HT can exceed 1
        self.assertAlmostEqual(m['top20_gain_over_random'], .3)
        self.assertAlmostEqual(r['random_expectation']['top20_policy_gain'], .1)
        self.assertAlmostEqual(r['curve']['gain'][-1], .5)

    def test_ties_order_and_random_expectation(self):
        s=np.ones(4); t=np.array([1,0,1,0]); y=np.array([1,0,0,0])
        r = evaluate(s,t,y)
        for order in permutations(range(4)):
            a=np.array(order); r2=evaluate(s[a],t[a],y[a])
            self.assertEqual(r['metrics'],r2['metrics'])
        self.assertAlmostEqual(r['metrics']['qini_area'],0.)
        self.assertAlmostEqual(r['metrics']['top20_policy_gain'],.1)

    def test_exact_area_not_plot_grid(self):
        n=103; s=np.arange(n)[::-1];t=np.arange(n)%2;y=np.zeros(n); y[1]=1
        r=evaluate(s,t,y,p=.5)
        z=t*y/.5-(1-t)*y/.5
        g=np.r_[0,np.cumsum(z)/n];q=np.arange(n+1)/n
        expected=np.sum(np.diff(q)*(g[:-1]+g[1:])/2)
        self.assertAlmostEqual(r['metrics']['auuc'],expected,15)

    def test_zero_outcome_null(self):
        r=evaluate([.9,.8,.2,.1],[1,0,1,0],[0,0,0,0])
        self.assertEqual(r['metrics']['qini_area'],0.)

    def test_null_expectation_over_assignments(self):
        # Finite randomization realizations need not each have Qini exactly zero.
        scores=np.array([4,3,2,1]);y=np.array([1,0,1,0]); vals=[]
        for t in set(permutations([0,0,1,1])):
            vals.append(evaluate(scores,t,y,p=.5)['metrics']['qini_area'])
        self.assertAlmostEqual(np.mean(vals),0.)

    def test_invalid_before_integer_cast(self):
        for t,y in [([0,.4],[0,1]),([0,1],[0,.3]),([0,1],[0,np.nan]),([1,1],[0,1])]:
            with self.assertRaises(ValueError):evaluate([1,2],t,y)
        with self.assertRaises(ValueError):evaluate([1,np.nan],[0,1],[0,1])
        with self.assertRaises(ValueError):evaluate([1,2],[0,1],[0,1],p=1)

    def test_weighted_bootstrap_matches_explicit_repeated_rows(self):
        s=np.array([2,2,1,0.]);t=np.array([0,1,0,1]);y=np.array([1,1,0,1]);w=np.array([0,2,1,1])
        z=t*y/.5-(1-t)*y/.5
        actual=_calculate(_prepare(s,z),w)['metrics']
        ix=np.repeat(np.arange(4),w)
        expected=evaluate(s[ix],t[ix],y[ix],p=.5)['metrics']
        for k in expected:self.assertAlmostEqual(actual[k],expected[k])

    def test_paired_bootstrap(self):
        s=np.array([.9,.8,.2,.1]);t=np.array([1,0,1,0]);y=np.array([1,0,0,0])
        r=paired_bootstrap({'a':s,'b':s.copy()},t,y,n_bootstrap=20)
        for ci in r['pairwise_difference_ci']['a-b'].values():self.assertEqual(ci,[0.,0.])
        self.assertEqual(r,paired_bootstrap({'a':s,'b':s.copy()},t,y,n_bootstrap=20))
        self.assertEqual(r['ci']['random_expectation']['qini_area'],[0.,0.])

if __name__ == '__main__':unittest.main()
