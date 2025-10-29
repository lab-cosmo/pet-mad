import numpy as np
from ase.build import bulk

from upet.explore import PETMADFeaturizer


def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    featurizer = PETMADFeaturizer("latest")
    feats = featurizer([atoms], None)
    assert isinstance(feats, np.ndarray)
