import pytest
import numpy as np
from pyl1.operators import OpTV2D, OpTV3D


def test_optv3d():
    """Test 3D tv operator."""
    TV = OpTV3D(8, 10, 15)
    
    TVd = TV * np.eye(TV.shape[1])
    TVTd = TV.T * np.eye(TV.shape[0])
    
    # check equal
    assert np.array_equal(TVd.T, TVTd)
    assert np.allclose(TVd.T, TVTd)
    assert np.allclose(TVTd.T, TVd)


def test_optv2d():
    """Test 2D tv operator."""
    TV = OpTV2D(6, 9)

    TVd = TV * np.eye(TV.shape[1])
    TVTd = TV.T * np.eye(TV.shape[0])
    
    # check equal
    assert np.array_equal(TVd.T, TVTd)
    assert np.allclose(TVd.T, TVTd)
    assert np.allclose(TVTd.T, TVd)
