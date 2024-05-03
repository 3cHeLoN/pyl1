"""Unit tests."""

from unittest import TestCase
import numpy as np
from pyl1.operators import OpTV2D, OpTV3D


class TvTest(TestCase):
    """Test the tv operator."""

    def test_tv_compare_with_actual_transpose3d_same(self):
        """Test 3D tv operator."""
        tv_operator = OpTV3D(8, 10, 15)

        tv_operator_sub = tv_operator * np.eye(tv_operator.shape[1])
        tv_operator_transposed_sub = tv_operator.T * np.eye(tv_operator.shape[0])

        # check equal
        self.assertTrue(np.array_equal(tv_operator_sub.T, tv_operator_transposed_sub))
        self.assertTrue(np.allclose(tv_operator_sub.T, tv_operator_transposed_sub))
        self.assertTrue(np.allclose(tv_operator_transposed_sub.T, tv_operator_sub))

    def test_tv_compare_with_actual_transpose2d_same(self):
        """Test 2D tv operator."""
        tv_operator = OpTV2D(6, 9)

        tv_operator_sub = tv_operator * np.eye(tv_operator.shape[1])
        tv_operator_transposed_sub = tv_operator.T * np.eye(tv_operator.shape[0])

        # check equal
        self.assertTrue(np.array_equal(tv_operator_sub.T, tv_operator_transposed_sub))
        self.assertTrue(np.allclose(tv_operator_sub.T, tv_operator_transposed_sub))
        self.assertTrue(np.allclose(tv_operator_transposed_sub.T, tv_operator_sub))
