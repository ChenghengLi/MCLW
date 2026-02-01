"""
Unit tests for the rotation matrix module.
"""

import pytest
import numpy as np
from ltw_watermark.rotation import generate_rotation_matrix, OrthogonalRotation


class TestRotationMatrix:
    """Tests for rotation matrix generation."""
    
    def test_matrix_is_orthogonal(self):
        """Generated matrix should be orthogonal (R @ R.T = I)."""
        R = generate_rotation_matrix("test-key", dim=384)
        product = R @ R.T
        identity = np.eye(384)
        assert np.allclose(product, identity, atol=1e-6)
    
    def test_deterministic_from_key(self):
        """Same key should produce same matrix."""
        R1 = generate_rotation_matrix("my-secret", dim=100)
        R2 = generate_rotation_matrix("my-secret", dim=100)
        assert np.allclose(R1, R2)
    
    def test_different_keys_different_matrices(self):
        """Different keys should produce different matrices."""
        R1 = generate_rotation_matrix("key-1", dim=100)
        R2 = generate_rotation_matrix("key-2", dim=100)
        assert not np.allclose(R1, R2)
    
    def test_rotation_strength_effect(self):
        """Different strengths should produce different matrices."""
        R_low = generate_rotation_matrix("key", dim=100, rotation_strength=0.1)
        R_high = generate_rotation_matrix("key", dim=100, rotation_strength=0.9)
        
        # Different strengths should produce different matrices
        assert not np.allclose(R_low, R_high)
    
    def test_determinant_is_one(self):
        """Rotation matrix should have determinant 1 (proper rotation)."""
        R = generate_rotation_matrix("key", dim=50)
        det = np.linalg.det(R)
        assert np.isclose(abs(det), 1.0, atol=1e-5)


class TestOrthogonalRotation:
    """Tests for OrthogonalRotation class."""
    
    @pytest.fixture
    def rotation(self):
        return OrthogonalRotation(
            secret_key="test-secret-key",
            dim=384,
            rotation_strength=0.3
        )
    
    def test_initialization(self, rotation):
        """Rotation should initialize properly."""
        assert rotation.dim == 384
        assert rotation.rotation_matrix.shape == (384, 384)
        assert rotation.verify_orthogonality()
    
    def test_rotate_preserves_norm(self, rotation):
        """Rotation should preserve vector norm."""
        v = np.random.randn(384)
        v_rotated = rotation.rotate(v)
        
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rotated), rtol=1e-5)
    
    def test_inverse_rotation(self, rotation):
        """Inverse rotation should recover original vector."""
        v = np.random.randn(384)
        v_rotated = rotation.rotate(v)
        v_recovered = rotation.inverse_rotate(v_rotated)
        
        assert np.allclose(v, v_recovered, atol=1e-6)
    
    def test_alignment_score_range(self, rotation):
        """Alignment score should be in [-1, 1]."""
        v1 = np.random.randn(384)
        v2 = np.random.randn(384)
        
        score = rotation.compute_alignment_score(v1, v2)
        assert -1 <= score <= 1
    
    def test_alignment_with_rotated_self(self, rotation):
        """Vector should have high alignment with its rotation."""
        v = np.random.randn(384)
        v_rotated = rotation.rotate(v)
        
        score = rotation.compute_alignment_score(v, v_rotated)
        # Should be close to 1 for perfect alignment
        assert score > 0.9
    
    def test_differential_score_for_random_vectors(self, rotation):
        """Random vectors should have differential score near 0."""
        scores = []
        for _ in range(100):
            v1 = np.random.randn(384)
            v2 = np.random.randn(384)
            score = rotation.compute_differential_score(v1, v2)
            scores.append(score)
        
        # Mean should be close to 0 for random vectors
        assert abs(np.mean(scores)) < 0.2
    
    def test_rotation_angle_positive(self, rotation):
        """Rotation angle should be positive."""
        angle = rotation.get_rotation_angle()
        assert angle > 0


class TestDifferentDimensions:
    """Test rotation with different embedding dimensions."""
    
    @pytest.mark.parametrize("dim", [128, 384, 512, 768, 1024])
    def test_various_dimensions(self, dim):
        """Rotation should work for common embedding dimensions."""
        rotation = OrthogonalRotation("key", dim=dim, rotation_strength=0.3)
        
        assert rotation.verify_orthogonality()
        
        v = np.random.randn(dim)
        v_rotated = rotation.rotate(v)
        assert v_rotated.shape == (dim,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
