"""
Rotation matrix generation for Latent Trajectory Watermarking.

Generates deterministic orthogonal rotation matrices from a secret key.
Uses the anisotropy-aware orthogonal injection method to reduce false positives.
"""

import hashlib
from typing import Optional, Tuple
import numpy as np
from scipy.stats import special_ortho_group
from scipy.linalg import qr


def generate_rotation_matrix(
    secret_key: str,
    dim: int,
    rotation_strength: float = 0.3
) -> np.ndarray:
    """
    Generate a deterministic rotation matrix from a secret key.
    
    The rotation is designed to be orthogonal to the natural semantic flow
    to minimize false positives from human text that naturally follows
    coherent semantic trajectories.
    
    Args:
        secret_key: Secret key string for deterministic generation
        dim: Dimension of the embedding space
        rotation_strength: How much to rotate (0=no rotation, 1=full rotation)
                          Lower values reduce text quality impact.
    
    Returns:
        Orthogonal rotation matrix of shape (dim, dim)
    """
    # Create deterministic seed from key
    key_hash = hashlib.sha256(secret_key.encode()).digest()
    seed = int.from_bytes(key_hash[:4], byteorder='big')
    rng = np.random.RandomState(seed)
    
    # Generate base random orthogonal matrix using QR decomposition
    random_matrix = rng.randn(dim, dim)
    Q, R = qr(random_matrix)
    
    # Ensure proper rotation (det = 1, not -1)
    d = np.diag(R)
    ph = np.sign(d)
    Q = Q @ np.diag(ph)
    
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    # Interpolate between identity and full rotation based on strength
    identity = np.eye(dim)
    rotation_matrix = identity + rotation_strength * (Q - identity)
    
    # Re-orthogonalize after interpolation
    rotation_matrix, R2 = qr(rotation_matrix)
    
    # Ensure det = 1 again after re-orthogonalization
    d2 = np.diag(R2)
    ph2 = np.sign(d2)
    rotation_matrix = rotation_matrix @ np.diag(ph2)
    
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] = -rotation_matrix[:, 0]
    
    return rotation_matrix


class OrthogonalRotation:
    """
    Manages orthogonal rotations in embedding space for watermarking.
    
    This class handles the core mathematical operations needed for LTW:
    - Rotation of embedding vectors
    - Verification of rotation alignment
    - Anisotropy-aware adjustments
    
    Attributes:
        secret_key: The secret key used for rotation generation
        dim: Dimension of the embedding space
        rotation_matrix: The orthogonal rotation matrix
    """
    
    def __init__(
        self,
        secret_key: str,
        dim: int,
        rotation_strength: float = 0.3,
        use_anisotropy_correction: bool = True
    ):
        """
        Initialize the orthogonal rotation.
        
        Args:
            secret_key: Secret key for deterministic rotation
            dim: Embedding dimension
            rotation_strength: Rotation intensity (0-1)
            use_anisotropy_correction: Whether to apply anisotropy correction
        """
        self.secret_key = secret_key
        self.dim = dim
        self.rotation_strength = rotation_strength
        self.use_anisotropy_correction = use_anisotropy_correction
        
        # Generate the rotation matrix
        self.rotation_matrix = generate_rotation_matrix(
            secret_key, dim, rotation_strength
        )
        
        # Precompute inverse for detection
        self.inverse_matrix = self.rotation_matrix.T  # Orthogonal: inverse = transpose
        
        # For anisotropy correction, we'll store principal directions
        self._principal_directions: Optional[np.ndarray] = None
        self._mean_embedding: Optional[np.ndarray] = None
    
    def fit_anisotropy(self, embeddings: np.ndarray):
        """
        Fit anisotropy correction based on sample embeddings.
        
        This helps ensure rotations are orthogonal to the natural
        semantic flow in the embedding space.
        
        Args:
            embeddings: Sample embeddings of shape (n_samples, dim)
        """
        if not self.use_anisotropy_correction:
            return
        
        # Compute mean and center embeddings
        self._mean_embedding = np.mean(embeddings, axis=0)
        centered = embeddings - self._mean_embedding
        
        # PCA to find principal directions
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self._principal_directions = eigenvectors[:, idx]
    
    def rotate(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply the secret rotation to an embedding.
        
        Args:
            embedding: Input embedding vector of shape (dim,)
            
        Returns:
            Rotated embedding of shape (dim,)
        """
        return self.rotation_matrix @ embedding
    
    def inverse_rotate(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply inverse rotation.
        
        Args:
            embedding: Rotated embedding vector
            
        Returns:
            Original embedding (approximately)
        """
        return self.inverse_matrix @ embedding
    
    def compute_alignment_score(
        self,
        source_embedding: np.ndarray,
        target_embedding: np.ndarray
    ) -> float:
        """
        Compute how well the target aligns with the rotated source.
        
        This is the core detection metric: if text is watermarked,
        consecutive embeddings should follow the rotation pattern.
        
        Args:
            source_embedding: The source embedding (e.g., previous word)
            target_embedding: The target embedding (e.g., current word)
            
        Returns:
            Alignment score (higher = more likely watermarked)
        """
        # Rotate the source
        rotated_source = self.rotate(source_embedding)
        
        # Normalize both
        rotated_norm = rotated_source / (np.linalg.norm(rotated_source) + 1e-8)
        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
        
        # Cosine similarity with rotation
        alignment = np.dot(rotated_norm, target_norm)
        
        return float(alignment)
    
    def compute_differential_score(
        self,
        source_embedding: np.ndarray,
        target_embedding: np.ndarray
    ) -> float:
        """
        Compute differential score to reduce false positives.
        
        Instead of just checking watermark alignment, we compare:
        - Alignment with watermark rotation
        - Natural alignment (semantic coherence)
        
        We only flag as watermarked if the rotation alignment is
        significantly better than natural coherence.
        
        Args:
            source_embedding: Source embedding
            target_embedding: Target embedding
            
        Returns:
            Differential score (positive = likely watermarked)
        """
        # Watermark alignment
        watermark_alignment = self.compute_alignment_score(
            source_embedding, target_embedding
        )
        
        # Natural coherence (direct similarity without rotation)
        source_norm = source_embedding / (np.linalg.norm(source_embedding) + 1e-8)
        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
        natural_coherence = np.dot(source_norm, target_norm)
        
        # Differential: watermark should align better than natural flow
        differential = watermark_alignment - natural_coherence
        
        return float(differential)
    
    def get_rotation_angle(self) -> float:
        """
        Get the effective rotation angle in radians.
        
        Returns:
            Rotation angle (average across dimensions)
        """
        # Trace of rotation matrix relates to rotation angle
        trace = np.trace(self.rotation_matrix)
        # For n-dimensional rotation: trace = (n-2) + 2*cos(theta)
        cos_theta = (trace - self.dim + 2) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        return float(np.arccos(cos_theta))
    
    def verify_orthogonality(self, tolerance: float = 1e-6) -> bool:
        """
        Verify that the rotation matrix is orthogonal.
        
        Args:
            tolerance: Numerical tolerance for orthogonality check
            
        Returns:
            True if matrix is orthogonal within tolerance
        """
        product = self.rotation_matrix @ self.rotation_matrix.T
        identity = np.eye(self.dim)
        error = np.max(np.abs(product - identity))
        return error < tolerance
