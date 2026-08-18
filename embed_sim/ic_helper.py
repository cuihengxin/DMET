import numpy as np


def same_col_space(A, B, atol=1e-10):
    """
    Check whether A and B span the same column space (under the Euclidean metric).
    Returns (is_same, sing_vals), where sing_vals are the singular values of Q_A^T Q_B.
    """
    # Orthonormal bases (only the full-rank column part is retained)
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)

    # Keep only the effective columns
    # to avoid extra zero columns caused by numerical rank issues
    rA = np.linalg.matrix_rank(A, tol=atol)
    rB = np.linalg.matrix_rank(B, tol=atol)
    QA = QA[:, :rA]
    QB = QB[:, :rB]

    # If the dimensions are different, the column spaces cannot be the same
    if rA != rB:
        return False, np.array([])

    # Principal angles: all singular values should be approximately 1
    s = np.linalg.svd(QA.T.conj() @ QB, compute_uv=False)
    is_same = np.allclose(s, np.ones_like(s), atol=10*atol)
    return is_same, s


def gram_schmidt_new(B, A, tol=1e-12):
    """
    Remove from the columns of B all components lying in the space spanned by
    the columns of A, and return B' that is orthogonal to A
    (while keeping the number of columns unchanged).

    Parameters:
        B : ndarray
            n×m matrix, the set of column vectors to be orthogonalized
        A : ndarray
            n×k matrix defining the column space to be removed
        tol : float
            Numerical tolerance threshold (default: 1e-12)

    Returns:
        B_proj : ndarray
            Matrix obtained after removing the A-space components
    """

    # Compute the Gram matrix of A (k×k)
    G_A = A.T.conj() @ A

    # Regularize A once for numerical stability
    try:
        G_A_inv = np.linalg.inv(G_A)
    except np.linalg.LinAlgError:
        # If it is singular, use the pseudoinverse instead
        G_A_inv = np.linalg.pinv(G_A)
    
    # Compute the projection of B onto the space of A
    # P_A = A (A^T A)^{-1} A^T
    # B_proj = B - P_A B
    P_A = A @ G_A_inv @ A.T.conj()
    B_proj = B - P_A @ B

    # For numerical stability, one may optionally apply QR orthogonalization again
    # while keeping the number of columns unchanged
    # B_proj, _ = np.linalg.qr(B_proj)

    return B_proj


def ic_orthogonalization(S, imp_idx, mol):
    # Step 1: Initialize important variables
    eigenvalues, U = np.linalg.eigh(S)
    s_minus_1_2 = np.diag(1.0 / np.sqrt(eigenvalues))
    S_minus_1_2 = U @ s_minus_1_2 @ U.T.conj()
    X = S_minus_1_2
    X_minus_1 = np.linalg.inv(X)
    ao2lo_old = X
    lo2ao_old = X_minus_1
    
    # Identify the indices for the orbitals of interest (Oimp and Oenv)
    Oimp_indices = imp_idx
    Oenv_indices = np.array([i for i in range(S.shape[0]) if i not in Oimp_indices])
    
    # Split the matrix into A and B parts
    lo2ao_B = lo2ao_old[:, Oenv_indices]
    lo2ao_A = lo2ao_old[:, Oimp_indices]
    
    # Step 2: Orthogonalize the A part (LO-A) and construct related matrices
    S_AA = lo2ao_A.T.conj() @ lo2ao_A
    eigenvalues, U = np.linalg.eigh(S_AA)
    s_AA_minus_1_2 = np.diag(1.0 / np.sqrt(eigenvalues))
    S_AA_minus_1_2 = U @ s_AA_minus_1_2 @ U.T.conj()
    X_AA = S_AA_minus_1_2
    X_AA_minus_1 = np.linalg.inv(X_AA)
    iclo_A2ao_A = X_AA_minus_1
    ao_A2iclo_A = X_AA
    lo2iclo_A = lo2ao_A @ ao_A2iclo_A
    iclo_A2lo = lo2iclo_A.T.conj()
    
    # Step 3: Project B part (LO-B) and orthogonalize with Gram-Schmidt
    lo2projected_ao_B = gram_schmidt_new(lo2ao_B, lo2ao_A)
    S_pBB = lo2projected_ao_B.T.conj() @ lo2projected_ao_B
    eigenvalues, U = np.linalg.eigh(S_pBB)
    s_pBB_minus_1_2 = np.diag(1.0 / np.sqrt(eigenvalues))
    S_pBB_minus_1_2 = U @ s_pBB_minus_1_2 @ U.T.conj()
    X_pBB = S_pBB_minus_1_2
    X_pBB_minus_1 = np.linalg.inv(X_pBB)
    projected_ao_B2iclo_B = X_pBB
    lo2iclo_B = lo2projected_ao_B @ projected_ao_B2iclo_B
    iclo_B2lo = lo2iclo_B.T.conj()

    # Step 4: Merge A and B to create the final ICLO-to-LO transformation
    iclo_2lo = np.vstack((iclo_A2lo, iclo_B2lo))
    
    # Step 5: Construct the 1DM in the IC orthogonal basis
    iclo2ao = iclo_2lo @ lo2ao_old
    
    # Return the transformation matrices
    lo2ao = iclo2ao
    ao2lo = np.linalg.inv(lo2ao)
    
    ok, svd = same_col_space(lo2iclo_A, lo2ao_A)
    print("Is the column space of AO_A the same as that of ICLO_A", ok)

    return ao2lo

