using LinearAlgebra
using ProgressMeter # To replace `parfor_progress`
using Arpack
using Test

function smallest_eigs(X::AbstractArray{T}, k::Integer, mode::String="LinearAlgebra") where {T<:Number}
     # Ensure X is a square matrix
    if size(X, 1) != size(X, 2)
        error("Input matrix must be square.")
    end
    
    if mode == "Arpack"
        try
            smallest_λs, smallest_V = Arpack.eigs(X, nev=k, tol=1e-6, maxiter=500, which=:SM)
            return smallest_λs, smallest_V
        catch e
            @warn "Arpack failure. Proceeding with LinearAlgebra.eigen. Note that performance may be suboptimal"
        end
    end
    
        # Compute the eigenvalues
    λs, V = LinearAlgebra.eigen(X)

    # Sort the eigenvalues and select the k smallest
    sorted_λs_perm = sortperm(abs.(λs))
    smallest_λs = λs[sorted_λs_perm][1:k]
    smallest_Vs = V[:, sorted_λs_perm][:, 1:k]

    return smallest_λs, smallest_Vs
end

function timeDelayEmbed(X::AbstractMatrix{T}, τ::Integer, d::Integer) where {T<:Number}
    # Ensure X is a matrix and τ and d are positive integers
    if !isa(X, AbstractMatrix) || τ <= 0 || d <= 0
        error("Invalid input. X must be a matrix, and τ and d must be positive integers.")
    end

    # Ensure X has enough rows for the embedding
    if size(X, 1) < τ * (d - 1) + 1
        error("X does not have enough rows for the specified embedding.")
    end

    # Initialize the embedded matrix
    X_embedded = zeros(T, size(X, 1) - τ * (d - 1), d * size(X, 2))
    for i = 1:size(X, 2)
        for j = 1:d
            X_embedded[:, (j - 1) * size(X, 2) + i] = X[τ * (j - 1) + 1:end - τ * (d - j), i]
        end
    end

    return X_embedded
end

# Note: Julia's equivalent of MATLAB's inputParser and varargin
# is typically handled using keyword arguments in function definitions.

"""
    KoopPseudoSpecQR(PX, PY, W, z_pts; kwargs...)

Computes the pseudospectrum of K (currently written for dense matrices).

# Arguments
- `PX`: dictionary evaluated at snapshots
- `PY`: dictionary evaluated at snapshots one time step later
- `W`: vector of weights for the quadrature
- `z_pts`: vector of complex points where we want to compute pseudospectra

# Keyword Arguments
- `Parallel::String="off"`: Use parallel processing (`"on"`) or not (`"off"`).
- `z_pts2::Vector{<:Number}=[]`: Vector of complex points where we want to compute pseudoeigenfunctions.
- `reg_param::Real=1e-14`: Regularisation parameter for G.

# Returns
- `RES`: Residual for shifts `z_pts`.
- `RES2`: Residual for pseudoeigenfunctions corresponding to shifts `z_pts2`.
- `V2`: Pseudoeigenfunctions corresponding to shifts `z_pts2`.
"""
function KoopPseudoSpecQR(
    PX::AbstractMatrix,
    PY::AbstractMatrix,
    W::AbstractVector,
    z_pts::AbstractVector{<:Number};
    z_pts2::AbstractVector{<:Number}=[],
    reg_param::Real=1e-14,
    progress::Bool=true,
    mode::String="LinearAlgebra"
)


    ## compute the pseudospectrum
    W_vec = vec(W)::Vector{eltype(W)} # Ensure W is a column vector
    # In Julia, element-wise multiplication is .*
    # `qr` by default computes the "thin" QR decomposition similar to MATLAB's "econ"
    Q, R = qr(sqrt.(W_vec) .* PX)
    Q = Matrix{eltype(Q)}(Q)

    C1 = (sqrt.(W_vec) .* PY) / R
    L = C1' * C1
    G = Matrix{eltype(PX)}(I, size(PX, 2), size(PX, 2)) # Identity matrix with correct type
    A = Q' * C1

    z_pts_vec = vec(z_pts) # Ensure z_pts is a column vector
    LL = length(z_pts_vec)
    RES = zeros(eltype(W), LL) # Julia uses 0-based indexing

    if LL > 0
        # Warning suppression is generally discouraged in Julia.
        # If specific warnings are problematic, it's better to address their root cause
        # or filter specific warning types if absolutely necessary.
        # The `eigs` function itself might throw warnings for convergence issues.

        # Julia equivalent of parfor_progress
        if progress
            p = Progress(LL, 1, "Computing pseudospectrum...")
        end

        # Julia's equivalent to parfor is achieved using distributed computing
        # or multithreading. For a direct translation, we can use Threads.@threads
        # for a simple loop parallelization if Julia is started with multiple threads.
        # Requires starting Julia with `julia -t auto` or similar.
        # A more robust parallelization might use Distributed.jl.
        # This example uses Threads.@threads.

        # Ensure Julia is started with multiple threads for Threads.@threads to have effect
        if Threads.nthreads() == 1
            @warn "Parallel processing possible, but Julia is running on a single thread. Start Julia with `julia -t auto` or similar for parallel execution."
        end
        local_num_threads = LinearAlgebra.BLAS.get_num_threads()
        LinearAlgebra.BLAS.set_num_threads(1) # Ensure single-threaded for eigs

        Threads.@threads for jj = 1:LL
            # Julia's `eigs` (from Arpack.jl) is similar to MATLAB's `eigs`.
            # The 'smallestabs' option is available.
            # `opnorm` in Julia computes the 2-norm by default, which is related
            # to the largest singular value, not the smallest eigenvalue of A - zI.
            # The residual definition in the MATLAB code looks like sqrt(smallest eigenvalue of H(z)).
            # So we need the smallest magnitude eigenvalue of the Hermitian matrix H(z).
            # The `eigs` function with `which=:SM` or `eigsolve` are appropriate.
            # `eigs` from Arpack.jl is a good match for MATLAB's `eigs`.
            # It returns eigenvalues and eigenvectors. We want the smallest magnitude eigenvalue.

            # Construct the matrix for eigenvalue calculation
            Hz = L - z_pts_vec[jj] * A' - conj(z_pts_vec[jj]) * A + abs(z_pts_vec[jj])^2 * G

            # Compute the smallest magnitude eigenvalue using `eigs`
            # `which=:SM` specifies smallest magnitude
            # `nev=1` asks for 1 eigenvalue
            # try
                λs, _ = smallest_eigs(Hz, 1, mode)
                # MATLAB's real(eigs(...)) takes the real part of the complex eigenvalue.
                # For a Hermitian matrix, eigenvalues should be real in theory, but
                # numerical errors can introduce small imaginary parts. Taking real()
                # is reasonable here.
                RES[jj] = sqrt(max(0, real(λs[1])))
            # catch e
                # Handle potential convergence issues from eigs
                # @warn "eigs failed to converge for z = $(z_pts_vec[jj]): $e"
                # RES[jj] = NaN # Or some other indicator of failure
            # end
            if progress
                next!(p) # Increment progress bar
            end
        end
        if progress
            finish!(p) # Finish progress bar
        end
    end
    LinearAlgebra.BLAS.set_num_threads(local_num_threads) # Restore original number of threads

    RES2 = []
    T = eltype(G) <: Complex ? eltype(G) : Complex{eltype(G)}
    V2 = Matrix{T}(undef, size(G, 1), 0) # Initialize as an empty matrix with correct type

    if !isempty(z_pts2)
        RES2 = zeros(length(z_pts2))
        V2 = zeros(Complex{eltype(G)}, size(G, 1), length(z_pts2)) # Preallocate V2

        if progress
            p2 = Progress(length(z_pts2), 1, "Computing pseudoeigenfunctions...")
        end

        if Threads.nthreads() == 1
            @warn "Parallel processing possible, but Julia is running on a single thread. Start Julia with `julia -t auto` or similar for parallel execution."
        end

        LinearAlgebra.BLAS.set_num_threads(1)
        Threads.@threads for jj = 1:length(z_pts2)
            Hz2 = L - z_pts2[jj] * A' - conj(z_pts2[jj]) * A + abs(z_pts2[jj])^2 * G
                # try
                # Get both eigenvalue and eigenvector
                λs, Vs = smallest_eigs(Hz2, 1, mode)
                V2[:, jj] = Vs[:, 1] # Store the eigenvector
                RES2[jj] = sqrt(max(0, real(λs[1]))) # Store the residual
                # catch e
                # @warn "eigs failed to converge for z2 = $(z_pts2[jj]): $e"
                # V2[:, jj] .= NaN # Indicate failure for eigenvector
                # RES2[jj] = NaN # Indicate failure for residual
                # end
                if progress
                    next!(p2) # Increment progress bar
                end
        end
        if progress
            finish!(p2) # Finish progress bar
        end

        # Final transformation of V2
        V2 = R \ V2 # Julia's backslash operator for solving linear systems
    end
    LinearAlgebra.BLAS.set_num_threads(local_num_threads) # Restore original number of threads

    return RES, RES2, V2
end


# PX = rand(100, 10)
# PY = rand(100, 10)
# W = rand(100)
# z_pts = [0.5 + 0.1im, 0.6 + 0.2im]
# z_pts2 = [0.7 + 0.3im]
# reg_param = 1e-14

# RES, RES2, V2 = KoopPseudoSpecQR(PX, PY, W, z_pts, z_pts2=z_pts2, reg_param=reg_param)
# println("RES: ", RES)

# RES_par, RES2_par, V2_par = KoopPseudoSpecQR(PX, PY, W, z_pts, z_pts2=z_pts2, reg_param=reg_param)
# println("RES_par: ", RES_par)
# println("RES2_par: ", RES2_par)
# println("V2_par: ", V2_par)