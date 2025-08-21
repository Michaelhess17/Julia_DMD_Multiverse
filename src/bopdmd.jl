using LinearAlgebra
using Plots
using Distributions, Random, Statistics
using DataFrames, DifferentialEquations, DelayEmbeddings

function lotka_volterra(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

function delayEmbed(X::AbstractMatrix{<:Number}, τ::Int, d::Int)
    n, m = size(X)
    if n == 0 || m == 0
        error("Input matrix X to delayEmbed is empty.")
    end
    num_delay_rows = n - (d - 1) * τ
    if num_delay_rows <= 0
        error("Cannot delay embed: result would have non-positive rows. n=$n, d=$d, τ=$τ")
    end
    X_delay = zeros(eltype(X), num_delay_rows, m * d)
    Threads.@threads for i in 1:num_delay_rows
        for j in 1:d
            start_row_idx = i + (j - 1) * τ
            start_col_idx = (j - 1) * m + 1
            end_col_idx = j * m
            # Ensure indices are within bounds
            if start_row_idx > n
                 error("Index out of bounds during delay embedding: start_row_idx=$start_row_idx > n=$n")
            end
            X_delay[i, start_col_idx:end_col_idx] = X[start_row_idx, :]
        end
    end
    return X_delay
end


# X IS (features x time)

function _svht(sigma_svd::AbstractArray{<:AbstractFloat, 1}, rows::Integer, cols::Integer)
    """
    Singular Value Hard Threshold.

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    https://ieeexplore.ieee.org/document/6846297
    """
    beta = sort([rows, cols])[1] / sort([rows, cols])[2]
    omega = 0.56 * beta^3 - 0.95 * beta^2 + 1.82 * beta + 1.43
    tau = median(sigma_svd) .* omega
    rank = sum(sigma_svd .> tau)
     rank, tau, sigma_svd

    return rank
end

function _compute_rank(sigma_svd::AbstractArray{<:AbstractFloat, 1}, rows::Integer, cols::Integer, svd_rank::Number=0)
    """
    Rank computation for the truncated Singular Value Decomposition.

    :param sigma_svd: 1D singular values of SVD.
    :type sigma_svd: np.ndarray
    :param rows: Number of rows of original matrix.
    :type rows: int
    :param cols: Number of columns of original matrix.
    :type cols: int
    :param svd_rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.
    :type svd_rank: int or float
    :return: the computed rank truncation.
    :rtype: int

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    """
    if svd_rank == 0
        rank = _svht(sigma_svd, rows, cols)
    elseif 0 < svd_rank < 1
        cumulative_energy = cumsum(sigma_svd.^2) ./ sum(sigma_svd.^2)
        rank = searchsortedfirst(cumulative_energy, svd_rank) # searchsortedfirst is more appropriate
    elseif svd_rank >= 1 && isa(svd_rank, Integer) # Check type explicitly
        rank = min(svd_rank, length(sigma_svd))
    elseif svd_rank == -1 # Handle -1 case for full rank
        rank = min(rows, cols)
    else
        # Fallback or error for invalid svd_rank types/values
        rank = min(rows, cols)
    end

    # Ensure rank is at least 1
    rank = max(1, rank)
    return rank
end

function compute_rank(X::AbstractArray{<:Number, 2}, svd_rank::Number = 0)
    """
    Rank computation for the truncated Singular Value Decomposition.

    :param X: the matrix to decompose.
    :type X: np.ndarray
    :param svd_rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.
    :type svd_rank: int or float
    :return: the computed rank truncation.
    :rtype: int

    References:
    Gavish, Matan, and David L. Donoho, The optimal hard threshold for
    singular values is, IEEE Transactions on Information Theory 60.8
    (2014): 5040-5053.
    """
    # Python version uses svd(X.T), so rows/cols are swapped relative to Julia X dims
    rows = size(X, 1) # Number of features
    cols = size(X, 2) # Number of snapshots (time)

    F = svd(X)
    U, s, V = F.U, F.S, F.Vt' # time x time, features, features x features

    return _compute_rank(s, rows, cols, svd_rank), U, s, V
end

function compute_svd(X::AbstractArray{<:Number, 2}, svd_rank::Integer = 0)
    if svd_rank == 0
        svd_rank, s, V = compute_rank(X, svd_rank)
    elseif svd_rank == -1
        svd_rank = min(size(X, 1), size(X, 2))
    end
    @assert sum(isnan.(X)) == 0 && sum(isinf.(X)) == 0
    F = svd(X)
    U, s, V = F.U, F.S, F.Vt' # features x features, features, time x features
    U, s, V = U[:, 1:svd_rank], s[1:svd_rank], V[:, 1:svd_rank]
    return U, s, V
end


function _initialize_alpha(s, V, svd_rank::Integer, t::AbstractArray{<:Number, 1})
    """ Initialize alpha using Exact DMD on projected data. X is features x time. """
        ux = diagm(s[1:svd_rank]) * V[:, 1:svd_rank]'
        
        ux1 = ux[:, 1:end-1]
        ux2 = ux[:, 2:end]

        # Define the matrices Y and Z as the following and compute the rank-truncated SVD of Y.
        Y = (ux1 .+ ux2) ./ 2
        # Element-wise division by time differences. w/o large T
        Z = ((ux2 .- ux1)' ./ (t[2:end] .- t[1:end-1]))'
         size(Y), size(Z)
        U, s2, V2 = compute_svd(Y, svd_rank)
        S2 = diagm(1 ./ s2)

        # Compute the matrix Atilde and return its eigenvalues.
        # @showsize(U), size(Z), size(V2), size(S2)
        # @show S2
        Atilde = U' * Z * V2 * S2

        return eigvals(Atilde)
end

function _diff_func(eigenvalues::AbstractArray{<:Number, 1}, omega::Number, ind::Integer, absolute_diff::Bool)
    if absolute_diff
        diff = abs.(abs.(imag.(eigenvalues) .- abs.(imag.(omega))))
        diff[ind] = NaN
        return complex.(0.0, diff)
    end

    sign_val = sign(imag(omega))

    # Catch the edge case of the eigenvalues being exactly 0.
    if sign_val == 0.0
        diff = abs.(abs.(imag(eigenvalues) .- imag(omega)))
        diff[ind] = NaN
        return complex.(0.0, diff)
    end

    same_sign_index = sign(eigenvalues.imag) == sign_val
    opp_sign_eigs = copy(eigenvalues.imag)
    opp_sign_eigs[same_sign_index] = NaN
    diff = abs.(abs.(opp_sign_eigs) .- abs.(imag(omega)))
    return complex.(0.0, diff)
end

function _push_eigenvalues!(eigenvalues::AbstractArray{<:Number, 1}, eig_constraints::AbstractArray{<:AbstractString, 1}, eig_limit::Number=Inf)
    if "conjugate_pairs" in eig_constraints
        nothing
    end

    if "stable" in eig_constraints
        right_half = real.(eigenvalues) .> 0.0
        eigenvalues[right_half] = complex.(0.0, imag.(eigenvalues[right_half]))

    elseif "imag" in eig_constraints
        eigenvalues = complex.(0.0, imag.(eigenvalues))

    elseif "limited" in eig_constraints
        too_big = abs.(real.(eigenvalues)) .> eig_limit
        eigenvalues[too_big] = complex.(0.0, imag(eigenvalues[too_big]))
    end

    return eigenvalues
end

function exp_function(alpha::AbstractArray{<:Number, 1}, t::AbstractArray{<:Number, 1})
    result = exp.(t .* transpose(alpha))
    return result
end

function exp_function_deriv(alpha::AbstractArray{<:Number, 1}, t::AbstractArray{<:Number, 1}, i::Integer)
    m = length(t)
    n = length(alpha)
    if i < 1 || i > n
        error("Invalid index i=$i given to exp_function_deriv for alpha length=$n.")
    end
    A = t .* exp.(alpha[i] .* t) # Use broadcasting .*
    out = zeros(promote_type(eltype(alpha), ComplexF64), m, n)
    out[:, i] = A
    return out
end

function compute_error(X_i::AbstractArray{<:Number, 2}, B_i::AbstractArray{<:Number, 2}, alpha::AbstractArray{<:Number, 1}, t::AbstractArray{<:Number, 1})
    """
    Compute the current residual, objective, and relative error.
    """
     size(X_i), size(B_i), size(exp_function(alpha, t))
    residual = X_i .- (exp_function(alpha, t) * B_i)
    objective = 0.5 * norm(residual) .^ 2
    err = norm(residual) / norm(X_i)

    return residual, objective, err
end

function computeB(alpha::AbstractArray{<:Number, 1}, t::AbstractArray{<:Number, 1}, H::AbstractArray{<:Number, 2})
    A = exp_function(alpha, t)
    return A \ H
end

function _bag(X::AbstractArray{<:Number, 2}, trial_size::Number, rng::AbstractRNG=Random.GLOBAL_RNG)
    if 0 < trial_size < 1
        batch_size = round(Int, trial_size * size(X, 2))
    elseif trial_size >= 1 && isa(trial_size, Integer)
        batch_size = Int(trial_size) # Ensure integer type
    else
        error("Invalid trial_size parameter. trial_size must be either a positive integer or a float between 0 and 1.")
    end

    # Throw an error if the batch size is too large or too small.
    if batch_size > size(X, 2)
        error("_bag Error: batch_size ($batch_size) cannot be larger than number of samples ($(size(H, 1))). Check trial_size.")
    end

    if batch_size <= 0 # Check for non-positive
        error("_bag Error: batch_size ($batch_size) must be positive. Check trial_size.")
    end

    # Obtain and return subset of the data.
    all_inds = randperm(rng, size(X, 2))
    subset_inds = all_inds[1:batch_size]
    return X[:, subset_inds], subset_inds
end

function _compute_irank_svd(X::AbstractArray{<:Number, 2}, tolrank::Number)
    U, s, V = compute_svd(X, -1)
    irank = sum(s .> tolrank * s[1])
    irank = max(1, irank) # Ensure rank is at least 1
    U = U[:, 1:irank]
    S = diagm(s[1:irank])
    V = V[:, 1:irank]
    return U, S, V'
end

function _argsort_eigenvalues(eigs::AbstractArray{<:Number, 1}, sort_method::String="auto")
    if sort_method == "auto"
        real_var = var(real.(eigs))
        imag_var = var(imag.(eigs))
        abs_var = var(abs.(eigs))
        all_var = [real_var, imag_var, abs_var]
        sort_method = ["real", "imag", "abs"][argmax(all_var)]
    end

    if sort_method == "real"
        # Sort by real part, then imaginary part for ties
        p = sortperm(real.(eigs) .+ imag.(eigs)*eps())
    elseif sort_method == "imag"
        # Sort by imaginary part, then real part for ties
        p = sortperm(imag.(eigs) .+ real.(eigs)*eps())
    elseif sort_method == "abs"
        # Sort by magnitude
        p = sortperm(abs.(eigs))
    else
        error("Provided eig_sort method '$sort_method' is not supported.")
    end
    return p
end

function _variable_projection(
    X::AbstractArray{<:Number, 2},
    t::AbstractArray{<:Number, 1},
    init_alpha::Array{<:Number, 1},
    amp_limit::Number,
    init_lambda::Number,
    maxlam::Number,
    lamup::Number,
    use_levmarq::Bool,
    maxiter::Integer,
    tol::Number,
    eps_stall::Number,
    use_fulljac::Bool,
    verbose::Bool,
    eig_constraints::AbstractArray{<:AbstractString, 1},
)
     size(X), size(t), size(init_alpha)
    # Define M, IS, and IA.
    M, IS = size(X)
    IA = length(init_alpha)
    tolrank = M * eps()

    # Apply amplitude limits if provided.
    if amp_limit != Inf
        b = sqrt(sum(abs.(B) .^ 2, dims=2))
        B[b .> amp_limit] = eps
    end

    # Initialize values.
    _lambda = init_lambda
    alpha = _push_eigenvalues!(init_alpha, eig_constraints)

    B = computeB(alpha, t, X')
     size(B), size(exp_function(alpha, t))
    U_itr, S_itr, Vh_itr = _compute_irank_svd(exp_function(alpha, t), tolrank)
     size(U_itr), size(S_itr), size(Vh_itr)

    # Initialize termination flags.
    converged = false
    stalled = false

    # Initialize storage.
    all_error = zeros(maxiter)
    djac_matrix = zeros(Complex, (M * IS, IA))
    rjac = zeros(Complex, (2 * IA, IA))
    scales = zeros(IA)

    # Initialize iteration progress indicators.
     size(X), size(B), size(alpha)
    residual, objective, err = compute_error(X', B, alpha, t)

    for itr in 2:maxiter
        # Build Jacobian matrix, looping over alpha indices.
        for i in 1:IA
            # Build the approximate expression for the Jacobian.
            dphi_temp = exp_function_deriv(alpha, t, i)
             size(U_itr), size(dphi_temp)
            ut_dphi = U_itr' * dphi_temp
             size(ut_dphi)
            uut_dphi = U_itr * ut_dphi
            djac_a = (dphi_temp - uut_dphi) * B
            djac_matrix[:, i] = djac_a[:]

            # Compute the full expression for the Jacobian.
            if use_fulljac
                transform = U_itr * inv(S_itr) * Vh_itr
                dphit_res = dphi_temp' * residual
                 size(transform), size(dphit_res)
                djac_b = transform * dphit_res
                djac_matrix[:, i] += djac_b[:]
            end

            # Scale for the Levenberg algorithm.
            scales[i] = 1
            # Scale for the Levenberg-Marquardt algorithm.
            if use_levmarq
                scales[i] = min(norm(djac_matrix[:, i]), 1)
                scales[i] = max(scales[i], 1e-6)
            end

            # Loop to determine lambda (the step-size parameter).
            rhs_temp = deepcopy(residual[:])
            rhs_tmp = reshape(rhs_temp, (size(rhs_temp)..., 1))
            q_out, djac_out, j_pvt = qr(djac_matrix, ColumnNorm())
            # Use the true variable projection.
            ij_pvt = zeros(Int, IA)
            ij_pvt[j_pvt] = collect(1:IA)
            rjac[1:IA, :] = triu(djac_out[1:IA, :])
            rhs_top = q_out' * rhs_tmp
            scales_pvt = scales[j_pvt[1:IA]]
            rhs = vcat(rhs_top[1:IA], zeros(Complex, IA, 1))
            
            function step(
                _lambda::Number,
                scales_pvt::AbstractArray{<:Number, 1},
                rhs::AbstractArray{<:Number, 1},
                ij_pvt::AbstractArray{<:Int, 1},
                )
                """
                Helper function that, when given a step size _lambda,
                computes and returns the updated step and alpha vectors.
                """
                # Compute the step delta.
                rjac[IA+1:end, :] = _lambda * diagm(scales_pvt)
                delta = rjac \ rhs
                delta = delta[ij_pvt]

                # Compute the updated alpha vector.
                alpha_updated = alpha[:] + delta[:]
                alpha_updated = _push_eigenvalues!(alpha_updated, eig_constraints)
                return delta, alpha_updated
            end

            # Take a step using our initial step size init_lambda.
            if verbose
                # println("alpha before step: {$alpha}")
            end
             size(rhs)
            delta_0, alpha_0 = step(_lambda, scales_pvt, vec(rhs), ij_pvt)
            B_0 = computeB(alpha_0, t, X')
            residual_0, objective_0, error_0 = compute_error(X', B_0, alpha_0, t)

            # Check actual improvement vs predicted improvement.
            actual_improvement = objective - objective_0
            pred_improvement = (real(0.5 * (delta_0' * djac_matrix' * rhs_temp)[1]))
            improvement_ratio = actual_improvement / pred_improvement

            if error_0 < err
                # Rescale lambda based on the improvement ratio.
                _lambda *= max(1 / 3, 1 - (2 * improvement_ratio - 1) ^ 3)
                alpha, B = alpha_0, B_0
                residual, objective, err = residual_0, objective_0, error_0
            else
                # Increase lambda until something works.
                for _ in 1:maxlam
                    _lambda *= lamup
                    delta_0, alpha_0 = step(_lambda, scales_pvt, vec(rhs), ij_pvt)
                    B_0 = computeB(alpha_0, t, X')
                    residual_0, objective_0, error_0 = compute_error(X', B_0, alpha_0, t)

                    if abs(error_0) < abs(err)
                        break
                    end
                end

                # Terminate if no appropriate step length was found...
                if abs(error_0) >= abs(err)
                    if verbose
                        msg = "Failed to find appropriate step length at iteration $itr. Current error $err. Consider increasing maxlam or lamup."
                        # println(msg)
                    end
                    return B, alpha, converged
                end

                # ...otherwise, update and proceed.
                alpha, B = alpha_0, B_0
                residual, objective, err = residual_0, objective_0, error_0

            end

            if verbose
                # println("alpha after step\n{$alpha}")
            end

            # Update SVD information.
            U, S_itr, Vh = _compute_irank_svd(exp_function(alpha, t), tolrank)

            # Record the current relative error.
            all_error[itr] = err

            # Print iterative progress if the verbose flag is turned on.
            if verbose
                update_msg = "Step {$itr+1} Error {$err} Lambda {$_lambda}"
                # println(update_msg)
            end

            # Update termination status and terminate if converged or stalled.
            converged = err < tol
            error_reduction = all_error[itr - 1] - all_error[itr]
            stalled = (itr > 0) & (
                error_reduction < eps_stall * all_error[itr - 1]
            )

            if converged
                if verbose
                    # println("Convergence reached!")
                end
                return B, alpha, converged
            end

            if stalled
                if verbose
                    msg = """
                        Stall detected: error reduced by $error_reduction < eps_stall*previous_error = $(eps_stall * all_error[itr-1]).
                        Iteration {$itr+1}. Current error {$err}. Consider
                        increasing tol or decreasing eps_stall.
                    """
                    # println(msg)
                end
                return B, alpha, converged
            end
        end
    end

    # Failed to meet tolerance in maxiter steps.
    if verbose
        msg = """
            Failed to reach tolerance after maxiter = $maxiter iterations. 
            Current error $err.
        """
        # println(msg)
    end
    return B, alpha, converged
end

function _single_trial_compute_operator(X::AbstractArray{<:Number, 2},
                                        t::AbstractArray{<:Number, 1},
                                        init_alpha::AbstractArray{<:Number, 1},
                                        proj_basis::AbstractArray{<:Number, 2},
                                        use_proj::Bool,
                                        compute_A::Bool,
                                        verbose::Bool,
                                        eig_constraints::AbstractArray{<:String, 1},
                                        init_lambda::Number,
                                        maxlam::Integer,
                                        lamup::Number,
                                        use_levmarq::Bool,
                                        maxiter::Integer,
                                        tol::Number,
                                        eps_stall::Number,
                                        use_fulljac::Bool,
    )
    """
    Helper function that computes the standard optimized dmd operator.
    Returns the resulting DMD modes, eigenvalues, amplitudes, reduced
    system matrix, full system matrix, and whether or not convergence
    of the variable projection routine was reached.
    H input here is time x features (matching python snp.T)
    """
    # An amplitude limit is available but not implemented.
    b_lim = Inf

    B, alpha, converged = _variable_projection(X, t, init_alpha, b_lim, init_lambda, maxlam, lamup, use_levmarq, maxiter, tol, eps_stall, use_fulljac, verbose, eig_constraints)
    # Save the modes, eigenvalues, and amplitudes respectively.
    w = transpose(B)
    e = alpha
    b = vec(sqrt.(sum(abs.(w) .^ 2, dims=1)))
     size(w), size(e), size(b)
    
    # Normalize the modes and the amplitudes.
    inds_small = abs.(b) .< (10 * eps() * maximum(b))
    b[inds_small] .= 1.0
    w = w * diagm(1 ./ b)
    w[:, inds_small] .= 0.0
    b[inds_small] .= 0.0

    if use_proj
        Atilde = w * diagm(ComplexF64.(e)) * pinv(w)
        w = proj_basis * w
    else
        w_proj = proj_basis' * w
        Atilde = w_proj * diagm(e) * pinv(w_proj)
    end

    # Compute the full system matrix A.
    local A::Union{Matrix{ComplexF64}, Vector{}} = []
    if compute_A
        A = w * diagm(ComplexF64.(e)) * pinv(w)
    end

    return w, e, b, Atilde, A, converged
end

function compute_operator(X::AbstractArray{<:Number, 2},
                          t::AbstractArray{<:Number, 1},
                          init_alpha::AbstractArray{<:Number, 1},
                          proj_basis::AbstractArray{<:Number, 2},
                          use_proj::Bool,
                          compute_A::Bool,
                          verbose::Bool,
                          eig_constraints::AbstractArray{<:String, 1},
                          init_lambda::Number,
                          maxlam::Integer,
                          lamup::Number,
                          use_levmarq::Bool,
                          maxiter::Integer,
                          tol::Number,
                          eps_stall::Number,
                          use_fulljac::Bool,
                          num_trials::Integer,
                          trial_size::Number,
                          maxfail::Integer,
                          remove_bad_bags::Bool)
    """
    Compute the low-rank and the full BOP-DMD operators using parallel trials.

    :param H: Matrix of data to fit (time x features_proj or time x features_full).
    :type H: numpy.ndarray
    :param t: Vector of sample times.
    :type t: numpy.ndarray
    :return: Mean amplitude, A (if computed), mean eigs, std eigs, mean modes, std modes, std amplitudes
    :rtype: Tuple
    """
    # Perform an initial optimized dmd solve using init_alpha.
    # Use verbose=false for the single trial calls within the loop later
    initial_optdmd_results = _single_trial_compute_operator(
        X, t, init_alpha, proj_basis, use_proj, compute_A, verbose, eig_constraints, init_lambda, maxlam, lamup, use_levmarq, maxiter, tol, eps_stall, use_fulljac
    )
    w_0, e_0, b_0, Atilde_0, A_0, converged = initial_optdmd_results

    # Generate a warning if convergence wasn't initially reached.
    if !converged && verbose
        msg = """
            Initial trial of Optimized DMD failed to converge.
            Consider re-adjusting your variable projection parameters
            or increasing maxiter/tol.
        """
        # println(stderr, msg) # Print warning to stderr
    end

    # Perform BOP-DMD using parallel threads.
    if verbose
        # println("Starting $num_trials parallel bagging trials using $(Threads.nthreads()) threads...")
        if remove_bad_bags
            # println("Non-converged trial results will be removed...")
        else
            # println("Using all bag trial results...")
        end
    end

    # Storage for results from each thread
    # Store tuples: (w_i, e_i, b_i, converged_i)
    results = Vector{Union{Nothing, Tuple{Matrix{ComplexF64}, Vector{ComplexF64}, Vector{Float64}, Bool}}}(nothing, num_trials)

    # Use a separate RNG for each thread
    rngs = [Random.Xoshiro(i) for i in 1:Threads.nthreads()]

    for k in 1:num_trials
        thread_rng = rngs[Threads.threadid()]
        # Bag data for this trial
        X_i, subset_inds = _bag(X, trial_size, thread_rng)
        
        # Ensure subset_inds is not empty, though _bag should handle zero batch_size
            isempty(subset_inds) && continue # Skip if bagging failed or resulted in empty set

        # Run a single trial (suppress verbose output within the loop)
         size(X_i)
        w_i, e_i, b_i, _, _, converged_i = _single_trial_compute_operator(
                X_i, t[subset_inds], e_0, proj_basis, use_proj,compute_A, verbose,
                eig_constraints, init_lambda, maxlam, lamup, use_levmarq, maxiter, tol, eps_stall, use_fulljac
            )

        # Sort eigenvalues and permute modes/amplitudes accordingly
        sorted_inds = _argsort_eigenvalues(e_i)
        w_i_sorted = w_i[:, sorted_inds]
        e_i_sorted = e_i[sorted_inds]
        b_i_sorted = b_i[sorted_inds]

        # Store the result for this trial
        results[k] = (w_i_sorted, e_i_sorted, b_i_sorted, converged_i)
    end

    # Filter results based on convergence and remove_bad_bags flag
    successful_results = filter(x -> x !== nothing && (x[4] || !remove_bad_bags), results)
    num_successful_trials = length(successful_results)

    if num_successful_trials == 0
         error("BOP-DMD failed: No successful trials completed after filtering. Check convergence or bagging parameters.")
    end

    if verbose
        # println("Completed $num_successful_trials successful trials out of $num_trials.")
    end

    # Extract successful results
    successful_w = [res[1] for res in successful_results]
    successful_e = [res[2] for res in successful_results]
    successful_b = [res[3] for res in successful_results]


    # Compute the BOP-DMD statistics (Mean)
    # Summing matrices and vectors; initialize with zeros of correct type and size
    w_sum = zeros(ComplexF64, size(w_0))
    e_sum = zeros(ComplexF64, size(e_0))
    b_sum = zeros(Float64, size(b_0)) # Amplitudes 'b' are derived from abs(modes), so Float64

    for i in 1:num_successful_trials
        w_sum .+= successful_w[i]
        e_sum .+= successful_e[i]
        b_sum .+= successful_b[i]
    end

    w_mu = w_sum ./ num_successful_trials
    e_mu = e_sum ./ num_successful_trials
    b_mu = b_sum ./ num_successful_trials # Mean amplitude

    # Compute the BOP-DMD statistics (Standard Deviation)
    # Variance/Std Dev calculation uses magnitude squared
    w_sum2 = zeros(Float64, size(w_0)) # Store sum of |w|^2
    e_sum2 = zeros(Float64, size(e_0)) # Store sum of |e|^2
    b_sum2 = zeros(Float64, size(b_0)) # Store sum of b^2

    for i in 1:num_successful_trials
        w_sum2 .+= abs.(successful_w[i]) .^ 2
        e_sum2 .+= abs.(successful_e[i]) .^ 2
        b_sum2 .+= successful_b[i] .^ 2 # b is already real magnitude
    end

    # Variance calculation: E[X^2] - |E[X]|^2. Need abs for complex means.
    w_var = (w_sum2 ./ num_successful_trials) .- abs.(w_mu) .^ 2
    e_var = (e_sum2 ./ num_successful_trials) .- abs.(e_mu) .^ 2
    b_var = (b_sum2 ./ num_successful_trials) .- b_mu .^ 2

    # Ensure variance is non-negative due to potential floating point issues
    w_var[w_var .< 0] .= 0.0
    e_var[e_var .< 0] .= 0.0
    b_var[b_var .< 0] .= 0.0

    w_std = sqrt.(w_var)
    e_std = sqrt.(e_var)
    b_std = sqrt.(b_var) # Std dev of amplitude

    # Save the BOP-DMD statistics.
    _modes = w_mu
    _eigenvalues = e_mu
    _modes_std = w_std
    _eigenvalues_std = e_std
    _amplitudes_std = b_std # This is std dev of the magnitude b

    # Compute Atilde using the average optimized dmd results.
    w_proj = proj_basis' * _modes
    _Atilde = w_proj * diagm(_eigenvalues) * pinv(w_proj)
    
    # Compute the full system matrix A.
    if compute_A
            _A = _modes * diagm(ComplexF64.(_eigenvalues)) * pinv(_modes)
    else
            _A = []
    end

    # Return mean amplitude (b_mu), A (if computed), mean eigenvalues, std dev eigenvalues,
    # mean modes, std dev modes (magnitude), std dev amplitudes (magnitude)
    return b_mu, _Atilde, _A, _eigenvalues, _eigenvalues_std, _modes, _modes_std, _amplitudes_std
end

function fitBOPDMD(X::AbstractArray{<:Number, 2}, t::AbstractArray{<:Number, 1}; _svd_rank::Number=0, _use_proj::Bool=true, compute_A::Bool=true, verbose::Bool=true, eig_constraints::AbstractArray{<:String, 1}=["real"], _init_lambda::Number=1.0, maxlam::Integer=100, lamup::Number=2.0, use_levmarq::Bool=true, maxiter::Integer=100, tol::Number=1e-6, eps_stall::Number=1e-10, use_fulljac::Bool=true, num_trials::Integer=50, trial_size::Number=0.5, maxfail::Integer=10, remove_bad_bags::Bool=false)
    """
    Compute the Optimized Dynamic Mode Decomposition.

    :param X: the input snapshots.
    :type X: numpy.ndarray or iterable
    :param t: the input time vector.
    :type t: numpy.ndarray or iterable
    """
    # @assert size(X, 1) == length(t)

    # Compute the rank of the fit.
    _svd_rank, _, _ = compute_rank(X, _svd_rank)
    _svd_rank = Int(_svd_rank)

    # Set/check the projection basis.
    if _use_proj
        U, s, V = compute_svd(X, _svd_rank)
        _proj_basis = U
        _init_alpha = _initialize_alpha(s, V, _svd_rank, t)
    else
        U, s, V = compute_svd(X, size(X, 2))
        _proj_basis = U
        _init_alpha = _initialize_alpha(s, V, size(X, 2), t)
    end

    # Define the snapshots that will be used for fitting.
    if _use_proj
        snp = _proj_basis' * X
    else
        snp = X
    end

     size(snp)

    # Fit the data.
    # Pass the transpose of snp to match python's compute_operator input H
    _b, _,  A, _eigenvalues, _eigenvalues_std, _modes, _modes_std, _amplitudes_std = compute_operator(snp, t, _init_alpha, _proj_basis, _use_proj, compute_A, verbose, eig_constraints, _init_lambda, maxlam, lamup, use_levmarq, maxiter, tol, eps_stall, use_fulljac, num_trials, trial_size, maxfail, remove_bad_bags)

    return _b, _eigenvalues, _eigenvalues_std, _modes, _modes_std, _amplitudes_std
end

# timesteps = 1000
# dt = 0.1
# t = dt * collect(0:timesteps-1)
# tspan = (t[1], t[end])

# p = [1.5, 1.0, 3.0, 1.0]
# nTrials = 100
# u₀s = rand(2, nTrials)
# Xs = Vector{Matrix{Float64}}(undef, nTrials)
# Threads.@threads for ii in 1:nTrials
#     prob = ODEProblem(lotka_volterra, u₀s[:, ii], tspan, p)
#     sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=t)
#     U = Array(sol)

#     d, τ = 10, 1
#     X = delayEmbed(U', τ, d)'
#     Xs[ii] = X
# end
# timesteps = size(Xs[1], 2)
# t = dt * collect(0:timesteps-1)


# B, eig, eig_std, _mode, mode_std, amplitude_std = fitBOPDMD(Xs[1], t, eig_constraints=String[])

# Bs = Vector{Vector{ComplexF64}}(undef, size(Xs, 1))
# eigs = Vector{Vector{ComplexF64}}(undef, size(Xs, 1))
# eigs_std = Vector{Vector{Float64}}(undef, size(Xs, 1))
# modes = Vector{Matrix{ComplexF64}}(undef, size(Xs, 1))
# modes_std = Vector{Vector{Float64}}(undef, size(Xs, 1))
# amplitudes_std = Vector{Vector{Float64}}(undef, size(Xs, 1))
# Threads.@threads for ii in 1:size(Xs, 1)
# # for ii in 1:size(X, 1)
#     B, eig, eig_std, _mode, mode_std, amplitude_std = fitBOPDMD(Xs[ii], t, eig_constraints=String[])
#     Bs[ii] = complex.(B)
#     eigs[ii] = eig
#     eigs_std[ii] = eig_std
#     modes[ii] = _mode
#     modes_std[ii] = mode_std[1, :]
#     amplitudes_std[ii] = amplitude_std
# end