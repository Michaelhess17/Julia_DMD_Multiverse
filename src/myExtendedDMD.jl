# NOTE: This is not what the standard "Extended DMD" algorithm is exactly. 
# Here, I am assuming periodic dynamics and using a Fourier basis to represent the data.
# I use the phaser algorithm to align the data to a common phase before applying DMD.
# This is more of a "Fourier DMD" with phase alignment, but this basis works very well for these data
# and can be more easily compared across individuals/trials than dataset-specific singular vectors.


ENV["LD_LIBRARY_PATH"] = "/run/opengl-driver/lib:/run/opengl-driver/lib32" # for NVIDIA drivers on NixOS
using Pkg
Pkg.activate("/home/michael/Synology/Julia/Julia_DMD_Multiverse")
Pkg.instantiate()
# using CairoMakie, KernelDensity, StatsPlots
using LinearAlgebra, DSP
using NPZ, JLD2, CSV, DataFrames, Statistics
using ProgressMeter
include("/home/michael/Synology/Julia/utils/phaser.jl")

function create_fourier_basis(frequencies, steps_per_cycle, n_modes)
    # Create a Fourier basis matrix with complex exponentials
    fourier_basis = reduce(hcat, exp.(2im * π * (i) * (0:n_modes-1) / steps_per_cycle) for i in frequencies)
    # Normalize each column
    fourier_basis ./= norm.(eachcol(fourier_basis))'
    # QR decomposition to get orthonormal basis
    Q, R = qr(fourier_basis)
    # Return the orthonormal basis
    return Q
end

function unproject_fourier(proj, fourier_basis, original_shape, n_original_features=6)
    # proj has shape (time_steps, num_fourier_modes * n_original_features)
    # We need to reshape to extract each feature's fourier embedding and multiply it by the transpose of the fourier basis to unproject.
    
    unprojected = zeros(ComplexF32, original_shape...)

    # For each of the original variables
    for ii in 1:n_original_features
        # Extract the projection for this variable)
        start_col = (ii-1) * size(fourier_basis, 2) + 1
        end_col = ii * size(fourier_basis, 2)
        var_proj = proj[:, start_col:end_col]
        
        # Unproject by multiplying with fourier basis transpose
        unprojected[:, ii:n_original_features:end] = var_proj * fourier_basis'
    end
    
    return unprojected
end

function fitExtendedDMD(data, τ, k, phase_data, frequencies; steps_per_cycle=100, n_original_features=nothing, bias=true)
    # Phase the data
    if ~phase_data
        # If phase_data is false, we assume the data is already phased or the user does not want to phase it.
        phased_data = data
        if n_original_features === nothing
            @warn "n_original_features is not specified and cannot determine from pre-phased or unphased data. Assuming 6 features per time step."
            n_original_features = 6
        end
    else
        # Otherwise, we apply the phaseOne function to the data
        # first use a one step, no delay embedding just to get the phase
        # then apply the timeDelayEmbed function to phase the data
        # with the specified τ and k.
        phased_data = timeDelayEmbed(phaseOne(data; τ=1, k=1, steps_per_cycle=steps_per_cycle), τ, k)
        n_original_features = size(data, 2)  # Number of original features is the number of columns in the data
    end
    phased_data .-= mean(phased_data, dims=1)

    # Create Fourier basis
    fourier_basis = create_fourier_basis(frequencies, steps_per_cycle, k)

    # Project the data onto the Fourier basis
    proj = hcat([phased_data[:, ii:n_original_features:end] * fourier_basis for ii in 1:n_original_features]...)

    original_shape = size(phased_data)

    X, Y = transpose(proj[1:end-1, :]), transpose(proj[2:end, :])
    if ~bias
        # If bias is false, we do not add a bias row to X
        Xb = X  # (d)×(T-1)
    else
        # Add a bias row of ones to X
        Xb = vcat(X, ones(1, size(X, 2)))  # (d+1)×(T-1)
    end
    Ktilde = Y * pinv(Xb)

    # Separate linear and bias parts
    if bias
        K = Ktilde[:, 1:end-1]  # d×d
        b = Ktilde[:, end]      # d
    else
        # If bias is false, we do not have a bias term
        K = Ktilde  # d×d
        b = zeros(size(K, 1))  # d, zero vector
    end

    F = svd(X, alg=LinearAlgebra.DivideAndConquer())

    reconstruction_error = sum(abs2, Y .- K * X .+ b)
    reconstruction_error /= size(Y, 2)  # Average over time steps

    # Return the results
    return K, b, F, reconstruction_error, proj, phased_data
end
