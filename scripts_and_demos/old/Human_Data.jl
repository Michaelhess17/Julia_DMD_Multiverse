using Distributed
# addprocs([("michael@localhost", 24), ("michael@170.140.242.213", 12)])
using PyCall
include("utils/phaser.jl")

@everywhere begin

using LinearAlgebra
using Statistics
using Clustering
using Random
using CairoMakie
using NPZ
using SharedArrays
using LaTeXStrings


end

@everywhere include("/home/michael/Synology/Julia/utils/resDMD.jl")

@everywhere function run_script(dataTDE::AbstractArray{T}) where T <: Number
    M = size(dataTDE, 1)
    PX = view(dataTDE, 1:M-1, :)
    PY = view(dataTDE, 2:M, :)

    # Parameters
    M1 = size(PX, 1)::Int
    M2 = size(PX, 2)::Int
    delta_t = 0.01
    Ns = Int[50, 250, 1000]
    PHI(r) = exp.(-r)

    # Grid for pseudospectra
    x_pts = T.(collect(-1.1:0.025:1.1))::Vector{T}
    y_pts = T.(collect(-1.1:0.025:1.1))::Vector{T}

    M = M1::Int

    # Compute scaling
    X_mean = mean(PX, dims=2)::Matrix{T}
    d = mean(norm.(eachcol(PX .- X_mean)))::T

    #--- Residual Computation ---

    # Solve Koopman operator
    K = PX \ PY
    eigvals_K, eigvecs_K = eigen(K)
    LAM = eigvals_K::Vector{Complex{T}}

    # Residuals
    res = [norm(PY * v - PX * v * λ) / norm(PX * v) for (v, λ) in zip(eachcol(eigvecs_K), LAM)]::Vector{T}

    # --- Pseudospectra computation ---
    # Form grid
    z_pts_mat = [complex(a, b) for a = x_pts, b = y_pts]::Matrix{Complex{T}}
    z_pts = vec(z_pts_mat)::Vector{Complex{T}}  # flatten

    # Compute pseudospectrum
    # Assuming KoopPseudoSpecQR takes PX, PY directly
    RES, _, _ = KoopPseudoSpecQR(PX, PY, ones(eltype(PX), size(PX, 1))/M, z_pts, z_pts2=ComplexF32[], reg_param=1e-14, progress=true, mode="Arpack") # Set progress=false
    RES = reshape(RES, (length(x_pts), length(y_pts)))::Matrix{T}

    return PX, PY, LAM, res, RES, x_pts, y_pts
end

#--- Plotting Functions ---

@everywhere function plot_residuals(PX::AbstractMatrix, PY::AbstractMatrix, LAM::AbstractVector, res::AbstractVector, jj::Int, k::Int)
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (700, 400), fontsize=20)

    ax = Axis(f[1, 1], aspect = 1, limits = (-1.05, 1.05, -1.05, 1.05), xlabel = L"$\textrm{Re}(\lambda)$", ylabel = L"$\textrm{Im}(\lambda)$")
    scatter!(ax, real(LAM), imag(LAM), color=res, colormap=:turbo, markersize=10)
    lines!(ax, cos.(0:0.01:2pi), sin.(0:0.01:2pi), color=:black, linestyle=:dash, linewidth=1.5)
    Colorbar(f[1, 2], label="Residual", colormap=:turbo, limits=(minimum(res), maximum(res)))

    set_theme!(theme_latexfonts())
    save("figures/resDMD_heatmaps/residuals_$(jj)_k$(k)_hankel.png", f) # Include k in filename
end


@everywhere function plot_pseudospectrum(RES::AbstractMatrix{T}, LAM::AbstractVector{S}, x_pts::AbstractVector, y_pts::AbstractVector, jj::Int, k::Int) where T <: Number where S <: Complex
    f = Figure(fontsize=20)
    min_res = minimum(real.(RES))
    # Ensure minimum of v is not zero or negative for log10
    min_v_log = min_res > 0 ? log10(min_res) : -10 # Use a small default if min_res is not positive
    v = 10 .^ (min_v_log:0.1:1)

    ax, hm = contourf(f[1, 1][1, 1],
            x_pts,
            y_pts,
            log10.(max.(minimum(v), real.(RES))),
            levels=log10.(v),
            colormap=:viridis,
            axis = (aspect = 1, limits = (minimum(x_pts), maximum(x_pts), minimum(y_pts), maximum(y_pts)),
                    xlabel = L"\mathrm{Re}(\lambda)", ylabel = L"\mathrm{Im}(\lambda)"))

    # Plot eigenvalues and unit circle (with dashed line)
    scatter!(ax, real(LAM), imag(LAM), color=:red, markersize=3)
    lines!(ax, cos.(0:0.01:2pi), sin.(0:0.01:2pi), color=:black, linestyle=:dash, linewidth=1.5)

    Colorbar(f[1, 1][1, 2], hm, label = L"\log_{10}(\tau(\lambda))")
    set_theme!(theme_latexfonts())
    save("figures/resDMD_heatmaps/pseudospectrum_$(jj)_k$(k)_hankel.png", f) # Include k in filename
end


@everywhere function mappable(dataTDE::AbstractArray{T}, jj::S, τ::S, k::S) where T <: Number where S <: Int
    println("Processing jj = $jj") # Add some progress indicator
    println("  Running analysis...")
    PX, PY, LAM, res, RES, x_pts, y_pts = run_script(dataTDE) # Get results

    # Create figures directory if it doesn't exist
    mkpath("figures/resDMD_heatmaps")

    # Call plotting functions
    println("  Plotting residuals...")
    plot_residuals(PX, PY, LAM, res, jj, k)
    println("  Plotting pseudospectrum...")
    plot_pseudospectrum(RES, LAM, x_pts, y_pts, jj, k)
    println("  Finished jj = $jj")
end

data = npzread("/home/michael/Synology/Julia/data/human_data.npy")::Array{Float64, 3}

τ::Int64, k::Int64 = 3, 100

newData = phaseAll(data, τ=τ, k=k, steps_per_cycle=200, peaks_window_size=20)::Vector{Matrix{Float64}}

f = jj::Int -> mappable(newData[jj], jj, τ, k)

pmap(f, 1:size(newData, 1))

# Consider running sequentially if parallelization is causing issues
# for jj in 1:size(data, 1)
#      mappable(data, jj, τ, k)
# end
