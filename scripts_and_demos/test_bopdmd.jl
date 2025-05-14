using DrWatson
quickactivate("/home/michael/Documents/Julia/DMD/env")
using LinearAlgebra, Plots
using Distributions, Random, Statistics
using DataFrames, Pickle, CSV, NPZ
using ProgressMeter

include("bopdmd.jl")

τ, d = 1, 20

df = CSV.read("/home/michael/Documents/Python/HAVOK/delase/data/all_human_data_metadata.csv", DataFrame)
df = select(df, Not(:Column1))

Xs = npzread("/home/michael/Documents/Python/HAVOK/delase/data/all_human_data.npy")
Xs = mapslices(x -> delayEmbed(x, τ, d), Xs, dims=(2, 3))
Xs = permutedims(Xs, (1, 3, 2))

dt = 0.01
t = dt .* (0:size(Xs, 3)-1)
svd_rank = 40

B, eig, eig_std, _mode, mode_std, amplitude_std = fitBOPDMD(Xs[1, :, :], t; _svd_rank=svd_rank)

Bs = Vector{Union{Vector{ComplexF32}, Nothing}}(nothing, size(Xs, 1))
eigs = Vector{Union{Vector{ComplexF32}, Nothing}}(nothing, size(Xs, 1))
eigs_std = Vector{Union{Vector{Float32}, Nothing}}(nothing, size(Xs, 1))
modes = Vector{Union{Matrix{ComplexF32}, Nothing}}(nothing, size(Xs, 1))
modes_std = Vector{Union{Vector{Float32}, Nothing}}(nothing, size(Xs, 1))
amplitudes_std = Vector{Union{Vector{Float32}, Nothing}}(nothing, size(Xs, 1))

p = Progress(size(Xs, 1))
Threads.@threads for ii in axes(Xs, 1)
    B, eig, eig_std, _mode, mode_std, amplitude_std = fitBOPDMD(Xs[ii, :, :], t; _svd_rank=svd_rank, trial_size=0.8, num_trials=30)
    Bs[ii] = complex.(B)
    eigs[ii] = eig
    eigs_std[ii] = eig_std
    modes[ii] = _mode
    modes_std[ii] = mode_std[1, :]
    amplitudes_std[ii] = amplitude_std
    next!(p)
end
finish!(p)

bad_inds = findall(isnothing, Bs)
all_inds = 1:size(Xs, 1)
keep_inds = setdiff(all_inds, bad_inds)
valid_df = df[keep_inds, :]

valid_Bs = filter(!isnothing, Bs)
valid_eigs = filter(!isnothing, eigs)
valid_eigs_std = filter(!isnothing, eigs_std)
valid_modes = filter(!isnothing, _mode)
valid_modes_std = filter(!isnothing, modes_std)
valid_amplitudes_std = filter(!isnothing, amplitudes_std)

plot_color_mapping = Dict("AB" => :blue, "LF" => :green, "HF" => :orange)
# Plot eigenvalues
plot(real(valid_eigs[1]), imag(valid_eigs[1]), seriestype=:scatter, color=plot_color_mapping[valid_df[1, :lf_or_hf]], legend=false)
for ii in 2:size(valid_eigs, 1)
    plot!(real(valid_eigs[ii]), imag(valid_eigs[ii]), seriestype=:scatter, color=plot_color_mapping[valid_df[ii, :lf_or_hf]], legend=false)
end
# plot unit circle
phi = 0:0.01:2pi
scatter!(sin.(phi), cos.(phi), seriestype=:scatter, color=:black, legend=false, aspect_ratio=1)
xlims!(-2.0, 2.0)
ylims!(-2.0, 2.0)
savefig("figures/eigenvalues.pdf")

AB_eigs = hcat(valid_eigs[valid_df[!, :lf_or_hf] .== "AB"]...)
LF_eigs = hcat(valid_eigs[valid_df[!, :lf_or_hf] .== "LF"]...)
HF_eigs = hcat(valid_eigs[valid_df[!, :lf_or_hf] .== "HF"]...)

function bootstrap_mean(X, nboot)
    tmp = Vector{Float64}(undef, nboot)
    for ii in 1:nboot
        @inbounds tmp[ii] = mean(X[rand(1:end, length(X))])
    end
    return tmp
end

histogram(vec(real.(AB_eigs)), bins=500, label="AB", alpha=0.5, normalize=true)
histogram!(vec(real.(LF_eigs)), bins=50, label="LF", alpha=0.5, normalize=true)
histogram!(vec(real.(HF_eigs)), bins=1000, label="HF", alpha=0.5, normalize=true)
xlims!(-2, 2)
savefig("figures/eigenvalues_hist.pdf")

AB_eigs_unstable = vec(AB_eigs[real.(AB_eigs) .> 0.0])
LF_eigs_unstable = vec(LF_eigs[real.(LF_eigs) .> 0.0])
HF_eigs_unstable = vec(HF_eigs[real.(HF_eigs) .> 0.0])

histogram(bootstrap_mean(real.(AB_eigs_unstable), 1000), bins=20, label="AB", alpha=0.5, normalize=true)
histogram!(bootstrap_mean(real.(LF_eigs_unstable), 1000), bins=20, label="LF", alpha=0.5, normalize=true)
histogram!(bootstrap_mean(real.(HF_eigs_unstable), 1000), bins=20, label="HF", alpha=0.5, normalize=true)
# xlims!(-0.6, 0.5)
savefig("figures/eigenvalues_hist_unstable.pdf")

AB_eigs_stable = vec(AB_eigs[real.(AB_eigs) .<= 0.0])
LF_eigs_stable = vec(LF_eigs[real.(LF_eigs) .<= 0.0])
HF_eigs_stable = vec(HF_eigs[real.(HF_eigs) .<= 0.0])

histogram(bootstrap_mean(real.(AB_eigs_stable), 1000), bins=20, label="AB", alpha=0.5, normalize=true)
histogram!(bootstrap_mean(real.(LF_eigs_stable), 1000), bins=20, label="LF", alpha=0.5, normalize=true)
histogram!(bootstrap_mean(real.(HF_eigs_stable), 1000), bins=20, label="HF", alpha=0.5, normalize=true)
# xlims!(-0.6, 0.5)
savefig("figures/eigenvalues_hist_stable.pdf")
