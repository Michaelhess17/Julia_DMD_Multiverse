using LinearAlgebra
using Flux
using NPZ, ProgressMeter, Plots, DSP
using PyCall, Glob, Statistics

include("../src/deepLearningDMD.jl")
include("../../utils/phaser.jl")
include("load_clean_fly_data.jl")
T = Float32
pattern = "*.pq"
data_loc = "/home/michael/Synology/Python/Gait-Signatures/data/FeCo_opto_data/"
data_loc_wt = "/home/michael/Synology/Python/Gait-Signatures/data/flyangles-dataset/"
files = glob(pattern, data_loc)
files = vcat(files, glob(pattern, data_loc_wt))


dats = load_data(T, files)

all_Xs = Matrix{T}[]
all_Ys = Matrix{T}[]
ids = Tuple{Int, Int}[]
for (dat_idx, dat) in enumerate(dats)
    for (fly_idx, fly) in enumerate(dat)
        X = fly[1]
        Y = fly[2]
        X_bar = mean(X, dims=2)
        X_std = std(X, dims=2)
        X .-= X_bar
        X ./= X_std
        Y .-= X_bar
        Y ./= X_std
        
        push!(all_Xs, X)
        push!(all_Ys, Y)
        push!(ids, (fly_idx, dat_idx))
    end
end

# Release redundant memory
dats = nothing

numSubjects = length(all_Xs)
println("Number of subjects: ", numSubjects)
layer_1_size = 128
latent_dim = 32
epochs = 500
max_retries = 5
device = cpu_device()
models = Vector{Chain}(undef, numSubjects)
losses = Vector{Vector{T}}(undef, numSubjects)
convergence_threshold = 2e-1
p = Progress(numSubjects, color=:red)


Threads.@threads for jj in 1:numSubjects
    f(x::Int) = fit_model(all_Xs[x], all_Ys[x], T, device, layer_1_size, latent_dim, epochs)
    train_until_converged(f, jj, max_retries)
    next!(p)
end