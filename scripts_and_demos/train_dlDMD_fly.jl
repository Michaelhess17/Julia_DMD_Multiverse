using LinearAlgebra
using Flux
using NPZ, ProgressMeter, Plots, DSP
using PyCall, Glob, Statistics

include("../src/deepLearningDMD.jl")
include("../../utils/phaser.jl")
include("load_clean_fly_data.jl")
T = Float32
pattern = "**/angles/*.csv"
data_loc = "/home/michael/Synology/Python/Gait-Signatures/data/Anipose/fly-anipose/fly-testing/"
# data_loc_wt = "/home/michael/Synology/Python/Gait-Signatures/data/flyangles-dataset/"
files = glob(pattern, data_loc)
# files = vcat(files, glob(pattern, data_loc_wt))

std_cutoff = 4.5


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
        X_tmp = X .- X_bar
        X_keep_mask = sum(abs.(X_tmp ./ X_std)  .<= std_cutoff, dims=1) .== size(X, 1)
        X = X[:, findall(X_keep_mask[:])]
        Y = Y[:, findall(X_keep_mask[:])]
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



plots = []
for col_name in use_cols
    p = @df data plot(
        getproperty(data, col_name),
        label = string(col_name)
    )
    push!(plots, p)
end


num_plots = length(plots)
layout_rows = ceil(Int, sqrt(num_plots))
layout_cols = ceil(Int, num_plots / layout_rows)

combined_plot = plot(plots..., layout = (layout_rows, layout_cols), size = (800, 600))
