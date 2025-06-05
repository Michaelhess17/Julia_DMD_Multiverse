using NPZ, LinearAlgebra, Statistics, Plots
using Flux, ProgressMeter
using JLD2
# [24, 119, 29, 79] HC Split, PCD Split, HC Tied, PCD Tied
T = Float32

include("../src/residualDMD.jl")
include("../src/deepLearningDMD.jl")

function load_data(T::DataType)
    data_base_loc = "/home/michael/Synology/Python/Gait-Signatures/data/"
    data_loc = data_base_loc * "All_Rodent_fps200.npy"
    speed_loc = data_base_loc * "speedlist.npy"
    data = T.(npzread(data_loc))[:, 1:2:end, :]
    speeds = T.(npzread(speed_loc))[:, 1, :]

    fs = 100
    responsetype = Lowpass(16)
    designmethod = Butterworth(4)
    for ii in size(data, 1)
        for jj in 1:size(data, 3)
            data[ii, :, jj] = filtfilt(digitalfilter(responsetype, designmethod; fs=fs), data[ii, :, jj])
        end
    end

    data .-= mean(data, dims=2)
    data ./= std(data, dims=2)
    τ, k = 1, 2
    data = permutedims(cat([transpose(timeDelayEmbed(data[idx, :, :], τ, k))::AbstractArray{T} for idx in 1:size(data, 1)]..., dims=3), (3, 2, 1))
    return data, speeds
end
0.275 -> normal tied, 0.175 -> normal split
0.2 -> mutant tied, 0.125 -> mutant split
HC_tied_inds = findall(speed .== 0.275f0)
HC_split_inds = findall(speed .== 0.175f0)
PCD_tied = findall(speed .== 0.20f0)
PCD_split = findall(speed .== 0.125f0)

numSubjects = 203
layer_1_size = 256
latent_dim = 64
epochs = 500
models = Vector{Chain}(undef, (numSubjects))
losses = Vector{Vector{T}}(undef, numSubjects)
convergence_threshold = 1600
max_retries = 5

device = cpu_device()

data, speeds = load_data(T)
f(x::Int) = fit_model(data[x, 1000:size(data, 2), :]', T, device, layer_1_size, latent_dim, epochs)

p = Progress(numSubjects, color=:red)
Threads.@threads for jj in 1:numSubjects
        train_until_converged(f, jj, max_retries)
        next!(p)
    end
end


defined_indices = get_defined_indices(models)
eigenvalues = fill!(Vector{Vector{Complex{T}}}(undef, numSubjects), T[(NaN)])
skipped = CartesianIndex[]

for idx in defined_indices
    idx2 = CartesianIndices(models)[idx]
    jj = idx2.I[1]
    if minimum(losses[jj]) < convergence_threshold
        eigenvalues[jj] = get_eigs(models[jj], data[jj, :, :])[1]
    end
end

save("outputs/dlDMD_models_losses_eigs_latent_dim_rodents.jld2", "models", models, "eigs", eigenvalues, "losses", losses)
