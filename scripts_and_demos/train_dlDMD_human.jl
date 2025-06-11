using LinearAlgebra
using Flux
using NPZ, ProgressMeter, Plots, DSP
using PyCall

include("../src/deepLearningDMD.jl")
include("../utils/phaser.jl")
T = Float32

function load_data(idx::Int, T::DataType, device::Union{CPUDevice, CUDADevice}, data_loc::String="/home/michael/Synology/Julia/data/all_human_data.npy")
    data = T.(npzread(data_loc)[idx, :, :])::Matrix{T}
    τ, k = 1, 5
    # data = transpose(timeDelayEmbed(data, τ, k))::AbstractArray{T}
    data = phaseOne(data; τ=τ, k=k)'
    dt = 0.01

    responsetype = Lowpass(5.5)
    designmethod = Butterworth(4)
    for jj in 1:size(data, 1)
        data[jj, :] = filtfilt(digitalfilter(responsetype, designmethod; fs=Int(1/dt)), data[jj, :])
    end

    data .-= mean(data, dims=2)
    data ./= std(data, dims=2)
    return data
end

numSubjects = 299
layer_1_size = 128
latent_dim = 32
epochs = 500
max_retries = 5
device = cpu_device()
models = Vector{Chain}(undef, numSubjects)
losses = Vector{Vector{T}}(undef, numSubjects)
convergence_threshold = 5e-3
p = Progress(numSubjects, color=:red)

all_data = Vector{Matrix{T}}(undef, numSubjects)
for jj in 1:numSubjects
    data = load_data(jj, T, device, "/home/michael/Synology/Julia/data/all_human_data.npy")
    all_data[jj] = data
end


Threads.@threads for jj in 1:numSubjects
    f(x::Int) = fit_model(all_data[x], T, device, layer_1_size, latent_dim, epochs)
    train_until_converged(f, jj, max_retries)
    next!(p)
end



# rerun failures with more retries
max_retries = 10
redos = findall([minimum(loss)>convergence_threshold for loss in losses])
p = Progress(length(redos), color=:red)
Threads.@threads for jj in redos
    f(x::Int) = fit_model(all_data[jj], T, device, layer_1_size, latent_dim, epochs)
    train_until_converged(f, jj, max_retries)
    next!(p)
end

eigenvalues = fill!(Vector{Vector{Complex{T}}}(undef, numSubjects), T[(NaN)])
defined_indices = CartesianIndices(losses)[get_defined_indices(losses)]
skipped = CartesianIndex[]

for idx in defined_indices
    jj = idx.I[1]
    data = load_data(jj, T, device)
    if minimum(losses[jj]) < convergence_threshold
        eigenvalues[jj] = get_eigs(models[jj], data)[1]
    else
        push!(skipped, idx)
    end
end

model_states = [Flux.state(model) for model in models[defined_indices]]

save("outputs/dlDMD_models_losses_eigs.jld2", "models", model_states, "eigs", eigenvalues, "losses", losses, "indices", defined_indices)

plot_array = []
for jj in 1:numSubjects
    scatter(real.(eigs[jj]), imag.(eigs[jj]), xlabel=nothing, ylabel=nothing, legend=false)
    fig = plot!(cos.(0.0:0.1:2.1pi), sin.(0.0:0.1:2.1pi), aspect_ratio=1)
    push!(plot_array, fig)
end
plot(plot_array[1:3:numSubjects]..., layout=(6, 4), size=(1600, 1200))
