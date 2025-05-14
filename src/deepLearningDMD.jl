using Flux, Flux.Optimise, CUDA, Statistics, ProgressMeter
using LinearAlgebra
using Plots, NPZ
using DSP
# using Enzyme
# using BackwardsLinalg

include("utils/phaser.jl")
include("tmp.jl")

function fit_model(idx::Int, T::Type)
    data = load_data(idx, T, device)


    layer_1_size = 64
    layer_2_size = Int(layer_1_size // 2)

    latent_dim = 32

    # encoder
    model = Chain(Dense(size(data, 1) => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_2_size, selu),
                    Dense(layer_2_size => latent_dim, selu), # ) |> device
    # decoder = Chain(
                    Dense(latent_dim => layer_2_size, selu),
                    Dense(layer_2_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => size(data, 1))) |> device
        

    loader = Flux.DataLoader(data, batchsize=128, shuffle=false);

    opt_state = Flux.setup(Optimiser([Flux.Adam(0.0003)]), model)  # will store optimiser momentum, etc.

    # Training loop, using the whole data set 1000 times:
    epochs = 1500
    losses = zeros(T, epochs)
    best_loss = T(Inf)
    best_model = nothing
    save_after = 1
    A = nothing
    p = Progress(epochs)
    for epoch in 1:epochs
        loss = nothing
        for x in loader
            Y = x |> device
            # grads = Flux.gradient(train_one, dup_model, Const(Y)) # use Enzyme
            grads = nothing
            try
                grads = Flux.gradient(train_one, model, Y)
            catch e
                @show e
                if isa(e, ArgumentError)
                    warning("Gradient calculation failed due to $e. Retrying with perturbed weights...")
                    model.layers[3].weight .+= 1e-4.*randn(size(model.layers[3].weight)...)
                    grads = Flux.gradient(train_one, model, Y)
                else
                    rethrow(e)
                end
            end
            Flux.update!(opt_state, model, grads[1])
            loss = train_one(model, Y)
            losses[epoch] = loss  # logging, outside gradient context
            if (epoch > save_after) && (loss < best_loss)
                best_model = deepcopy(model)
                best_loss = loss
            end
        end
        next!(p; showvalues = [("Best loss:", best_loss), ("Current loss", loss)])
    end
    return best_model, losses
end

function train_one(m::Chain, y::AbstractArray{T}, α1::T=1, α2::T=1, α3::T=1, α4::T=1e-9)
    enc, dec = Chain(m.layers[1:3]), Chain(m.layers[4:end])
    Ψ = enc(y)
    Ȳ = dec(Ψ)

    reconstruction_loss = sum(abs2, Ȳ .- y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    E, V = eigen(K |> cpu) |> device

    k = V \ Ψ[:, 1]
    A = V*Diagonal(E)*pinv(V)
    Ψ̂ = reduce(hcat, [real.(V*Diagonal(E)^(jj)*k) for jj in collect(1:size(Ψ_minus, 2))::Vector{Int}])
    Ŷ = dec(Ψ̂ )

    F = svd(Ψ_minus)
    linearity_loss = sum(abs2, Ψ_plus*(I - F.V*F.Vt))
    dmd_loss = sum(abs2, Ŷ .- y[:, 2:end])

    l1_loss = 0
    # Regularize encoder weights
    for layer in enc
        if hasproperty(layer, :weight)
            l1_loss += sum(abs, layer.weight)
        end
    end
    # Regularize decoder weights
    for layer in dec
        if hasproperty(layer, :weight)
            l1_loss += sum(abs, layer.weight)
        end
    end
    total_loss = (α1*reconstruction_loss + α2*linearity_loss + α3*dmd_loss) + α4*l1_loss
    return total_loss
end

function get_eigs(m::Chain, y::AbstractArray{T})
    enc, dec = Chain(m.layers[1:3]), Chain(m.layers[4:end])
    Ψ = enc(y)
    Ȳ = dec(Ψ)

    reconstruction_loss = sum(abs2, Ȳ .- y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    E, V = eigen(K |> cpu) |> device
    return E, V
end

function load_data(idx::Int, T::DataType, device::Union{CPUDevice, CUDADevice})
    data = T.(transpose(npzread("/home/michael/Synology/Julia/data/human_data.npy")[idx, :, :]::Matrix))
    dt = 0.01
    t = T.(collect(LinRange(0.0, size(data, 2)*dt, size(data, 2)))) |> device

    responsetype = Lowpass(5.5)
    designmethod = Butterworth(4)
    for jj in 1:size(data, 1)
        data[jj, :] = filtfilt(digitalfilter(responsetype, designmethod; fs=Int(1/dt)), data[jj, :])
    end

    data .-= mean(data, dims=2)
    return data
end

function train_until_converged(f::Function, jj::Int, max_retries::Int=5)
    converged = false
    attempt = 1
    best_loss = Inf
    best_model = Chain()
    while attempt < max_retries &&  ~converged
        try
            model, loss = f(jj)
            if minimum(loss) <= convergence_threshold
                converged = true
                models[jj], losses[jj] = model, loss
            else
                if minimum(loss) < minimum(best_loss)
                    best_model = deepcopy(model)
                    best_loss = loss
                end
                attempt += 1
            end
        catch e
            attempt += 1
        end
    end
    if ~converged && minimum(best_loss) < Inf
        models[jj], losses[jj] = best_model, best_loss
    end
end


numSubjects = 72
T = Float32

models = Vector{Chain}(undef, numSubjects)
losses = Vector{Vector{T}}(undef, numSubjects)

f(x::Int) = fit_model(x, T)
convergence_threshold = 120
max_retries = 5
p = Progress(numSubjects, color=:red)

Threads.@threads for jj in 1:numSubjects
    train_until_converged(f, jj, max_retries)
end

# rerun failures with more retries
max_retries = 10
redos = findall([minimum(loss)>convergence_threshold for loss in losses])
p = Progress(length(redos), color=:red)
Threads.@threads for jj in redos
    train_until_converged(f, jj, max_retries)
end

eigs = fill!(Vector{Vector{Complex{T}}}(undef, numSubjects), T[(NaN)])

for jj in 1:numSubjects
    data = load_data(jj, T, device)
    if minimum(losses[jj]) < convergence_threshold
        eigs[jj] = get_eigs(models[jj], data)[1]
    end
end



plot_array = []
for jj in 1:numSubjects
    scatter(real.(eigs[jj]), imag.(eigs[jj]), xlabel=nothing, ylabel=nothing, legend=false)
    fig = plot!(cos.(0.0:0.1:2.1pi), sin.(0.0:0.1:2.1pi), aspect_ratio=1)
    push!(plot_array, fig)
end
plot(plot_array[1:3:numSubjects]..., layout=(6, 4), size=(1600, 1200))

# out = hcat(run_model(best_model, data[:, 1], size(data, 2)-1)...) |> cpu
# Flux.mse(out, data[:, 2:end])

# plot(out' |> cpu)

# X, Y = data[:, 1:end-1]', data[:, 2:end]'
# K = X \ Y

# out2 = vcat(run_model(K, data[:, 1], size(data, 2)-1)...) |> cpu