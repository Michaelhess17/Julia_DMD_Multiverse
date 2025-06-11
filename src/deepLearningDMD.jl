using Flux, Flux.Optimise, Statistics, ProgressMeter
using LinearAlgebra
using Plots, NPZ
using DSP
using JLD2
# using Enzyme
# using BackwardsLinalg

# include("../utils/phaser.jl")
# include("tmp.jl")
include("residualDMD.jl")

T = Float32
device = cpu_device()
function get_model(data_size::Int, layer_1_size::Int, layer_2_size::Int, latent_dim::Int, device)
    model = Chain(Dense(data_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_2_size, selu),
                    Dense(layer_2_size => latent_dim), # ) 

                    Dense(latent_dim => layer_2_size, selu),
                    Dense(layer_2_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => data_size)) 
    return model
end


function fit_model(data::AbstractArray, T::Type, device::Union{CPUDevice, CUDADevice}=cpu_device(), layer_1_size::Int=128, latent_dim::Int=32, epochs::Int=1200)


    layer_2_size = Int(layer_1_size // 2)

    model = get_model(size(data, 1), layer_1_size, layer_2_size, latent_dim, device)

    loader = Flux.DataLoader(data, batchsize=128, shuffle=false, partial=false);

    opt_state = Flux.setup(Optimiser([Flux.Adam(0.0003)]), model)  # will store optimiser momentum, etc.

    # Training loop, using the whole data set 1000 times:
    losses = zeros(T, epochs)
    best_loss = T(Inf)
    best_model = nothing
    save_after = 1
    A = nothing
    p = Progress(epochs)
    for epoch in 1:epochs
        loss = nothing
        for Y in loader
            # grads = Flux.gradient(train_one, dup_model, Const(Y)) # use Enzyme
            grads = Flux.gradient(train_one, model, Y)
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

function train_one(m::Chain, y::AbstractArray{S}, α1=1.0, α2=1.0, α3=1.0, α4=1e-9) where S <: AbstractFloat
    enc, dec = Chain(m.layers[1:4]), Chain(m.layers[5:end])
    Ψ = enc(y)
    Ȳ = dec(Ψ)

    reconstruction_loss = Flux.mse(Ȳ, y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    F = svd(Ψ_minus)
    # K = F.U'*Ψ_plus*F.V*inv(Diagonal(F.S))
    E, V = eigen(K)

    k = V \ Ψ[:, 1]
    A = V*Diagonal(E)*pinv(V)
    Ψ̂ = reduce(hcat, [real.(V*Diagonal(E)^(jj)*k) for jj in collect(1:size(Ψ_minus, 2))::Vector{Int}])
    Ŷ = dec(Ψ̂ )

    linearity_mat = Ψ_plus*(I(size(Ψ_minus, 2)) .- F.V*F.Vt)
    linearity_loss = sum(abs2, linearity_mat) / length(linearity_mat)
    dmd_loss = Flux.mse(Ŷ, y[:, 2:end])

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
    total_loss = α1*reconstruction_loss + α2*linearity_loss + α3*dmd_loss + α4*l1_loss
    return total_loss
end

function get_eigs(m::Chain, y::AbstractArray{T})
    enc, dec = Chain(m.layers[1:4]), Chain(m.layers[5:end])
    Ψ = enc(y)
    Ȳ = dec(Ψ)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    F = svd(Ψ_minus)
    # K = F.U'*Ψ_plus*F.V*inv(Diagonal(F.S))
    E, V = eigen(K) 
    return E, V
end


function train_until_converged(f::Function, jj::Int, max_retries::Int=5)
    converged = false
    attempt = 1
    best_loss = [Inf]
    best_model = Chain()
    try
        best_loss, best_model = losses[jj], models[jj]
    catch e
        if e isa UndefRefError
            best_loss = [Inf]
            best_model = Chain()
        else
            rethrow(e)
        end
    end
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
            @warn e
            attempt += 1
        end
    end
    if ~converged && minimum(best_loss) < Inf
        models[jj], losses[jj] = best_model, best_loss
    end
end

function get_defined_indices(X::AbstractArray)
    good_indices = Int[]
    for idx in eachindex(X)
        try
            y = X[idx]
            push!(good_indices, idx)
        catch e
            if e isa UndefRefError
                nothing
            else
                rethrow(e)
            end
        end
    end
    return good_indices
end