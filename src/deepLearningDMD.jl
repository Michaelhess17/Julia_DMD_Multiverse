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
                    Dense(layer_2_size => latent_dim), 

                    Dense(latent_dim => layer_2_size, selu),
                    Dense(layer_2_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => data_size)) 
    return model
end


function fit_model(X::AbstractArray, Y::AbstractArray, T::Type, device::Union{CPUDevice, CUDADevice}=cpu_device(), layer_1_size::Int=128, latent_dim::Int=32, epochs::Int=1200)
    # X: Ψ_minus, Y: Ψ_plus
    layer_2_size = Int(layer_1_size // 2)
    model = get_model(size(X, 1), layer_1_size, layer_2_size, latent_dim, device)
    loader = Flux.DataLoader((X, Y), batchsize=168, shuffle=true, partial=true)
    opt_state = Flux.setup(Optimiser([Flux.Adam(0.00003)]), model)
    losses = zeros(T, epochs)
    fill!(losses, NaN)
    best_loss = T(Inf)
    best_model = nothing
    save_after = 1
    A = nothing
    p = Progress(epochs)
    for epoch in 1:epochs
        # Train the model
        for (Xb, Yb) in loader
            if size(Xb, 2) < size(Xb, 1)
                # @warn "Batch size smaller than data dimension, skipping batch"
                continue
            end
            grads = Flux.gradient(train_one, model, Xb, Yb)
            Flux.update!(opt_state, model, grads[1])
        end
        # Evaluate the model
        epoch_loss = T(0.0)
        for (Xb, Yb) in loader
            epoch_loss += train_one(model, Xb, Yb)
        end
        epoch_loss /= length(loader)
        losses[epoch] = epoch_loss
        if (epoch > save_after) && (epoch_loss < best_loss)
            best_model = deepcopy(model)
            best_loss = epoch_loss
        end
    next!(p; showvalues = [("Best loss:", best_loss), ("Current loss", epoch_loss)])
end
    return best_model, losses
end

# Overload for backward compatibility (continuous data)
function fit_model(data::AbstractArray, T::Type, device::Union{CPUDevice, CUDADevice}=cpu_device(), layer_1_size::Int=128, latent_dim::Int=32, epochs::Int=1200)
    X = @view data[:, 1:end-1]
    Y = @view data[:, 2:end]
    return fit_model(X, Y, T, device, layer_1_size, latent_dim, epochs)
end

function train_one(m::Chain, X::AbstractArray{S}, Y::AbstractArray{S}, α1=1.0, α2=1.0, α3=1.0, α4=1e-9) where S <: AbstractFloat
    enc, dec = Chain(m.layers[1:4]), Chain(m.layers[5:end])
    Ψ_minus = enc(X)
    Ψ_plus = enc(Y)
    X̄ = dec(Ψ_minus)
    Ȳ = dec(Ψ_plus)

    reconstruction_loss = Flux.mse(X̄, X)

    # DMD operator from latent space
    if sum(isnan, Ψ_minus) > 0 || sum(isnan, Ψ_plus) > 0
        @warn "NaN detected in Ψ_minus or Ψ_plus"
        return Inf
    elseif sum(isinf, Ψ_minus) > 0 || sum(isinf, Ψ_plus) > 0
        @warn "Inf detected in Ψ_minus or Ψ_plus"
        return Inf
    end

    K = Ψ_plus * pinv(Ψ_minus)
    F = svd(Ψ_minus, alg=LinearAlgebra.DivideAndConquer())
    E, V = eigen(K)

    # k = V \ Ψ_minus[:, 1]
    # Ψ̂ = reduce(hcat, [real.(V*Diagonal(E)^(jj)*k) for jj in 1:size(Ψ_minus, 2)])
    Ψ̂ = K * Ψ_minus
    Ŷ = dec(Ψ̂)

    linearity_mat = Ψ_plus*(I(size(Ψ_minus, 2)) .- F.V*F.Vt)
    linearity_loss = sum(abs2, linearity_mat) / length(linearity_mat)
    dmd_loss = Flux.mse(Ŷ, Y)

    l1_loss = 0
    for layer in enc
        if hasproperty(layer, :weight)
            l1_loss += sum(abs, layer.weight)
        end
    end
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