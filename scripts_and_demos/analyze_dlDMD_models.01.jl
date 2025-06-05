
using JLD2, LinearAlgebra, CairoMakie, KernelDensity
using StatsBase
using LaTeXStrings

set_theme!(theme_latexfonts(font_size=14))

include("../../utils/statistics.jl")

results = load("../outputs/dlDMD_models_losses_eigs.jld2")
models_states = results["models"]
eigenvalues = results["eigs"]

AB_inds = 1:30
ST_inds = 31:72
model_inds = CartesianIndices(eigenvalues)

all_inds = vcat(AB_inds, ST_inds)
skip_inds = [ind for ind in all_inds if length(eigenvalues[ind]) == 1]
AB_inds = [ab for ab in AB_inds if ab ∉ skip_inds]
ST_inds = [st for st in ST_inds if st ∉ skip_inds]
AB_eigs = reduce(hcat, [eig for eig in eigenvalues[AB_inds] if length(eig) > 1])
ST_eigs = reduce(hcat, [eig for eig in eigenvalues[ST_inds] if length(eig) > 1])

all_eigs = hcat(AB_eigs, ST_eigs)

all_eigs_density = kde(abs.(all_eigs)[:])


peak_density = all_eigs_density.x[argmax(all_eigs_density.density)]
cutoff_density = peak_density - 1e-3

function get_var_explained_Fourier_eigs(model::Chain, y::AbstractArray, cutoff_density::AbstractFloat)
    enc, dec = Chain(model.layers[1:4]), Chain(model.layers[5:end])
    Ψ = enc(y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    E, V = eigen(K |> cpu) |> device

    k = V \ Ψ[:, 1]

    E_Diagonal = Matrix(Diagonal(E))
    E_Fourier = deepcopy(E_Diagonal)
    for ii in 1:size(E_Fourier, 1)
        if abs(E_Fourier[ii, ii]) < cutoff_density
            E_Fourier[ii, ii] = 0 + 0im
        end
    end

    A = V*E_Fourier*pinv(V)
    Ψ̂ = reduce(hcat, [real.(V*E_Fourier^(jj)*k) for jj in collect(1:size(Ψ_minus, 2))::Vector{Int}])

    residual = dec(Ψ̂ ) .- y[:, 2:end]
    var_explained = 1 - (var(residual)/var(y))
    return var_explained
end

function get_var_explained_Fourier_eigs(model::Chain, y::AbstractArray, n_eigs::Int)
    enc, dec = Chain(model.layers[1:4]), Chain(model.layers[5:end])
    Ψ = enc(y)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    K = Ψ_plus * pinv(Ψ_minus)
    E, V = eigen(K |> cpu) |> device

    k = V \ Ψ[:, 1]

    eig_inds = sortperm(abs.(E))
    E = E[eig_inds]
    E[end-n_eigs+1:end] .= 0.0+0.0im
    E_Diagonal = Matrix(Diagonal(E))

    A = V*E_Diagonal*pinv(V)
    Ψ̂ = reduce(hcat, [real.(V*E_Diagonal^(jj)*k) for jj in collect(1:size(Ψ_minus, 2))::Vector{Int}])

    residual = dec(Ψ̂ ) .- y[:, 2:end]
    var_explained = 1 - (var(residual)/var(y))
    return var_explained
end

function get_model(data_size::Int, layer_1_size::Int, layer_2_size::Int, latent_dim::Int, device)
    model = Chain(Dense(data_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_2_size, selu),
                    Dense(layer_2_size => latent_dim, selu), # ) |> device

                    Dense(latent_dim => layer_2_size, selu),
                    Dense(layer_2_size => layer_1_size, selu),
                    Dense(layer_1_size => layer_1_size, selu),
                    Dense(layer_1_size => data_size)) |> device
    return model
end

function get_eigs_scaled(m::Chain, y::AbstractArray{T})
    enc, dec = Chain(m.layers[1:4]), Chain(m.layers[5:end])
    Ψ = enc(y)
    Ȳ = dec(Ψ)

    Ψ_minus = @view Ψ[:, 1:end-1]
    Ψ_plus = @view Ψ[:, 2:end]

    # K = Ψ_plus * pinv(Ψ_minus)
    F = svd(Ψ_minus)
    # K = Ψ_plus*F.V*inv(Diagonal(F.S))*F.U'
    K = F.U'*Ψ_plus*F.V*inv(Diagonal(F.S))
    
    E, V = eigen(inv(Diagonal(sqrt.(F.S)))*K*Diagonal(sqrt.(F.S)))
    W = Diagonal(sqrt.(F.S))*V

    modes = Ψ_plus*F.V*inv(Diagonal(sqrt.(F.S)))*W

    ω = log.(E) ./ dt
    freqs = abs.(imag.(ω) / 2pi)
    powers = [norm(col)^2 for col in eachcol(modes)]
    return E, V
end


T = Float32
numSubjects = 72

all_data = [Matrix(load_data(ii, T, cpu_device())) for ii in 1:numSubjects]
all_data = permutedims(reshape(reduce(hcat, all_data), size(all_data[1])..., length(all_data)), (3, 1, 2))
loaded_models = Vector{Chain}(undef, size(models))
for jj in 1:length(loaded_models)
    model = get_model(size(all_data[jj], 1), 128, 64, size(models[jj].layers[4].weight, 1), cpu_device())
    model = Flux.loadmodel!(model, models[jj])
    loaded_models[jj] = model
end

var_exp = [get_var_explained_Fourier_eigs(model, y, cutoff_density) for (ii, (model, y)) in enumerate(zip(loaded_models, all_data)) if ii ∉ skip_inds ]

transient_eigs = [eig[abs.(eig) .< cutoff_density] for eig in eachcol(all_eigs)]
mean_transient_eigs = [median(abs.(eig)) for eig in transient_eigs]

fig = Figure(fontsize=18)
ax = Axis(fig[1, 1], xlabel="Proportion of Variance Explained by Unitary Eigenmodes", ylabel=L"\bar{\lambda}_{\textrm{transient}}")
CairoMakie.scatter!(ax, var_exp[1:length(AB_inds)], mean_transient_eigs[1:length(AB_inds)], label="Tied", color=:dodgerblue)
CairoMakie.scatter!(ax, var_exp[length(AB_inds):end], mean_transient_eigs[length(AB_inds):end], label="Split", color=:crimson)
axislegend(position=:lb)
save("figures/percent_var_explained_vs_median_transient_eigs_64_dims_mice.png", fig)


ts = range(0, stop=5, step=1/fs)  # seconds
signal = y[1, :]
#  plot(ts, signal)   # > 200K points, better to use InspectDR
n = length(signal)
nw = n÷50
spec = spectrogram(signal, nw, nw÷2; fs=fs)
heatmap(spec.time, spec.freq, spec.power, xguide="Time [s]", yguide="Frequency [Hz]")
