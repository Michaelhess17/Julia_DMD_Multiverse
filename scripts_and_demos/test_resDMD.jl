using NPZ
using Plots
using LinearAlgebra
using ProgressMeter
using Folds
using ThreadsX

include("resDMD.jl")

data = npzread("/home/michael/Synology/Julia/data/human_data_TDE.npy")

# p = Progress(size(data, 1), 1, "Computing resDMD heatmaps...")

# for idx in 1:size(data, 1)
    # X = @view data[idx, :, :]
function getPlotDataResDMD(X, idx)

    PX = X[1:end-1, :]
    PY = X[2:end, :]
    W = ones(size(PX, 1)) #/size(PX, 1)

    x_pts = -1.5:0.01:1.5
    y_pts = -1.5:0.01:1.5
    z_pts_mat = [complex(a, b) for a in x_pts, b in y_pts]
    z_pts = z_pts_mat[:]

    res, _, _ = KoopPseudoSpecQR(PX, PY, W, z_pts; z_pts2=ComplexF64[], reg_param=1e-14, progress=false)

    res = reshape(res, size(x_pts, 1), size(y_pts, 1))
    # next!(p)
    return res
end
# finish!(p)

results = map(getPlotDataResDMD, eachslice(data; dims=1), 1:size(data, 1))

# Plot results
for (idx, res) in enumerate(results)
    x_pts = -1.5:0.01:1.5
    y_pts = -1.5:0.01:1.5
    contourf(x_pts, y_pts, res)
    savefig("figures/resDMD_heatmaps/$idx.png")
end