using Parquet2
using Glob
using DSP, DataFrames, Tables
using JLD2
using Statistics, PyCall
using Plots

include("/home/michael/Synology/Julia/utils/phaser.jl")

function load_data(T, files)
    leg_sides = ["L", "R"]
    leg_idx = ["1", "2", "3"]
    joint_idx = ["A", "B", "C", "D", "E"]
    movement_type = ["rot", "flex", "abduct"]
    combinations = [leg_side * leg_idx * joint_idx * "_" * movement_type for leg_side in leg_sides for leg_idx in leg_idx for joint_idx in joint_idx for movement_type in movement_type]
    all_dats = Vector{Vector{Tuple{Matrix{T}, Matrix{T}}}}(undef, length(files))
    println("Number of files: ", length(files))
    for (file_idx, file) in enumerate(files)
        println("Processing file $(file_idx): $(file)")
        # data = Parquet.read_parquet(file)
        data = Parquet2.Dataset(file)
        columnsnames = [string(i) for i in Tables.columnnames(data)]
        df = DataFrame(data)
        index = nothing
        idex_name = "index"
        try
            index = DataFrame(data)[!, "index"]
        catch e
            if e isa ArgumentError
                @warn "KeyError: 'index' column not found in file $(file)"
                index = df[!, "__index_level_0__"]
                idex_name = "__index_level_0__"
            else
                rethrow(e)
            end
        end
        fly_id = df[!, "flyid"]
        stimlen = df[!, "stimlen"]
        use_columns = []
        for combination in combinations
            if combination in columnsnames
                push!(use_columns, combination)
            end
        end
        dt = 1/180

        responsetype = Lowpass(30.0)
        designmethod = Butterworth(4)
        dats = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, length(unique(fly_id)))
        for (ii, fly) in enumerate(unique(fly_id))
            dat = T.(Matrix(df[fly_id .== fly, use_columns]))
            discont_locs = findall(diff(df[fly_id .== fly, idex_name]) .> 1)
            cdats = []
            for (start_loc, end_loc) in zip([1; discont_locs], [discont_locs .- 1; length(index[fly_id .== fly])])
                @show end_loc - start_loc
                current_dat = dat[start_loc:end_loc, :]
                for jj in 1:size(current_dat, 2)
                    current_dat[:, jj] = filtfilt(digitalfilter(responsetype, designmethod; fs=Int(1/dt)), current_dat[:, jj])
                end
                current_dat .-= mean(current_dat, dims=1)
                phased_dat = nothing
                try
                    phased_dat = phaseOne(current_dat, τ=1, k=5)'
                catch e
                    if e isa PyCall.PyError && occursin("SVD", string(e.val))
                        @warn "SVD Convergence error in phaser for fly $(fly), data shape: $(size(current_dat))"
                        continue
                    else
                        rethrow(e)
                    end
                end
                push!(cdats, phased_dat)
            end
            X, Y = nothing, nothing
            for jj in 1:length(cdats)
                if size(cdats[jj], 2) > 1
                    if ~isnothing(X)
                        X = hcat(X, cdats[jj][:, 1:end-1])
                        Y = hcat(Y, cdats[jj][:, 2:end])
                    else
                        X = cdats[jj][:, 1:end-1]
                        Y = cdats[jj][:, 2:end]
                    end
                end
            end
            dats[ii] = (X, Y)
        end
        all_dats[file_idx] = dats
    end
    return all_dats
end
# println("Total bouts: ", bouts)
# 
# total_size = 0
# for file_idx in 1:length(all_dats)
#     for ii in 1:length(all_dats[file_idx])
#         if !isnothing(all_dats[file_idx][ii])
#             X, Y = all_dats[file_idx][ii]
#             @show size(X)
#             total_size += size(X, 2)
#         end
#     end
# end
# 
# println("Total number of steps: ", floor(total_size // 99))
# 
# plot(all_dats[1][1][1][1:3, 1:end]', title="Example Data", xlabel="Time", ylabel="Amplitude")