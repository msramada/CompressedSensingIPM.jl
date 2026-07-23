using DelimitedFiles

println("Running IPM benchmarks...")
try
    include("cpu_vs_gpu_IPM.jl")
    println("Finished IPM benchmarks.")
catch e
    println("IPM benchmarks failed with error: ", e)
end

println("Running Gondzio benchmarks...")
try
    include("cpu_vs_gpu_Gondzio.jl")
    println("Finished Gondzio benchmarks.")
catch e
    println("Gondzio benchmarks failed with error: ", e)
end

println("Running ADMM benchmarks...")
try
    include("cpu_vs_gpu_ADMM.jl")
    println("Finished ADMM benchmarks.")
catch e
    println("ADMM benchmarks failed with error: ", e)
end

# Build a combined CSV table from whatever records each benchmark managed to collect.
# Missing/failed runs are filled with "NA" rather than aborting the whole table.
sizes_list = [(8, 8, 8), (8, 8, 8), (16, 16, 16),
              (32, 32, 32), (64, 64, 64), (96, 96, 96),
              (128, 128, 128), (192, 192, 192), (256, 256, 256),
              (384, 384, 384), (512, 512, 512), (560, 560, 560)]

ipm_cpu     = @isdefined(ipm_cpu_records)     ? ipm_cpu_records     : NamedTuple[]
ipm_gpu     = @isdefined(ipm_gpu_records)     ? ipm_gpu_records     : NamedTuple[]
gondzio_cpu = @isdefined(gondzio_cpu_records) ? gondzio_cpu_records : NamedTuple[]
gondzio_gpu = @isdefined(gondzio_gpu_records) ? gondzio_gpu_records : NamedTuple[]
admm_cpu    = @isdefined(admm_cpu_records)    ? admm_cpu_records    : NamedTuple[]
admm_gpu    = @isdefined(admm_gpu_records)    ? admm_gpu_records    : NamedTuple[]

ipm_cols(records, i) = i <= length(records) ?
    (records[i].timer, records[i].ipm_iters, records[i].krylov_total) : ("NA", "NA", "NA")
admm_cols(records, i) = i <= length(records) ? (records[i].timer,) : ("NA",)

header = ["Size",
          "IPM_CPU_Timer", "IPM_CPU_IPMIters", "IPM_CPU_KrylovIters",
          "IPM_GPU_Timer", "IPM_GPU_IPMIters", "IPM_GPU_KrylovIters",
          "Gondzio_CPU_Timer", "Gondzio_CPU_IPMIters", "Gondzio_CPU_KrylovIters",
          "Gondzio_GPU_Timer", "Gondzio_GPU_IPMIters", "Gondzio_GPU_KrylovIters",
          "ADMM_CPU_Timer", "ADMM_GPU_Timer"]

rows = Vector{Any}[header]
for (i, (N1, N2, N3)) in enumerate(sizes_list)
    row = Any["$N1 x $N2 x $N3"]
    append!(row, collect(ipm_cols(ipm_cpu, i)))
    append!(row, collect(ipm_cols(ipm_gpu, i)))
    append!(row, collect(ipm_cols(gondzio_cpu, i)))
    append!(row, collect(ipm_cols(gondzio_gpu, i)))
    append!(row, collect(admm_cols(admm_cpu, i)))
    append!(row, collect(admm_cols(admm_gpu, i)))
    push!(rows, row)
end

csv_path = "benchmark_results.csv"
writedlm(csv_path, rows, ',')
println("Benchmark results written to $(csv_path)")