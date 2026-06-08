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