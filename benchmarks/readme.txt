The codes cpu_vs_gpu_IPM.jl, cpu_vs_gpu_ADMM.jl, cpu_vs_gpu_Gondzio.jl
are used to generate Tables~2, 3 and 4, respectively.

The code run_benchmarks.jl run all of the above files in a single run

The output to terminal is stored in output_all.txt via using the following in command line
/benchmarks$ julia --project=. run_benchmarks.jl | tee output_all.txt