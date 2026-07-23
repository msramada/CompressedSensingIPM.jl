using CompressedSensingIPM, FFTW
using MadNLP, MadNLPGPU
using CUDA, AMDGPU
using Test

include("../test/fft_wei.jl")
include("../test/punching_centering.jl")
include("../test/ipm_example_3D.jl")


run_cpu = true
rdft_cpu = true
check_cpu = false
gondzio_cpu_records = NamedTuple[]
# CPU

if run_cpu
    for (N1, N2, N3) in [(8, 8, 8), (8, 8, 8), (16, 16, 16),
                         (32, 32, 32), (64, 64, 64), (96, 96, 96),
                         (128, 128, 128), (192, 192, 192), (256, 256, 256),
                         (384, 384, 384), (512, 512, 512), (560, 560, 560)]

        println("Running Gondzio on CPU for size: $N1 x $N2 x $N3")
        nlp, solver, results, timer = redirect_stdout(devnull) do
            ipm_example_3D(N1, N2, N3; kkt=GondzioKKTSystem, gpu=false, rdft=rdft_cpu, check=check_cpu)
        end
        println("Timer: $(timer)")
        krylov_iters = solver.kkt.krylov_iterations
        println("IPM iterations: $(solver.cnt.k)")
        println("Krylov iterations: $(krylov_iters) (total=$(sum(krylov_iters)), calls=$(length(krylov_iters)))")
        push!(gondzio_cpu_records, (N1=N1, N2=N2, N3=N3, timer=timer, ipm_iters=solver.cnt.k, krylov_total=sum(krylov_iters)))
    end
end

# GPU
run_gpu = true
rdft_gpu = true
check_gpu = false
gpu_arch = "cuda"  # "rocm"
gondzio_gpu_records = NamedTuple[]

if run_gpu
    for (N1, N2, N3) in [(8, 8, 8), (8, 8, 8), (16, 16, 16),
                         (32, 32, 32), (64, 64, 64), (96, 96, 96),
                         (128, 128, 128), (192, 192, 192), (256, 256, 256),
                         (384, 384, 384), (512, 512, 512), (560, 560, 560)]

        println("Running Gondzio on GPU for size: $N1 x $N2 x $N3")

        nlp, solver, results, timer = redirect_stdout(devnull) do
            ipm_example_3D(N1, N2, N3; kkt=GondzioKKTSystem, gpu=true, gpu_arch, rdft=rdft_gpu, check=check_gpu)
        end
        println("Timer: $(timer)")
        krylov_iters = solver.kkt.krylov_iterations
        println("IPM iterations: $(solver.cnt.k)")
        println("Krylov iterations: $(krylov_iters) (total=$(sum(krylov_iters)), calls=$(length(krylov_iters)))")
        push!(gondzio_gpu_records, (N1=N1, N2=N2, N3=N3, timer=timer, ipm_iters=solver.cnt.k, krylov_total=sum(krylov_iters)))
        GC.gc(true)
        if gpu_arch == "cuda"
        CUDA.reclaim()
        elseif gpu_arch == "rocm"
        AMDGPU.reclaim()
        end
    end
end
