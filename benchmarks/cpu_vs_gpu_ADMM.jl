using CompressedSensingIPM, FFTW
using MadNLP, MadNLPGPU
using CUDA, AMDGPU
using Test

include("../test/fft_wei.jl")
include("../test/punching_centering.jl")
include("../test/ipm_example_3D.jl")
include("../test/admm_example_3D.jl")


run_admm_cpu = true
run_admm_gpu = true

if run_admm_gpu
    for (N1, N2, N3) in [(8, 8, 8), (8, 8, 8), (16, 16, 16),
                         (32, 32, 32), (64, 64, 64), (96, 96, 96),
                         (128, 128, 128), (192, 192, 192), (256, 256, 256),
                         (384, 384, 384), (512, 512, 512), (560, 560, 560)]

        println("Running ADMM on GPU for size: $N1 x $N2 x $N3")
        results, timer = admm_example_3D(N1, N2, N3; gpu=true, gpu_arch="cuda", rdft=true, check=false)
        println("Timer: $(timer)")
    end
end

if run_admm_cpu
    for (N1, N2, N3) in [(8, 8, 8), (8, 8, 8), (16, 16, 16),
                         (32, 32, 32), (64, 64, 64), (96, 96, 96),
                         (128, 128, 128), (192, 192, 192), (256, 256, 256),
                         (384, 384, 384), (512, 512, 512), (560, 560, 560)]

        println("Running ADMM on CPU for size: $N1 x $N2 x $N3")
        results, timer = admm_example_3D(N1, N2, N3; gpu=false)
        println("Timer: $(timer)")
    end
end
