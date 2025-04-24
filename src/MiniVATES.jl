module MiniVATES

import JACC
JACC.@init_backend
import Pkg

if JACC.backend == "cuda"
    import CUDA.unsafe_free!
elseif JACC.backend == "amdgpu"
    import AMDGPU.unsafe_free!
else
    function unsafe_free!(arr) end
end

import MPI

function __init__()
    # - Initialize here instead of main so that the MPI context can be available
    # for tests.
    # - Conditional allows for case when external users (or tests) have already
    # initialized an MPI context.
    if !MPI.Initialized()
        MPI.Init()
    end
end

include("Util.jl")
include("PreallocArrays.jl")
include("Sort.jl")
include("Hist.jl")
include("Load.jl")
include("BinMD.jl")
include("BinBoxes.jl")
include("MDNorm.jl")
include("BinSeries.jl")

end
