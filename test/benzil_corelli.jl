include("test_data_constants.jl")
include("common.jl")

import MiniVATES
import MiniVATES: Hist3, C3

import MPI
# import Cthulhu

dur = @elapsed begin
    x = range(start = -7.5375, length = 604, stop = 7.5375)
    y = range(start = -13.16524, length = 604, stop = 13.16524)
    z = range(start = -0.5, length = 2, stop = 0.5)

    extras_events_files = Vector{NTuple{3,AbstractString}}()
    for file_num = benzil_event_nxs_min:benzil_event_nxs_max
        fNumStr = string(file_num)
        exFile = benzil_event_nxs_prefix * fNumStr * "_extra_params.hdf5"
        eventFile = benzil_event_nxs_prefix * fNumStr * "_BEFORE_MDNorm.nxs"
	fastEventFile = benzil_event_nxs_prefix * fNumStr * "_events.nxs"
        push!(extras_events_files, (exFile, eventFile, fastEventFile))
    end

    # MiniVATES.verbose()
    signal, h = MiniVATES.binSeries!(
        MiniVATES.parse_args(ARGS),
        # MiniVATES.parse_args(["--partition=histogram"])
        (x,y,z),
        benzil_sa_nxs_file,
        benzil_flux_nxs_file,
        extras_events_files,
        C3[1.0 1.0 0.0; 1.0 -1.0 0.0; 0.0 0.0 1.0],
    )
end

# if MPI.Comm_rank(MPI.COMM_WORLD) == 0
#     write_cat(signal, h)
#     println("Total app time: ", dur, "s")
# end
write_rank_cat(signal, h)
