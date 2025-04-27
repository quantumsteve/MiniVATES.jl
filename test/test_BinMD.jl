include("test_data_constants.jl")
include("common.jl")

import MiniVATES
import JACC
import Atomix
import Adapt
import Test: @testset, @test

function benzil_ranges()
    x = range(start = -7.5375, length = 604, stop = 7.5375)
    y = range(start = -13.16524, length = 604, stop = 13.16524)
    z = range(start = -0.5, length = 2, stop = 0.5)
    return (x, y, z)
end

function bixbyite_ranges()
    x = range(start = -16.0, length = 602, stop = 16.0)
    y = range(start = -16.0, length = 602, stop = 16.0)
    z = range(start = -0.1, length = 2, stop = 0.1)
    return (x, y, z)
end

@testset "BinMD" begin
    x, y, z = benzil_ranges()
    rotFile = benzil_event_nxs_prefix * "0_extra_params.hdf5"
    eventFile = benzil_event_nxs_prefix * "0_BEFORE_MDNorm.nxs"
    fastEventFile = benzil_event_nxs_prefix * "0_events.nxs"
    if length(ARGS) > 0
        base_name = last(ARGS)
	if contains(base_name, "TOPAZ")
	    x, y, z = bixbyite_ranges()
    end
    rotFile = base_name * "_extra_params.hdf5"
    eventFile = base_name * "_BEFORE_MDNorm.nxs"
	fastEventFile = base_name * "_events.nxs"
    end

    extrasData = MiniVATES.loadExtrasData(rotFile)
    transforms2 = MiniVATES.makeTransforms(extrasData)
    transforms2b = MiniVATES.makeTransformsTranspose(extrasData)
    #events = MiniVATES.getEvents(MiniVATES.EventWorkspace(eventFile))
    fastEventData = MiniVATES.loadFastEventData(fastEventFile)

    h = MiniVATES.Hist3(x, y, z)
    hf = MiniVATES.Hist3(x, y, z)

    @time MiniVATES.binEvents!(h, fastEventData.events, fastEventData.weights, transforms2b)
    @time MiniVATES.binBoxes!(hf, fastEventData, transforms2)
    MiniVATES.reset!(h)
    MiniVATES.reset!(hf)
    @time MiniVATES.binEvents!(h, fastEventData.events, fastEventData.weights, transforms2b)
    @time MiniVATES.binBoxes!(hf, fastEventData, transforms2)

    @test binweights(h) == binweights(hf)

    write_cat(h)
end
