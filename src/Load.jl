import HDF5
import JACC
import Adapt: adapt_structure

struct ExtrasWorkspace
    file::HDF5.File

    function ExtrasWorkspace(filename::AbstractString)
        if MiniVATES.be_verbose
            println("ExtrasWorkspace: ", filename)
        end
        new(HDF5.h5open(filename, "r"))
    end
end

@inline function getSkipDets(ws::ExtrasWorkspace)
    return adapt_structure(Array1, read(ws.file["skip_dets"]))
end

@inline function getRotationMatrix(ws::ExtrasWorkspace)
    return transpose(SquareMatrix3c(read(ws.file["expinfo_0"]["goniometer_0"])))
end

@inline function getSymmMatrices(ws::ExtrasWorkspace)
    symm = Vector{SquareMatrix3c}()
    #push!(symm, SquareMatrix3c([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]))
    symmGroup = ws.file["symmetryOps"]
    for i = 1:length(symmGroup)
        push!(symm, transpose(read(symmGroup["op_" * string(i - 1)])))
    end
    return symm
end

@inline function getUBMatrix(ws::ExtrasWorkspace)
    return transpose(SquareMatrix3c(read(ws.file["ubmatrix"])))
end

mutable struct ExtrasData
    skip_dets::Array1{Bool}
    rotMatrix::SquareMatrix3c
    symm::Vector{SquareMatrix3c}
    m_UB::SquareMatrix3c
    m_W::SquareMatrix3c

    ExtrasData() = new()

    function ExtrasData(skip_dets, rotMatrix, symm, m_UB)
        #m_W = SquareMatrix3c([1.0 1.0 0.0; 1.0 -1.0 0.0; 0.0 0.0 1.0])
        m_W = SquareMatrix3c([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
        new(skip_dets, rotMatrix, symm, m_UB, m_W)
    end
end

function set_m_W!(extrasData::ExtrasData, m_W)
    extrasData.m_W = SquareMatrix3c(m_W)
    return extrasData.m_W
end

function loadExtrasData(rot_nxs_file::AbstractString)
    let ws = ExtrasWorkspace(rot_nxs_file)
        skip_dets = getSkipDets(ws)
        rotMatrix = getRotationMatrix(ws)
        symm = getSymmMatrices(ws)
        m_UB = getUBMatrix(ws)
        return ExtrasData(skip_dets, rotMatrix, symm, m_UB)
    end
end

@inline ndet(extrasData::ExtrasData) = length(extrasData.skip_dets)

@inline function makeRotationTransforms(d::ExtrasData)
    transforms = Array1{SquareMatrix3c}(undef, length(d.symm))
    JACC.parallel_for(
        length(d.symm),
        (i, t) -> begin
            t.transforms[i] = inv(t.rotMatrix * t.m_UB * t.symm[i] * t.m_W)
        end,
        (
            transforms = transforms,
            symm = adapt_structure(JACCArray, d.symm),
            d.rotMatrix,
            d.m_UB,
            d.m_W,
        ),
    )
    # return Array1(map(op -> inv(d.rotMatrix * d.m_UB * op * d.m_W), d.symm))
    # return Array1([inv(d.rotMatrix * d.m_UB * op * d.m_W) for op in d.symm])
    return transforms
end

@inline function makeTransforms(d::ExtrasData)
    return Array1(map(op -> inv(d.m_UB * op * d.m_W), d.symm))
    # return Array1{SquareMatrix3c}(map(op -> inv(d.m_UB * op * d.m_W), d.symm))
end

@inline function makeTransformsTranspose(d::ExtrasData)
    #n_symm = size(d.symm)[1]
    #println(n_symm)
    #t = Array2c(undef,(9, n_symm))
    #for i in 1:n_symm
    #    @views t[:,i] = reshape(transpose(inv(d.m_UB * d.symm[i] * d.m_W)), (9))
    #end;
    #return t
    return Array1(map(op -> inv(d.m_UB * op * d.m_W), d.symm))
end

struct SolidAngleWorkspace
    file::HDF5.File

    function SolidAngleWorkspace(filename::AbstractString)
        if MiniVATES.be_verbose
            println("SolidAngleWorkspace: ", filename)
        end
        new(HDF5.h5open(filename, "r"))
    end
end

@inline function getSolidAngleValues(ws::SolidAngleWorkspace)
    saGroup = ws.file["mantid_workspace_1"]
    saData = read(saGroup["workspace"]["values"])
    dims = size(saData)
    @assert length(dims) == 2
    @assert dims[1] == 1
    # readData = saData[1, :]
    # solidAngleWS = map(x -> [x], readData)
    # return solidAngleWS
    return adapt_structure(JACCArray, saData[1,:])
end

@inline function getSolidAngleToIdx_Array(ws::SolidAngleWorkspace; dataSize = nothing)
    saGroup = ws.file["mantid_workspace_1"]
    dcData = read(saGroup["instrument"]["detector"]["detector_count"])
    dims = size(dcData)
    @assert length(dims) == 1
    if dataSize != nothing
        @assert dims[1] == dataSize
    end

    solidAngDetToIdx = Vector{SizeType}()
    detector = 1
    idx = 1
    for value in dcData
        for i = 1:value
            push!(solidAngDetToIdx, idx)
        end
        idx += 1
    end
    return solidAngDetToIdx
end

@inline function getSolidAngleToIdx_Dict(ws::SolidAngleWorkspace; dataSize = nothing)
    saGroup = ws.file["mantid_workspace_1"]
    dcData = read(saGroup["instrument"]["detector"]["detector_count"])
    dims = size(dcData)
    @assert length(dims) == 1
    if dataSize != nothing
        @assert dims[1] == dataSize
    end

    solidAngDetToIdx = Dict{Int32,SizeType}()
    detector::Int32 = 1
    idx::SizeType = 1
    for value in dcData
        for i = 1:value
            push!(solidAngDetToIdx, detector => idx)
            detector += 1
        end
        idx += 1
    end
    return solidAngDetToIdx
end

struct SolidAngleData
    solidAngleValues::Array1{ScalarType}
    solidAngDetToIdx::Array1{SizeType}
end

function loadSolidAngleData(sa_nxs_file::AbstractString)
    let ws = SolidAngleWorkspace(sa_nxs_file)
        solidAngleValues = getSolidAngleValues(ws)
        len = length(solidAngleValues)

        solidAngDetToIdx = getSolidAngleToIdx_Array(ws, dataSize = len)

        return SolidAngleData(solidAngleValues, solidAngDetToIdx)
    end
end

struct FluxWorkspace
    file::HDF5.File

    function FluxWorkspace(filename::AbstractString)
        if MiniVATES.be_verbose
            println("FluxWorkspace: ", filename)
        end
        new(HDF5.h5open(filename, "r"))
    end
end

@inline function getIntegrFlux_x(ws::FluxWorkspace)
    group = ws.file["mantid_workspace_1"]
    readData::Vector{ScalarType} = read(group["workspace"]["axis1"])
    dims = size(readData)
    @assert length(dims) == 1
    len = dims[1]
    ret = range(length = len, start = first(readData), stop = last(readData))
    @assert isapprox(ret[2] - ret[1], readData[2] - readData[1], rtol = 1.0e-8)
    return ret
end

@inline function getIntegrFlux_y(ws::FluxWorkspace)
    group = ws.file["mantid_workspace_1"]
    readDataY = read(group["workspace"]["values"])
    dims = size(readDataY)
    @assert (length(dims) == 2 || length(dims) == 1)
    if length(dims) == 2
        retData = readDataY
    else
	retData = reshape(readDataY,(dims[1],1))
    end
    return adapt_structure(JACCArray, retData)
end

@inline function getFluxDetToIdx_Array(ws::FluxWorkspace)
    group = ws.file["mantid_workspace_1"]
    dcData = read(group["instrument"]["detector"]["detector_count"])
    dims = size(dcData)
    @assert length(dims) == 1
    fluxDetToIdx = Vector{SizeType}()
    detector = 1
    idx = 1
    for value in dcData
        for i = 1:value
            push!(fluxDetToIdx, idx)
        end
        idx += 1
    end
    return adapt_structure(JACCArray, fluxDetToIdx)
end

@inline function getFluxDetToIdx_Dict(ws::FluxWorkspace)
    group = ws.file["mantid_workspace_1"]
    dcData = read(group["instrument"]["detector"]["detector_count"])
    dims = size(dcData)
    @assert length(dims) == 1
    fluxDetToIdx = Dict{Int32,SizeType}()
    detector = 1
    idx = 1
    for value in dcData
        for i = 1:value
            push!(fluxDetToIdx, detector => idx)
            detector += 1
        end
        idx += 1
    end
    return fluxDetToIdx
end

@inline function get_ndets(ws::FluxWorkspace)
    group = ws.file["mantid_workspace_1"]
    detGroup = group["instrument"]["physical_detectors"]
    readData = read(detGroup["number_of_detectors"])
    dims = size(readData)
    @assert dims == (1,)
    ndets::SizeType = readData[1]
end

struct FluxData
    integrFlux_x::AbstractRange{ScalarType}
    integrFlux_y::Array2{ScalarType}
    # fluxDetToIdx::Dict{Int32,SizeType}
    fluxDetToIdx::Array1{SizeType}
    ndets::SizeType
end

function loadFluxData(flux_nxs_file::AbstractString)
    let ws = FluxWorkspace(flux_nxs_file)
        group = ws.file["mantid_workspace_1"]

        integrFlux_x = getIntegrFlux_x(ws)

        integrFlux_y = getIntegrFlux_y(ws)

        fluxDetToIdx = getFluxDetToIdx_Array(ws)

        ndets = get_ndets(ws)

        return FluxData(integrFlux_x, integrFlux_y, fluxDetToIdx, ndets)
    end
end

struct EventWorkspace
    file::HDF5.File

    function EventWorkspace(filename::AbstractString)
        if MiniVATES.be_verbose
            println("EventWorkspace: ", filename)
        end
        new(HDF5.h5open(filename, "r"))
    end
end

struct FastEventWorkspace
    file::HDF5.File
    function FastEventWorkspace(filename::AbstractString)
        if MiniVATES.be_verbose
            println("EventWorkspace: ", filename)
        end
        new(HDF5.h5open(filename, "r"))
    end
end

@inline function getProtonCharge(ws::EventWorkspace)
    expGroup = ws.file["MDEventWorkspace"]["experiment0"]
    pcData = read(expGroup["logs"]["gd_prtn_chrg"]["value"])
    @assert size(pcData) == (1,)
    protonCharge = pcData[1]
end

@inline function getProtonCharge(ws::FastEventWorkspace)
    expGroup = ws.file["MDEventWorkspace"]["experiment0"]
    pcData = read(expGroup["logs"]["gd_prtn_chrg"]["value"])
    @assert size(pcData) == (1,)
    protonCharge = pcData[1]
end

@inline function _getEventsDataset(ws::EventWorkspace)
    ds = ws.file["MDEventWorkspace"]["event_data"]["event_data"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 2
    @assert dims[1] == 8
    return ds
end

@inline function _getWeightsDataset(ws::FastEventWorkspace)
    ds = ws.file["MDEventWorkspace"]["event_data"]["weights"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 1
    return ds
end

@inline function getWeights(ws::FastEventWorkspace)
    return adapt_structure(JACCArray, read(_getWeightsDataset(ws)))
end

@inline function _getBoxTypeDataset(ws::FastEventWorkspace)
    ds = ws.file["MDEventWorkspace"]["box_structure"]["box_type"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 1
    return ds
end

@inline function getBoxType(ws::FastEventWorkspace)
    return adapt_structure(JACCArray, read(_getBoxTypeDataset(ws)))
end

@inline function _getBoxExtents(ws::FastEventWorkspace)
    ds = ws.file["MDEventWorkspace"]["box_structure"]["extents"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 2
    @assert dims[2] == 6
    return ds
end

@inline function getBoxExtents(ws::FastEventWorkspace)
    return adapt_structure(JACCArray, view(read(_getBoxExtents(ws)), :, :))
end

@inline function _getBoxSignalDataset(ws::FastEventWorkspace)
    ds = ws.file["MDEventWorkspace"]["box_structure"]["box_signal"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 1
    return ds
end

@inline function getBoxSignal(ws::FastEventWorkspace)
    return adapt_structure(JACCArray, read(_getBoxSignalDataset(ws)))
end

@inline function _getEventIndexDataset(ws::FastEventWorkspace)
    ds = ws.file["MDEventWorkspace"]["box_structure"]["box_event_index"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 2
    @assert dims[2] == 2
    return ds
end

@inline function getEventIndex(ws::FastEventWorkspace)
    return adapt_structure(JACCArray, view(read(_getEventIndexDataset(ws)), :, :))
end

@inline function getEvents(ws::EventWorkspace)
    return adapt_structure(JACCArray, view(read(_getEventsDataset(ws)), :, :))
end

@inline function _getEventsDataset(ws::FastEventWorkspace)
    ds = ws.file["MDEventWorkspace"]["event_data"]["position"]
    dims, _ = HDF5.get_extent_dims(ds)
    @assert length(dims) == 2
    @assert dims[2] == 3
    return ds
end

@inline function getEvents(ws::FastEventWorkspace)
    return adapt_structure(JACCArray, view(read(_getEventsDataset(ws)), :, :))
end

@inline function getDetIds(ws::EventWorkspace)
    expGroup = ws.file["MDEventWorkspace"]["experiment0"]
    detGroup = expGroup["instrument"]["physical_detectors"]
    return adapt_structure(JACCArray, read(detGroup["detector_number"]))
end

@inline function getLowValues(ws::EventWorkspace)
    expGroup = ws.file["MDEventWorkspace"]["experiment0"]
    ds = expGroup["logs"]["MDNorm_low"]["value"]
    return adapt_structure(Array1, read(ds))
end

@inline function getHighValues(ws::EventWorkspace)
    expGroup = ws.file["MDEventWorkspace"]["experiment0"]
    ds = expGroup["logs"]["MDNorm_high"]["value"]
    return adapt_structure(Array1, read(ds))
end

mutable struct EventData
    lowValues::Array1c
    highValues::Array1c
    protonCharge::ScalarType
    thetaValues::Array1c
    phiValues::Array1c
    detIDs::Array1{SizeType}
    events::AbstractArray
end

mutable struct FastEventData
    protonCharge::ScalarType
    events::AbstractArray
    weights::Array1c
    boxType::Array1{UInt8}
    extents::AbstractArray
    signal::Array1r
    eventIndex::AbstractArray
end

@inline function updateEvents!(data::EventData, ws::EventWorkspace)
    unsafe_free!(parent(data.events))
    data.events = getEvents(ws)
    return nothing
end

@inline function updateEvents!(data::FastEventData, ws::FastEventWorkspace)
    unsafe_free!(parent(data.events))
    unsafe_free!(parent(data.weights))
    unsafe_free!(parent(data.boxType))
    unsafe_free!(parent(data.extents))
    unsafe_free!(parent(data.signal))
    unsafe_free!(parent(data.eventIndex))
    data.events = getEvents(ws)
    data.weights = getWeights(ws)
    data.boxType = getBoxType(ws)
    data.extents = getBoxExtents(ws)
    data.signal = getBoxSignal(ws)
    data.eventIndex = getEventIndex(ws)
    return nothing
end

function loadEventData(event_nxs_file::AbstractString)
    let ws = EventWorkspace(event_nxs_file)
        expGroup = ws.file["MDEventWorkspace"]["experiment0"]
        lowValues = read(expGroup["logs"]["MDNorm_low"]["value"])
        dims = size(lowValues)
        @assert length(dims) == 1
        dataSize = dims[1]
        highValues = read(expGroup["logs"]["MDNorm_high"]["value"])
        dims = size(highValues)
        @assert length(dims) == 1
        @assert dims[1] == dataSize

        protonCharge = getProtonCharge(ws)

        detGroup = expGroup["instrument"]["physical_detectors"]
        thetaData::Array1{CoordType} = read(detGroup["polar_angle"])
        dims = size(thetaData)
        @assert length(dims) == 1
        @assert dims[1] == dataSize

        phiData::Array1{CoordType} = read(detGroup["azimuthal_angle"])
        dims = size(phiData)
        @assert length(dims) == 1
        @assert dims[1] == dataSize

        thetaValues = thetaData
        phiValues = phiData
        JACC.parallel_for(
            dataSize,
            (i, thetaValues, phiValues) -> begin
                thetaValues[i] = deg2rad(thetaValues[i])
                phiValues[i] = deg2rad(phiValues[i])
            end,
            thetaValues,
            phiValues,
        )

        detIDs = getDetIds(ws)

        events = getEvents(ws)

        return EventData(
            lowValues,
            highValues,
            protonCharge,
            thetaValues,
            phiValues,
            detIDs,
            events,
        )
    end
end

function loadFastEventData(event_nxs_file::AbstractString)
    let ws = FastEventWorkspace(event_nxs_file)
        protonCharge = getProtonCharge(ws)
        events = getEvents(ws)
	weights = getWeights(ws)
	boxType = getBoxType(ws)
	boxExtents = getBoxExtents(ws)
	boxSignal = getBoxSignal(ws)
	eventIndex = getEventIndex(ws)
        return FastEventData(
            protonCharge,
            events,
	    weights,
	    boxType,
	    boxExtents,
	    boxSignal,
	    eventIndex
        )
    end
end

