import Base: @propagate_inbounds

mutable struct MDNorm{TArray}
    # hX::TArray
    # kX::TArray
    # lX::TArray
    # eX::TArray
    extrasData::ExtrasData

    intersections::PreallocJaggedArray{Crd4}
    # iPerm::PreallocJaggedArray{SizeType}
    yValues::PreallocJaggedArray{ScalarType}

    function MDNorm(
        hx::TArray,
        kx::TArray,
        lx::TArray,
        extrasData::ExtrasData,
    ) where {TArray}
        maxIx = _maxIntersections(hx, kx, lx)
        ndets = ndet(extrasData)
        # intersections = PreallocJaggedArray{Crd4}(ndets, maxIx)
        intersections = PreallocJaggedArray{Crd4}()
        # iPerm = PreallocJaggedArray{SizeType}(ndets, maxIx)
        # yValues = PreallocJaggedArray{ScalarType}(ndets, maxIx)
        yValues = PreallocJaggedArray{ScalarType}()
        new{TArray}(
            # hx,
            # kx,
            # lx,
            # TArray(),
            extrasData,
            intersections,
            # iPerm,
            yValues,
        )
    end

    function MDNorm(
        hx::TArray,
        kx::TArray,
        lx::TArray,
    ) where {TArray}
        maxIx = _maxIntersections(hx, kx, lx)
        # intersections = PreallocJaggedArray{Crd4}(1, maxIx)
        # yValues = PreallocJaggedArray{ScalarType}(1, maxIx)
        intersections = PreallocJaggedArray{Crd4}()
        yValues = PreallocJaggedArray{ScalarType}()
        new{TArray}(#= hx, kx, lx, TArray(), =# ExtrasData(), intersections, yValues)
    end
end

function MDNorm()
    MDNorm(Array1r(), Array1r(), Array1r())
end

function MDNorm(
    hx::AbstractRange,
    kx::AbstractRange,
    lx::AbstractRange,
    extrasData::ExtrasData,
)
    MDNorm(collect(hx), collect(kx), collect(lx), extrasData)
end

function MDNorm(hx::AbstractRange, kx::AbstractRange, lx::AbstractRange)
    MDNorm(collect(hx), collect(kx), collect(lx))
end

function MDNorm(hist::Hist3, extrasData::ExtrasData)
    MDNorm(edges(hist)..., extrasData)
end

MDNorm(hist::Hist3) = MDNorm(edges(hist)...)

@inline function setExtrasData!(mdn::MDNorm, extrasData::ExtrasData)
    # maxIx = maxIntersections(mdn)
    mdn.extrasData = extrasData
    ndets = ndet(extrasData)
    # mdn.intersections = PreallocJaggedArray{Crd4}(ndets, maxIx)
    # iPerm = PreallocJaggedArray{SizeType}(ndets, maxIx)
    # mdn.yValues = PreallocJaggedArray{ScalarType}(ndets, maxIx)
end


@propagate_inbounds function countIntersections(
    histogram::Hist3,
    theta::CoordType,
    phi::CoordType,
    transform::SquareMatrix3c,
    lowvalue::CoordType,
    highvalue::CoordType,
)
    sin_theta = sin(theta)
    qout = C3[sin_theta * cos(phi), sin_theta * sin(phi), cos(theta)]
    qin = C3[0, 0, 1]

    qout = transform * qout
    qin = transform * qin

    kimin = lowvalue
    kimax = highvalue
    kfmin = kimin
    kfmax = kimax

    hx = edges(histogram)[1]
    kx = edges(histogram)[2]
    lx = edges(histogram)[3]

    hNPts = length(hx)
    kNPts = length(kx)
    lNPts = length(lx)

    hStart = qin[1] * kimin - qout[1] * kfmin
    hEnd = qin[1] * kimax - qout[1] * kfmax
    hStartIdx = binindex1d(histogram, 1, hStart)
    hEndIdx = binindex1d(histogram, 1, hEnd)
    hStartIdx = clamp(hStartIdx, 0, hNPts)
    hEndIdx = clamp(hEndIdx, 0, hNPts)
    if hStartIdx > hEndIdx
        hStartIdx, hEndIdx = (hEndIdx, hStartIdx)
    end
    hStartIdx += 1

    kStart = qin[2] * kimin - qout[2] * kfmin
    kEnd = qin[2] * kimax - qout[2] * kfmax
    kStartIdx = binindex(histogram, 0, kStart, 0)[2]
    kEndIdx = binindex(histogram, 0, kEnd, 0)[2]
    kStartIdx = clamp(kStartIdx, 0, kNPts)
    kEndIdx = clamp(kEndIdx, 0, kNPts)
    if kStartIdx > kEndIdx
        kStartIdx, kEndIdx = (kEndIdx, kStartIdx)
    end
    kStartIdx += 1

    lStart = qin[3] * kimin - qout[3] * kfmin
    lEnd = qin[3] * kimax - qout[3] * kfmax
    lStartIdx = binindex(histogram, 0, 0, lStart)[3]
    lEndIdx = binindex(histogram, 0, 0, lEnd)[3]
    lStartIdx = clamp(lStartIdx, 0, lNPts)
    lEndIdx = clamp(lEndIdx, 0, lNPts)
    if lStartIdx > lEndIdx
        lStartIdx, lEndIdx = (lEndIdx, lStartIdx)
    end
    lStartIdx += 1

    count = zero(SizeType)

    # calculate intersections with planes perpendicular to h
    fmom = (kfmax - kfmin) / (hEnd - hStart)
    fk = (kEnd - kStart) / (hEnd - hStart)
    fl = (lEnd - lStart) / (hEnd - hStart)
    for i = hStartIdx:hEndIdx
        hi = hx[i]
        # if hi is between hStart and hEnd, then ki and li will be between
        # kStart, kEnd and lStart, lEnd and momi will be between kfmin and
        # kfmax
        ki = fk * (hi - hStart) + kStart
        li = fl * (hi - hStart) + lStart
        if (ki >= kx[1]) && (ki <= kx[kNPts]) && (li >= lx[1]) && (li <= lx[lNPts])
            momi = fmom * (hi - hStart) + kfmin
            count += 1
        end
    end

    # calculate intersections with planes perpendicular to k
    fmom = (kfmax - kfmin) / (kEnd - kStart)
    fh = (hEnd - hStart) / (kEnd - kStart)
    fl = (lEnd - lStart) / (kEnd - kStart)
    for i = kStartIdx:kEndIdx
        ki = kx[i]
        # if ki is between kStart and kEnd, then hi and li will be between
        # hStart, hEnd and lStart, lEnd and momi will be between kfmin and
        # kfmax
        hi = fh * (ki - kStart) + hStart
        li = fl * (ki - kStart) + lStart
        if (hi >= hx[1]) && (hi <= hx[hNPts]) && (li >= lx[1]) && (li <= lx[lNPts])
            momi = fmom * (ki - kStart) + kfmin
            count += 1
        end
    end

    # calculate intersections with planes perpendicular to l
    fmom = (kfmax - kfmin) / (lEnd - lStart)
    fh = (hEnd - hStart) / (lEnd - lStart)
    fk = (kEnd - kStart) / (lEnd - lStart)
    for i = lStartIdx:lEndIdx
        li = lx[i]
        hi = fh * (li - lStart) + hStart
        ki = fk * (li - lStart) + kStart
        if (hi >= hx[1]) && (hi <= hx[hNPts]) && (ki >= kx[1]) && (ki <= kx[kNPts])
            momi = fmom * (li - lStart) + kfmin
            count += 1
        end
    end

    # endpoints
    if (hStart >= hx[1]) &&
       (hStart <= hx[hNPts]) &&
       (kStart >= kx[1]) &&
       (kStart <= kx[kNPts]) &&
       (lStart >= lx[1]) &&
       (lStart <= lx[lNPts])
        count += 1
    end
    if (hEnd >= hx[1]) &&
       (hEnd <= hx[hNPts]) &&
       (kEnd >= kx[1]) &&
       (kEnd <= kx[kNPts]) &&
       (lEnd >= lx[1]) &&
       (lEnd <= lx[lNPts])
        count += 1
    end

    return count
end

"""
    calculateIntersections!() -> Vector{Crd4}

Calculate the points of intersection for the given detector with cuboid
surrounding the detector position in HKL.

# Arguments
- `histogram`: umm...
- `theta`: Polar angle with detector
- `phi`: Azimuthal angle with detector
- `transform::SquareMatrix3c`: Matrix to convert frm Q_lab to HKL ``(2Pi*R *UB*W*SO)^{-1}``
- `lowvalue`: The lowest momentum or energy transfer for the trajectory
- `highvalue`: The highest momentum or energy transfer for the trajectory
"""
@propagate_inbounds function calculateIntersections!(
    # hx,
    # kx,
    # lx,
    histogram::THistogram,
    theta::CoordType,
    phi::CoordType,
    transform::SquareMatrix3c,
    lowvalue::CoordType,
    highvalue::CoordType,
    intersections::PreallocArrayRow{Crd4},
    # iPerm::PreallocArrayRow{SizeType},
) where {THistogram}
    sin_theta = sin(theta)
    qout = C3[sin_theta * cos(phi), sin_theta * sin(phi), cos(theta)]
    qin = C3[0, 0, 1]

    qout = transform * qout
    qin = transform * qin

    kimin = lowvalue
    kimax = highvalue
    kfmin = kimin
    kfmax = kimax

    hx = edges(histogram)[1]
    kx = edges(histogram)[2]
    lx = edges(histogram)[3]

    hNPts = length(hx)
    kNPts = length(kx)
    lNPts = length(lx)

    hStart = qin[1] * kimin - qout[1] * kfmin
    hEnd = qin[1] * kimax - qout[1] * kfmax
    hStartIdx = binindex1d(histogram, 1, hStart)
    hEndIdx = binindex1d(histogram, 1, hEnd)
    hStartIdx = clamp(hStartIdx, 0, hNPts)
    hEndIdx = clamp(hEndIdx, 0, hNPts)
    if hStartIdx > hEndIdx
        hStartIdx, hEndIdx = (hEndIdx, hStartIdx)
    end
    hStartIdx += 1

    kStart = qin[2] * kimin - qout[2] * kfmin
    kEnd = qin[2] * kimax - qout[2] * kfmax
    kStartIdx = binindex(histogram, 0, kStart, 0)[2]
    kEndIdx = binindex(histogram, 0, kEnd, 0)[2]
    kStartIdx = clamp(kStartIdx, 0, kNPts)
    kEndIdx = clamp(kEndIdx, 0, kNPts)
    if kStartIdx > kEndIdx
        kStartIdx, kEndIdx = (kEndIdx, kStartIdx)
    end
    kStartIdx += 1

    lStart = qin[3] * kimin - qout[3] * kfmin
    lEnd = qin[3] * kimax - qout[3] * kfmax
    lStartIdx = binindex(histogram, 0, 0, lStart)[3]
    lEndIdx = binindex(histogram, 0, 0, lEnd)[3]
    lStartIdx = clamp(lStartIdx, 0, lNPts)
    lEndIdx = clamp(lEndIdx, 0, lNPts)
    if lStartIdx > lEndIdx
        lStartIdx, lEndIdx = (lEndIdx, lStartIdx)
    end
    lStartIdx += 1

    empty!(intersections)

    # calculate intersections with planes perpendicular to h
    fmom = (kfmax - kfmin) / (hEnd - hStart)
    fk = (kEnd - kStart) / (hEnd - hStart)
    fl = (lEnd - lStart) / (hEnd - hStart)
    for i = hStartIdx:hEndIdx
        hi = hx[i]
        # if hi is between hStart and hEnd, then ki and li will be between
        # kStart, kEnd and lStart, lEnd and momi will be between kfmin and
        # kfmax
        ki = fk * (hi - hStart) + kStart
        li = fl * (hi - hStart) + lStart
        if (ki >= kx[1]) && (ki <= kx[kNPts]) && (li >= lx[1]) && (li <= lx[lNPts])
            momi = fmom * (hi - hStart) + kfmin
            push!(intersections, C4[hi, ki, li, momi])
        end
    end

    # calculate intersections with planes perpendicular to k
    fmom = (kfmax - kfmin) / (kEnd - kStart)
    fh = (hEnd - hStart) / (kEnd - kStart)
    fl = (lEnd - lStart) / (kEnd - kStart)
    for i = kStartIdx:kEndIdx
        ki = kx[i]
        # if ki is between kStart and kEnd, then hi and li will be between
        # hStart, hEnd and lStart, lEnd and momi will be between kfmin and
        # kfmax
        hi = fh * (ki - kStart) + hStart
        li = fl * (ki - kStart) + lStart
        if (hi >= hx[1]) && (hi <= hx[hNPts]) && (li >= lx[1]) && (li <= lx[lNPts])
            momi = fmom * (ki - kStart) + kfmin
            push!(intersections, C4[hi, ki, li, momi])
        end
    end

    # calculate intersections with planes perpendicular to l
    fmom = (kfmax - kfmin) / (lEnd - lStart)
    fh = (hEnd - hStart) / (lEnd - lStart)
    fk = (kEnd - kStart) / (lEnd - lStart)
    for i = lStartIdx:lEndIdx
        li = lx[i]
        hi = fh * (li - lStart) + hStart
        ki = fk * (li - lStart) + kStart
        if (hi >= hx[1]) && (hi <= hx[hNPts]) && (ki >= kx[1]) && (ki <= kx[kNPts])
            momi = fmom * (li - lStart) + kfmin
            push!(intersections, C4[hi, ki, li, momi])
        end
    end

    # endpoints
    if (hStart >= hx[1]) &&
       (hStart <= hx[hNPts]) &&
       (kStart >= kx[1]) &&
       (kStart <= kx[kNPts]) &&
       (lStart >= lx[1]) &&
       (lStart <= lx[lNPts])
        push!(intersections, C4[hStart, kStart, lStart, kfmin])
    end
    if (hEnd >= hx[1]) &&
       (hEnd <= hx[hNPts]) &&
       (kEnd >= kx[1]) &&
       (kEnd <= kx[kNPts]) &&
       (lEnd >= lx[1]) &&
       (lEnd <= lx[lNPts])
        push!(intersections, C4[hEnd, kEnd, lEnd, kfmax])
    end

    # sort intersections by final momentum
    # resize!(iPerm, length(intersections))
    # sortperm!(iPerm, intersections, lt = (v1, v2) -> v1[4] < v2[4])
    # bubbleSortPerm!(iPerm, intersections, lt = (v1, v2) -> v1[4] < v2[4])
    # bubbleSort!(intersections, lt = (v1, v2) -> v1[4] < v2[4])
    # cocktailSort!(intersections, lt = (v1, v2) -> v1[4] < v2[4])
    combSort!(intersections, lt = (v1, v2) -> v1[4] < v2[4])

    # # TODO: sort on construction ?
    # return SortedPreallocVector(iPerm, intersections)
end

# @propagate_inbounds function calculateIntersections!(
#     mdn::MDNorm,
#     histogram::THistogram,
#     theta::CoordType,
#     phi::CoordType,
#     transform::SquareMatrix3c,
#     lowvalue::CoordType,
#     highvalue::CoordType,
#     intersections::PreallocVector{Crd4},
#     iPerm::PreallocVector{SizeType},
# ) where {THistogram}
#     calculateIntersections!(
#         mdn.hX,
#         mdn.kX,
#         mdn.lX,
#         histogram,
#         theta,
#         phi,
#         transform,
#         lowvalue,
#         highvalue,
#         intersections,
#         iPerm,
#     )
# end

# @propagate_inbounds function calculateIntersections(
#     mdn::MDNorm,
#     histogram::THistogram,
#     theta::CoordType,
#     phi::CoordType,
#     transform::SquareMatrix3c,
#     lowvalue::CoordType,
#     highvalue::CoordType,
# ) where {THistogram}
#     intersections = PreallocVector(Vector{Crd4}(undef, maxIntersections(mdn)))
#     return calculateIntersections!(
#         mdn,
#         histogram,
#         theta,
#         phi,
#         transform,
#         lowvalue,
#         highvalue,
#         intersections,
#     )
# end


"""
Calculate the normalization among intersections on a single detector in 1
specific SpectrumInfo/ExperimentInfo.

# Arguments
- `intersections::Vector{Crd4}`: intersections
- `solid::ScalarType`: proton charge
- `yValues::Vector{ScalarType}`: diffraction intersection integral and common to sample and background
"""
@propagate_inbounds function calculateSingleDetectorNorm!(
    # intersections::SortedPreallocVector{Crd4},
    intersections::PreallocArrayRow{Crd4},
    solid::ScalarType,
    yValues::PreallocArrayRow{ScalarType},
    histogram::THistogram,
) where {THistogram}
    for i = 2:length(intersections)
        curX = intersections[i]
        prevX = intersections[i - 1]

        # The full vector isn't used so compute only what is necessary
        # If the difference between 2 adjacent intersection is trivial, no
        # intersection normalization is to be calculated
        # diffraction
        delta = curX[4] - prevX[4]
        eps = 1.0e-7
        if delta < eps
            continue # Assume zero contribution if difference is small
        end

        # Average between two intersections for final position
        # [Task 89] Sample and background have same 'pos[]'
        pos = C3[
            0.5 * (curX[1] + prevX[1]),
            0.5 * (curX[2] + prevX[2]),
            0.5 * (curX[3] + prevX[3]),
        ]

        # Diffraction
        # signal = integral between two consecutive intersections
        signal = (yValues[i] - yValues[i - 1]) * solid

        atomic_push!(histogram, pos[1], pos[2], pos[3], signal)
    end
end

struct XValues
    ix::AbstractVector{Crd4}
end

@propagate_inbounds Base.getindex(xValues::XValues, i::Integer) = xValues.ix[i][4]
@inline Base.length(xValues::XValues) = length(xValues.ix)

"""
Linearly interpolate between the points in integrFlux at xValues and save the
results in yValues.

# Arguments
- `xValues`: X-values at which to interpolate
- `integrFlux`: A workspace with the spectra to interpolate
- `sp::SizeType`: A workspace index for a spectrum in integrFlux to interpolate.
"""
@propagate_inbounds function calculateIntegralsForIntersections!(
    xValues::XValues,
    integrFlux_x::AbstractRange{ScalarType},
    integrFlux_y::AbstractVector{ScalarType},
    yValues::PreallocArrayRow{ScalarType},
)
    # the x-data from the workspace
    xData = integrFlux_x
    xStart = first(xData)
    xEnd = last(xData)

    # the values in integrFlux are expected to be integrals of a non-negative
    # function
    # ie they must make a non-decreasing function
    yData = integrFlux_y
    yMin = 0.0
    yMax = last(yData)

    nData = length(xValues)

    # all integrals below xStart must be 0
    if xValues[nData] < xStart
        fill!(yValues, yMin, nData)
        return yValues
    end

    # all integrals above xEnd must be equal to yMax
    if xValues[1] > xEnd
        fill!(yValues, yMax, nData)
        return yValues
    end

    fill!(yValues, 0.0, nData)

    i = 1
    # integrals below xStart must be 0
    while i < nData && xValues[i] <= xStart
        yValues[i] = yMin
        i += 1
    end

    iMax = nData
    # integrals above xEnd must be yMax
    while iMax > i && xValues[iMax] >= xEnd
        yValues[iMax] = yMax
        iMax -= 1
    end

    j = 2
    while i <= iMax
        xi = xValues[i]
        while xi > xData[j]
            j += 1
        end
        # interpolate between the consecutive points
        x0 = xData[j - 1]
        x1 = xData[j]
        y0 = yData[j - 1]
        y1 = yData[j]
        yValues[i] = lerp(y0, y1, (xi - x0) / (x1 - x0))
        i += 1
    end

    return yValues
end

@propagate_inbounds function calculateDiffractionIntersectionIntegral!(
    # intersections::SortedPreallocVector{Crd4},
    intersections::PreallocArrayRow{Crd4},
    integrFlux_x::AbstractRange{ScalarType},
    integrFlux_y::AbstractVector{ScalarType},
    yValues::PreallocArrayRow{ScalarType},
)
    return calculateIntegralsForIntersections!(
        XValues(intersections),
        integrFlux_x,
        integrFlux_y,
        yValues,
    )
end

"""
Calculate the diffraction MDE's intersection integral of a certain
detector/spectru

# Arguments
- `intersections::Vector{Crd4}`: vector of intersections
- `integrFlux`: integral flux workspace
- `wsIdx::SizeType`: workspace index
"""
# @propagate_inbounds function calculateDiffractionIntersectionIntegral(
#     intersections::PreallocVector{Crd4},
#     integrFlux_x::AbstractRange{ScalarType},
#     integrFlux_y::Vector{ScalarType},
# )
#     yValues = PreallocVector(Vector{ScalarType}(undef, length(intersections)))
#     calculateDiffractionIntersectionIntegral!(
#         intersections,
#         integrFlux_x,
#         integrFlux_y,
#         yValues,
#     )
#     return yValues
# end


@inline function mdNorm!(
    signal::Hist3,
    mdn::MDNorm,
    saData::SolidAngleData,
    fluxData::FluxData,
    eventData::EventData,
    transforms::Array1{SquareMatrix3c},
)
    for n = 1:length(transforms)
        maxIx = JACC.parallel_reduce(
            fluxData.ndets,
            max,
            (i, t) -> begin
                @inbounds begin
                    if t.skip_dets[i]
                        return 0
                    end

                    detID = i ### DELETEME
                    if detID > length(t.fluxDetToIdx)
                        return 0
                    end
                    wsIdx = t.fluxDetToIdx[detID]

                    return countIntersections(
                        t.signal,
                        t.thetaValues[i],
                        t.phiValues[i],
                        t.transforms[t.n],
                        t.lowValues[i],
                        t.highValues[i],
                    )
                end
            end,
            (
                transforms = transforms,
                n,
                skip_dets = mdn.extrasData.skip_dets,
                signal,
                eventData.thetaValues,
                eventData.phiValues,
                eventData.lowValues,
                eventData.highValues,
                fluxData.fluxDetToIdx,
            ),
            init = typemin(SizeType)
        )

        if maxIx == 0
            continue
        end

        if maxIx > rowSize(mdn.intersections)
            reset!(mdn.intersections)
            reset!(mdn.yValues)
            mdn.intersections = PreallocJaggedArray{Crd4}(fluxData.ndets, maxIx)
            mdn.yValues = PreallocJaggedArray{ScalarType}(fluxData.ndets, maxIx)
        end

        JACC.parallel_for(
            fluxData.ndets,
            (i, t) -> begin
                @inbounds begin
                    if t.skip_dets[i]
                        return nothing
                    end

                    # detID = t.detIDs[i]
                    # wsIdx = get(t.fluxDetToIdx, detID, nothing)
                    # if wsIdx == nothing
                    #     return nothing
                    # end
                    detID = i ### DELETEME
                    if detID > length(t.fluxDetToIdx)
                        return nothing
                    end
                    wsIdx = t.fluxDetToIdx[detID]

                    intersections = row(t.intersections, i)
                    # sortedIntersections = calculateIntersections!(
                    calculateIntersections!(
                        t.signal,
                        t.thetaValues[i],
                        t.phiValues[i],
                        t.transforms[t.n],
                        t.lowValues[i],
                        t.highValues[i],
                        intersections,
                    )

                    # if isempty(sortedIntersections)
                    if isempty(intersections)
                        return nothing
                    end

                    yValues = row(t.yValues, i)
                    calculateDiffractionIntersectionIntegral!(
                        intersections,
                        # sortedIntersections,
                        t.integrFlux_x,
                        view(t.integrFlux_y,:,wsIdx),
                        yValues,
                    )

                    saIdx = t.solidAngDetToIdx[detID]
                    saFactor = t.solidAngleValues[saIdx]
                    solid::ScalarType = t.protonCharge * saFactor

                    calculateSingleDetectorNorm!(
                        intersections,
                        # sortedIntersections,
                        solid,
                        yValues,
                        t.signal,
                    )

                    return nothing
                end
            end,
            (
                transforms = transforms,
                n,
                skip_dets = mdn.extrasData.skip_dets,
                intersections = mdn.intersections,
                # mdn.iPerm,
                yValues = mdn.yValues,
                signal,
                eventData.thetaValues,
                eventData.phiValues,
                eventData.lowValues,
                eventData.highValues,
                fluxData.fluxDetToIdx,
                fluxData.integrFlux_x,
                fluxData.integrFlux_y,
                saData.solidAngDetToIdx,
                saData.solidAngleValues,
                eventData.protonCharge,
            ),
        )
    end

    return signal
end
