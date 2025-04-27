import StaticArrays: SMatrix, MVector
import Atomix

function binMD!(h::Hist3, events::AbstractArray, weights::AbstractArray, op)
    for n = 1:size(events, 1)
        v = op * C3[events[n, 1], events[n, 2], events[n, 3]]
    atomic_push!(h, v[1], v[2], v[3], weights[n])
    end
end

function binBoxes!(h::Hist3, events::FastEventData, transforms::AbstractArray)
    JACC.parallel_for(
        length(events.boxType),
        (i, t) -> begin
            @inbounds begin
                eid = t.eventIndex
                if t.boxType[i] == 1 && eid[i, 2] != 0
                    vi = SMatrix{3,8,CoordType}(t.extents[i, 1], t.extents[i, 3], t.extents[i, 5],
                                                t.extents[i, 1], t.extents[i, 3], t.extents[i, 6],
                                                t.extents[i, 1], t.extents[i, 4], t.extents[i, 5],
                                                t.extents[i, 1], t.extents[i, 4], t.extents[i, 6],
                                                t.extents[i, 2], t.extents[i, 3], t.extents[i, 5],
                                                t.extents[i, 2], t.extents[i, 3], t.extents[i, 6],
                                                t.extents[i, 2], t.extents[i, 4], t.extents[i, 5],
                                                t.extents[i, 2], t.extents[i, 4], t.extents[i, 6])
                    for op in t.transforms
                        vf = op * vi
                        startIdx = MVector{3,SizeType}(0, 0, 0)
                        singleBox = true
                        for j = 1:3
                            startIdx[j] = binindex1d(t.h, j, vf[j, 1])
                            if startIdx[j] < 1 || startIdx[j] > t.h.nbins[j]
                                singleBox = false
                                break
                            end
                            for k = 2:8
                                endIdx = binindex1d(t.h, j, vf[j, k])
                                if startIdx[j] != endIdx
                                    singleBox = false
                                    break
                                end
                            end
                            if singleBox == false
                                break
                            end
                        end
                        if singleBox
                            Atomix.@atomic binweights(t.h)[startIdx...] += t.boxSignal[i, 1]
                        else
                            startId = eid[i, 1] + 1
                            stopId = startId + eid[i, 2] - 1
                            localevents = @view t.events[startId:stopId, :]
                            localweights = @view t.weights[startId:stopId]
                            binMD!(t.h, localevents, localweights, op)
                        end
                    end
                end
            end
        end,
        (
            h = h,
            boxSignal = events.signal,
            events.boxType,
            events.events,
            events.weights,
            events.eventIndex,
            events.extents,
            transforms,
        ),
    )
end

