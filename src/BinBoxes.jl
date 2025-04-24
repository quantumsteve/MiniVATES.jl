import StaticArrays: SMatrix, MVector

function binMD!(h::Hist3, events::AbstractArray, op)
    for n = 1:size(events, 1)
        v = op * C3[events[n, 1], events[n, 2], events[n, 3]]
        atomic_push!(h, v[1], v[2], v[3], one(CoordType))
    end
end

using CUDA
function binBoxes!(h::Hist3, events::FastEventData, transforms::AbstractArray)
    JACC.parallel_for(
        length(events.boxType),
        (i, t) -> begin
            @inbounds begin
                eid = t.eventIndex
                if t.boxType[i] == 1 && eid[i, 2] != 0
                    vi = transpose(SMatrix{2,3,CoordType}(@view t.extents[i, :]))
                    for op in t.transforms
                        vf = op * vi
                        startIdx = MVector{3,SizeType}(0, 0, 0)
                        singleBox = true
                        for j = 1:3
                            startIdx[j] = binindex1d(t.h, j, vf[j, 1])
                            endIdx = binindex1d(t.h, j, vf[j, 2])
                            if startIdx[j] != endIdx
                                singleBox = false
                                break
                            end
                        end
                        if singleBox
                            binweights(t.h)[startIdx...] += t.boxSignal[i, 1]
                        else
                            startId = eid[i, 1] + 1
                            stopId = startId + eid[i, 2] - 1
                            localevents = @view t.events[startId:stopId, :]
                            binMD!(t.h, localevents, op)
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
            events.eventIndex,
            events.extents,
            transforms,
        ),
    )
end

