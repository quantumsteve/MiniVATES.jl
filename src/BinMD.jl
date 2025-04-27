import Adapt

function binEvents!(h::Hist3, events::AbstractArray, weights::AbstractArray, transforms::Array1{SquareMatrix3c})
    JACC.parallel_for(
        size(events, 1),
        (i, t) -> begin
            @inbounds begin
                for n in 1:t.t_length
                    op = t.transforms[n]
                    v = op * C3[t.events[i, 1], t.events[i, 2], t.events[i, 3]]
                    atomic_push!(t.h, v[1], v[2], v[3], t.weights[i])
                end
            end
        end,
        (h, events, transforms, weights, t_length = length(transforms)),
    )
end

