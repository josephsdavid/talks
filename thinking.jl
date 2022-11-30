using Random
using OneHotArrays

softmax(x) = exp.(x) ./ sum(exp.(x); dims=1)

function make_y(n_samples=100, n_classes=5; seed=1)
    return softmax(100 .* rand(MersenneTwister(seed), n_samples, n_classes))
end
