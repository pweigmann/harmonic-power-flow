__precompile__()

module Functions

export yrx, ygb, kron_reduction, pf, smooth_step, get_run, get_batch

yrx(r, x; pu=1) = (1 / (r + im * x)) / pu

ygb(g, b; pu=1) = (g + im * b) / pu

kron_reduction(Y, passive) = Y[.~passive, .~passive] - Y[.~passive, passive] * inv(Array{Complex}(Y[passive, passive])) * Y[passive, .~passive]

pf(V, Y) = real(V .* conj(Y * V))

smooth_step(x, onset, duration) = (2 - tanh.(1000 * (x - onset)) + tanh.(1000 * (x - onset - duration))) / 2

smooth_step(x, onset, duration, h1, h2; factor=1000) = (1 - tanh.(factor * (x - onset))) * h1 / 2 + (1 + tanh.(factor * (x - onset - duration))) * h2 / 2

get_run(i, batch_size) = mod(i, batch_size)==0 ? batch_size : mod(i, batch_size)

get_batch(i, batch_size) = 1 + (i - 1) ÷ batch_size

end