# Getting to know Newton-Rhapson algorithm
# using ForwardDiff
using Calculus
# most basic form of newton-rhapson, 1D
let
    f(x) = (x-3)*(x+2)
    x0 = -3
    f(x0)
    f'(x0)
    n = 0
    err = abs(f(x0))

    while err > 1e-6
        x1 = x0 - f(x0)/f'(x0)
        err = abs(f(x1))
        x0 = x1
        n = n + 1
    end
    n
    x0
end
# dishonest form: use derivative f'(x0) during all iterations
# -> converges slowly but has reduced computational need

# now with a 2D function

X0 = [1,3]
f_a1 = x -> x[1]^3 + x[2]
f_a2 = x -> 2*x[1]^2 - 4*x[2]
F1 = Calculus.gradient(f_a1, X0)
F2 = Calculus.gradient(f_a2, X0)
J = transpose([F1 F2])

X1 = X0 - inv(J)*[f_a1(X0), f_a2(X0)]
f_a1(X1)
f_a2(X1)

F1_2 = Calculus.gradient(f_a1, X1)
F2_2 = Calculus.gradient(f_a2, X1)
J_2 = transpose([F1_2 F2_2])

X2 = X1 - inv(J_2)*[f_a1(X1), f_a2(X1)]
f_a1(X2)
f_a2(X2)

# ... build as loop

# How to define f as only one function?
function f!(F,x)
    F[1] = x[1] + x[2]
    F[2] = x[1] - 2*x[2]
end

function f_2d(x::Vector)
    F = [0,0]
    F[1] = x[1] + 3*x[2]
    F[2] = 2*x[1] - 9*x[2]
    F
end

# Power Flow example
x_km = 0.0175
θ_km = 10/360*2*pi
U_k = 0.984
U_m = 0.962

P_km = U_k * U_m * sin(θ_km) / x_km
Q_km = (U_k^2 - U_k * U_m * cos(θ_km)) / x_km
