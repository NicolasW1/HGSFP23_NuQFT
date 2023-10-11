using Plots
using BenchmarkTools

struct ModelParameters{T, I}
    Λ::T
    dim::I
    msqUV::T
    λUV::T
    grid::Vector{T}
end
struct NumericParameters{T, I}
    tmax::T
    ρmax::T
    num_grid_pts::I
end

function RG_scale(t, Λ)
    Λ * exp(-t)
end
function RG_scale(t, params::ModelParameters)
    RG_scale(t, params.Λ)
end

function RG_time(k, Λ)
    -log(k/Λ)
end

function deriv_right(f, i, x)
    (f[i+1] - f[i])/(x[i+1] - x[i])
end

function deriv_left(f, i, x)
    (f[i-1] - f[i])/(x[i-1] - x[i])
end

function flow_kernel(u, t, params::ModelParameters)
    # Only works for d = 3
    curvature_mass = similar(u)
    flux = similar(u)
    du = similar(u)

    k = RG_scale(t, params)
    Ad3 = 1/(6 * π^2)
    Ngrid = length(u)

    for i in 1:Ngrid
        if i == 1
            curvature_mass[1] = u[i]
        else
            curvature_mass[i] = u[i] + 2 * params.grid[i] * deriv_left(u, i, params.grid)
        end
    end

    for i in 1:Ngrid
        flux[i] = - Ad3 * k^(params.dim + 2) / (k^2 + curvature_mass[i])
    end

    for i in 1:Ngrid
        if i == Ngrid
            du[i] = deriv_left(flux, i, params.grid)
        else
            du[i] = deriv_right(flux, i, params.grid)
        end
    end

    flux
end

function generate_grid(params::NumericParameters)
    collect(range(zero(params.ρmax), params.ρmax, length=params.num_grid_pts))
end

function initial_values(params::ModelParameters)
    params.msqUV .+ params.λUV .* params.grid
end

num_params = NumericParameters(5., 7.5, 20)
model_params = ModelParameters(7.5, 3, 2.5, 1., generate_grid(num_params))

uinit = initial_values(model_params)

flow_kernel(uinit, 0., model_params)