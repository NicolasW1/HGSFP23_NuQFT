using Plots
using BenchmarkTools
using DifferentialEquations

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
RG_time(t, params::ModelParameters) = RG_time(t, params.Λ)

function deriv_right(f, i, x)
    (f[i+1] - f[i])/(x[i+1] - x[i])
end

function deriv_left(f, i, x)
    (f[i-1] - f[i])/(x[i-1] - x[i])
end

function flow_kernel!(du, u, t, params::ModelParameters)
    # Only works for d = 3
    k = RG_scale(t, params)
    Ad = 1/(6 * π^2)
    Ngrid = length(u)
    x = @view params.grid[:]
    d = 3

    du[1] = (Ad*k^(2 + d)*(1/(k^2 + u[1]) - 1/(k^2 + u[2] + (2*(u[1] - u[2])*x[2])/(x[1] - x[2]))))/(-x[1] + x[2])

    for i in 2:Ngrid-1
        du[i] = (Ad*k^(2 + d)*(1/(k^2 + u[i] + (2*(u[i-1] - u[i])*x[i])/(x[i-1] - x[i])) - 1/(k^2 + u[i+1] + (2*(u[i] - u[i+1])*x[i+1])/(x[i] - x[i+1]))))/(-x[i] + x[i+1])
    end

    du[Ngrid] = (Ad*k^(2 + d)*(-(1/(k^2 + u[Ngrid-1] + (2*(u[Ngrid-2] - u[Ngrid-1])*x[Ngrid-1])/(x[Ngrid-2] - x[Ngrid-1]))) + 1/(k^2 + u[Ngrid] + (2*(u[Ngrid-1] - u[Ngrid])*x[Ngrid])/(x[Ngrid-1] - x[Ngrid]))))/(x[Ngrid-1] - x[Ngrid])

    nothing
end

function generate_grid(params::NumericParameters)
    collect(range(zero(params.ρmax), params.ρmax, length=params.num_grid_pts))
end

function generate_log_grid(params::NumericParameters)
    log_rho_max = log10(params.ρmax)
    log_space_range = collect(range(1, log_rho_max, length=params.num_grid_pts))
    10 .^ log_space_range
end

function initial_values(params::ModelParameters)
    params.msqUV .+ params.λUV .* params.grid
end

function get_RG_time_span(params::NumericParameters)
    (zero(params.tmax), params.tmax)
end

function kernel_DiffEQ!(du, u, p, t)
    flow_kernel!(du, u, t, p)
end

num_params = NumericParameters(5., 7.5, 256)
model_params = ModelParameters(7.5, 3, -0.5, 1., generate_grid(num_params))

uinit = initial_values(model_params)
tspan = get_RG_time_span(num_params)

prob = ODEProblem(kernel_DiffEQ!, uinit, tspan, model_params)
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

plot(model_params.grid[1:150], [sol(t)[1:150] for t in 0:0.1:5], lw=3, legend=false)

plot(t->sol(t)[256], tspan[1], tspan[2])


########## BenchmarkTools
buffer_du = similar(uinit)

flow_kernel!(buffer_du, uinit, 1., model_params)

@benchmark flow_kernel!($buffer_du, $uinit, 5., $model_params)

@code_warntype flow_kernel!(buffer_du, uinit, 1., model_params)