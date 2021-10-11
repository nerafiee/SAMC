module SpeedupNeda

### load the modules and packages
using Random
using CSV
using DataFrames
using PyCall
using Dates

### declare constant variables
const X_AXIS_INIT = 1
const X_AXIS_END = 289 
const Y_AXIS_INIT = 1
const Y_AXIS_END = 193
const Z_AXIS_INIT = 1
const Z_AXIS_END = 257
###
const K1 = 12.0
const B1 = 15.0   ### 7.5 Angestrom
const K2 = 3.0
const B2 = 32.5   ### 16.25 Angestrom
###
const TEMPERATURE_MAX = 3000.0
###
const COARSE_CELL_LENGTH = 10
###
# const X_AXIS_OFFSET = 3
# const Y_AXIS_OFFSET = 1
# const Z_AXIS_OFFSET = 2
###
const SEARCH_RADIUS = 48 
###
const NUMBER_OF_STEPS = 10000
const NUMBER_OF_ACCEPTED_STEPS = 1000

### declare structs
struct LogData
    time::DateTime
    acceptance::Float64
    rejection::Float64
    temperature::Float64
    level::Int64
end

struct GridData
    lys::Array{Float64,3}
    gcp::Array{Float64,3}
    fine::Array{Float64,3}
    ### coarse::Array{Float64,3}
end

struct Point
    x::Int64
    y::Int64
    z::Int64
end

struct Offset
    x_axis::Int64
    y_axis::Int64
    z_axis::Int64
end

struct Boundary
    x_start::Int64
    x_end::Int64
    y_start::Int64
    y_end::Int64
    z_start::Int64
    z_end::Int64
end

### declare functions
function get_GridData()
    file_names = ["14-3-3-with-RAf-peptides_epi_Lys.dx", "14-3-3-with-RAf-peptides_epi_GCP.dx",
        "AIE_grid_with_Raf.dx"]
    gridData = pyimport("gridData")
    grids = map(x -> gridData.Grid(joinpath(@__DIR__, "..", "data", x)).grid, file_names)

    return GridData(grids...)
end # get_GridData


function get_distance(p1::Point, p2::Point)
    return sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)
end # get_distance


function get_harmonic(k, b, dist::Float64)
    return 0.5 * k * (dist - b)^2
end # get_harmonic


function has_self_overlap(p1, p2::Point)
    return (p1.x == p2.x) && (p1.y == p2.y) && (p1.z == p2.z)
end # has_self_overlap


function has_overlap_with_protein(p::Point, g::GridData)
    return g.fine[p.x, p.y, p.z] == 0.0
end # has_overlap_with_protein


function is_out_of_boundary(p::Point, b::Boundary)
    return (p.x < b.x_start) || (p.x > b.x_end) ||
    (p.y < b.y_start) || (p.y > b.y_end) ||
    (p.z < b.z_start) || (p.z > b.z_end)
end # is_out_of_boundary


function get_search_boundary(AIE_point::Point, radius::Int64)
    x_start = AIE_point.x - radius
    x_end = AIE_point.x + radius
    y_start = AIE_point.y - radius
    y_end = AIE_point.y + radius
    z_start = AIE_point.z - radius
    z_end = AIE_point.z + radius

    if x_start < X_AXIS_INIT
        x_start = X_AXIS_INIT
    end

    if x_end > X_AXIS_END
        x_end = X_AXIS_END
    end

    if y_start < Y_AXIS_INIT
        y_start = Y_AXIS_INIT
    end

    if y_end > Y_AXIS_END
        y_end = Y_AXIS_END
    end

    if z_start < Z_AXIS_INIT
        z_start = Z_AXIS_INIT
    end

    if z_end > Z_AXIS_END
        z_end = Z_AXIS_END
    end

    return Boundary(x_start, x_end, y_start, y_end, z_start, z_end)
end # get_search_boundary


function has_any_self_overlap(arr::Array)
    for i in 1 : length(arr)-1
        for j in i+1 : length(arr)
            if has_self_overlap(arr[i], arr[j])
                return true
            end
        end
    end
    return false
end # has_any_self_overlap


function has_any_overlap_with_protein(arr::Array, g::GridData)
    for i in 1:length(arr)
        if has_overlap_with_protein(arr[i], g)
            return true
        end
    end
    return false
end # has_any_overlap_with_protein


function get_initial_points(b::Boundary, p_AIE::Point, g::GridData)
    while true
        p_g1 = Point(rand(b.x_start:b.x_end), rand(b.y_start:b.y_end), rand(b.z_start:b.z_end))
        p_g2 = Point(rand(b.x_start:b.x_end), rand(b.y_start:b.y_end), rand(b.z_start:b.z_end))
        p_l1 = Point(rand(b.x_start:b.x_end), rand(b.y_start:b.y_end), rand(b.z_start:b.z_end))
        p_l2 = Point(rand(b.x_start:b.x_end), rand(b.y_start:b.y_end), rand(b.z_start:b.z_end))

        points = [p_g1, p_l1, p_AIE, p_l2, p_g2]
        if has_any_overlap_with_protein(points, g)
            continue
        end
        if has_any_self_overlap(points)
            continue
        end
        if has_repulsion(points)
            continue
        end
        return points
    end
end # get_initial_points


function get_distances(points::Array)
    dist1 = get_distance(points[1], points[2])
    dist2 = get_distance(points[2], points[3])
    dist3 = get_distance(points[3], points[4])
    dist4 = get_distance(points[4], points[5])
    return [dist1, dist2, dist3, dist4]
end # get_distances


function get_sum_harmonic(distances::Array)
    sum = 0
    sum += get_harmonic(K1, B1, distances[1])
    sum += get_harmonic(K2, B2, distances[2])
    sum += get_harmonic(K2, B2, distances[3])
    sum += get_harmonic(K1, B1, distances[4])
    return sum
end # get_sum_harmonic


function get_sum_grid_potentials(points::Array, g::GridData)
    gcp1 = points[1]
    lys1 = points[2]
    lys2 = points[4]
    gcp2 = points[5]
    return 2.47 * (g.gcp[gcp1.x, gcp1.y, gcp1.z] +
                   g.lys[lys1.x, lys1.y, lys1.z] +
                   g.lys[lys2.x, lys2.y, lys2.z] +
                   g.gcp[gcp2.x, gcp2.y, gcp2.z])
end # get_sum_grid_potentials


function get_random_number(from,to,except::Int64)
    while true
        r = rand(from : to)
        if r == except
            continue
        else
            return r
        end
    end
end # get_random_number


function get_population_density(grid_potentials)
    return exp(-(grid_potentials)/(300 * 8.31 * 0.001))
end # get_population_density


function has_repulsion(points::Array)
    if get_distance(points[1], points[3]) < 24.0 ||
        get_distance(points[1], points[4]) < 24.0 ||
        get_distance(points[1], points[5]) < 24.0 ||
        get_distance(points[2], points[4]) < 24.0 ||
        get_distance(points[2], points[5]) < 24.0 ||
        get_distance(points[3], points[5]) < 24.0 
        return true
    end
    return false
end # has_repulsion


function save_logs(logs::Array, path_out::String, p::Int64)
    f = open(joinpath(path_out, "log_$p.txt"), "w")
    foreach(l -> write(f, string("time: ", l.time, "\n",
                "acceptance: ", l.acceptance, "\n",
                "rejection: ", l.rejection, "\n",
                "temperature: ", l.temperature, "\n",
                "level: ", l.level, "\n\n",)) , logs)
    close(f)
end # save_logs


function get_traj_output_index(file_name_idx)
    idx_str = string(file_name_idx)
    result = ""
    for i in 1:6-length(idx_str)
        result = string(result,"0")
    end
    result = string(result, idx_str)

    return result
end


function monte_carlo(temperature::Float64, points::Array, boundary::Boundary,
    level::Int64, g::GridData, path_out, file_name_idx)

    delta_x = 6
    delta_y = 6 
    delta_z = 6 
    num_accepted_moves = 0

    moves = DataFrame(temperature = Float64[], harmonic = Float64[],
                    grid_potential = Float64[], population_density = Float64[],
                    x = Float64[], y = Float64[], z = Float64[])

    distances = get_distances(points)
    sum_harmonics = get_sum_harmonic(distances)
    sum_grid_potentials = get_sum_grid_potentials(points, g)
    population_density = get_population_density(sum_grid_potentials)
    foreach(p -> push!(moves, [temperature sum_harmonics sum_grid_potentials population_density p.x p.y p.z]), points)

    for i in 1:NUMBER_OF_STEPS
        temporary_points = deepcopy(points)
        r = get_random_number(1, 5, 3)

        while true
            rand_x = points[r].x + rand(-delta_x : delta_x)
        	rand_y = points[r].y + rand(-delta_y : delta_y)
            rand_z = points[r].z + rand(-delta_z : delta_z)
            p = Point(rand_x, rand_y, rand_z)

            if is_out_of_boundary(p, boundary)
                continue
            end
            if has_overlap_with_protein(p, g)
                continue
            end
            temporary_points[r] = p
            break
        end # while loop

        if has_repulsion(temporary_points)
            continue
        end

        distances_new = get_distances(temporary_points)
        sum_harmonics_new = get_sum_harmonic(distances_new)
        sum_grid_potentials_new = get_sum_grid_potentials(temporary_points, g)

        ###########
        delta_energy = sum_harmonics_new - sum_harmonics + sum_grid_potentials_new - sum_grid_potentials
        ###########

        if delta_energy < 0.0 || rand() < exp(-(delta_energy)/(temperature * 8.31 * 0.001))
            points = temporary_points
            sum_harmonics = sum_harmonics_new
            sum_grid_potentials  = sum_grid_potentials_new
            num_accepted_moves += 1
            population_density = get_population_density(sum_grid_potentials)
        end

        foreach(p -> push!(moves, [temperature sum_harmonics sum_grid_potentials population_density p.x p.y p.z]), points)

        if num_accepted_moves >= NUMBER_OF_ACCEPTED_STEPS
            idx_str = get_traj_output_index(file_name_idx)
            CSV.write(joinpath(path_out, "trajectory_$idx_str.csv"), moves, append=true)
            log = LogData(now(),
                    round(num_accepted_moves / i, digits=3),
                    round((i - num_accepted_moves) / i, digits=3),
                    round(temperature, digits=3),
                    level)
            return points, log
        end
    end # for loop

    idx_str = get_traj_output_index(file_name_idx)
    CSV.write(joinpath(path_out, "trajectory_$idx_str.csv"), moves, append=true)
    log = LogData(now(),
            round(num_accepted_moves / NUMBER_OF_STEPS, digits=3),
            round((NUMBER_OF_STEPS - num_accepted_moves) / NUMBER_OF_STEPS, digits=3),
            round(temperature, digits=3),
            level)
    return points, log
end # monte_carlo


function simmulated_annealing(temperature::Float64, points::Array, boundary::Boundary, g::GridData, path_out, file_name_idx)
    level = 0
    logs = LogData[]
    while temperature >= 10.0
        level += 1
        points, log = monte_carlo(temperature, points, boundary, level, g, path_out, file_name_idx)
        temperature *= 0.9
        push!(logs, log)
    end
    return logs
end # simmulated_annealing


#### ENTRY POINT ####
function run(output_dir::String, num_sims::Int64)
    if !isdir(output_dir)
        error("directory $output_dir does not exist.")
        return
    end

    g = get_GridData()
    csv_coarse_file = joinpath(@__DIR__, "..", "data", "AIE_Possible_Poses_with_Raf.csv")
    csv_file = CSV.read(csv_coarse_file, header = false)

    for p in 1:num_sims
        mkpath(joinpath(output_dir, "sim_$p"))
        path_out = joinpath(output_dir, "sim_$p")

        init_index = rand(1 : size(csv_file)[1])

        x = csv_file[init_index, 1] + 1
        y = csv_file[init_index, 2] + 1
        z = csv_file[init_index, 3] + 1
        AIE_point = Point(x, y, z)
	
        ### offset = Offset(X_AXIS_OFFSET, Y_AXIS_OFFSET, Z_AXIS_OFFSET)
        ### AIE_point = get_random_fine_grid_point_with_value_2(coarse_init_point, COARSE_CELL_LENGTH, offset, g)
        
	b = get_search_boundary(AIE_point, SEARCH_RADIUS)
        init_points = get_initial_points(b, AIE_point, g)

        logs = simmulated_annealing(TEMPERATURE_MAX, init_points, b, g, path_out, p)

        save_logs(logs, path_out, p)
    end
end # run

end # module
