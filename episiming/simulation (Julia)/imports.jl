using NPZ

function readFiles(scenario)
    folder = joinpath("data", scenario)

    output = Dict()

    for i in ["asymptomatic", "initialState", "initialTransition", "positions", "preInfectionPeriod", "rho", "squares", "viralCharge"]
        output[i] = npzread(joinpath(folder, string(i, ".eps")))
    end

    output["networks"] = Dict();

    for network in readdir(joinpath(folder, "networks"))
        aux = Dict()
        for i in ["peopleParticle", "position", "square", "thetaa", "thetai"]
            aux[i] = npzread(joinpath(folder, "networks", network, string(i, ".eps")))
        end
        output["networks"][network] = aux
    end
    return output
end;

function createSquares(input)
    squareNumbers = unique(input["squares"])
    nSquares = length(squareNumbers)

    squares = Array{Square}([])
    centers = zeros(nSquares, 2)

    for (i, j) in enumerate(squareNumbers)
        people = (1:length(input["asymptomatic"]))[input["squares"] .== j]
        centers[i, :] .= mean(input["positions"][people, :], dims=1)[1, :]
        push!(
            squares,
            Square(length(people), people, zeros(nSquares))
        )
    end

    for i in 1:nSquares
        squares[i].distances .= rowWiseNorm((centers[i, :] .- centers')')
    end
    return squares
end;

function createNetwork(input, networkName)
    network = input["networks"][networkName];

    particleNumbers = unique(network["peopleParticle"])
    particleNumbers = particleNumbers[particleNumbers .!= -1]
    sort!(particleNumbers)

    nParticles = length(particleNumbers)

    particles = Array{Particle}([])

    for (i, j) in enumerate(particleNumbers)
        people = (1:length(input["asymptomatic"]))[network["peopleParticle"] .== j]
        push!(
            particles,
            Particle(
                i,
                length(people),
                people,
                network["square"][i],
                network["position"][i, :],
                network["thetai"][i],
                network["thetaa"][i],
            )
        )
    end
    return Network(networkName, particles)
end;

function createNetworks(input)
    networks = Array{Network}([])
    for i in keys(input["networks"])
        push!(networks, createNetwork(input, i))
    end
    return networks
end

function readScenario(scenario)
    input = readFiles(scenario);
    squares = createSquares(input);
    networks = createNetworks(input);
    return Population(
        scenario,
        length(input["asymptomatic"]),
        input["initialState"],
        input["initialTransition"],
        input["rho"],
        input["positions"],
        networks,
        squares,
        input["viralCharge"],
        input["preInfectionPeriod"],
        input["asymptomatic"]
    )
end;