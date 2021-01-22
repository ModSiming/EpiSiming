using StatsBase

struct Square
    popSize::Integer                               # numero de pessoas naquele bairro
    people::Array{T} where T <: Integer            # indice das pessoas do bairro
    distances::Array{T} where T <: Number          # distancia media entre uma pessoa do bairro e os demais bairros
end

struct Particle
    index::Integer                                  # indice da particula na rede
    popSize::Integer                                # numero de pessoas naquela particula
    people::Array{T} where T <: Integer             # indice das pessoas da particula
    square::Integer                                 # indice do bairro que a particula pertence
    position::Array{T} where T <: Number            # posicao da particula no mapa
    θᵢ::Number                                      # funcao para gerar a taxa de transmissao dos infectados
    θₐ::Number                                      # funcao para gerar a taxa de transmissao dos assintomaticos
end

struct Network
    name::AbstractString                            # nome da rede
    particles::Array{Particle}                      # particulas que compoem a rede
end

struct Population
    folder::AbstractString                             # n da rodada (nome da pasta com arquivos)
    popSize::Integer                                   # tamanho da população
    initialState::Array{T} where T <: Integer          # estado inicial para cada individuo
    initialTransitions::Array{T, 2} where T <: Number
    ρ::Array{T} where T <: Number                      # infecciosidade de cada pessoa
    positions::Array{T, 2} where T <: Number           # posicao de cada pessoa
    networks::Array{Network}                           # lista com todas as redes
    squares::Array{Square}                             # lista de bairros
    viralCharge::Array{T, 2} where T <: Number         # cargaViral
    preInfectionPeriod::Array{T} where T <: Number
    asymptomatic::Array{T} where T <: Number
end

function selectiveRand(v)
    """
        Função para geração de números aleatórios de acordo com um vetor booleano
    
        Parametros:
            v: vetor booleano
        
        Saida:
            aux: vetor com um número aleatório nas entradas verdadeiras de v o 0 nas demais
    """
    aux = zeros(length(v))
    aux[v] .= rand(sum(v))
    return aux
end

function rowWiseNorm(A)
    """
        Função para calcular a norma dos vetores linha de uma matriz
    """
    return sqrt.(sum(abs2, A, dims=2)[:, 1])
end

function calcGlobalContacts(
    population::Population, 
    infectionDay::Array{T} where T <: Integer, 
    t::Integer, 
    S, 
    I, 
    fKernel, 
    dist=True
)
    """
        Calculo da distância entre todas as pessoas, tomando a distnacia media entre pessoas de bairros diferentes
    """
    # the sum of current viral charge in each square
    ISquares = [(length(i.people[I[i.people]]) > 0) ? sum(population.ρ[k] .* population.viralCharge[k, t - infectionDay[k]] for k in i.people[I[i.people]]) : 0. for i in population.squares]
    
    # auxiliary variables
    aux = zeros(population.popSize)
    indexes = 1:length(population.squares)

    for (i, sq) in enumerate(population.squares)
        # sucetible and infected people in the square
        SSq = sq.people[S[sq.people]]
        ISq = sq.people[I[sq.people]]

        # if you want to calculate the actual distances between people inside the square
        if dist & length(ISq) > 0
            aux[SSq] .+= sum(ISquares .* sq.distances .* (indexes .!= i))
            
            rates = [population.ρ[k] .* population.viralCharge[k, t - infectionDay[k]] for k in ISq]
            for j in SSq
                aux[j] += sum(fKernel(population.positions[ISq, :] .- population.positions[j, :]') .* rates)
            end
        else
            aux[SSq] .+= sum(ISquares .* sq.distances)
        end
    end
    return aux[S]
end

function timeStep(
    population::Population, 
    currentState::Array{T} where T <: Number, 
    t::Number, 
    δ::Number, 
    transitions::Array{T, 2} where T <: Integer, 
    fKernel,
    θᵢ,
    θₐ, 
    dist=true
)
    """
        Entrada:
            populacao: 
            δ: tamanho do passo temporal
            fKernel: ?
    """
    Spop = currentState .== 1
    Epop = currentState .== 2
    Apop = currentState .== 3
    Ipop = currentState .== 4
    Rpop = currentState .== 5
    
    contacts = zeros(population.popSize)
    
    # first we calculate the contects in each network
    for i in population.networks
        for j in i.particles
            # contacts of the infected people in the particle
            Iparticle = j.people[Ipop[j.people]]
            Icontacts = 0.
            if length(Iparticle) > 0
                Icontacts += sum(population.ρ[k] .* population.viralCharge[k, t - transitions[k, 2]] for k in Iparticle)
            end
            
            # contacts of the asymptomatic people in the particle
            Aparticle = j.people[Apop[j.people]]
            Acontacts = 0.
            if length(Aparticle) > 0
                Acontacts += sum(population.ρ[k] .* population.viralCharge[k, t - transitions[k, 2]] for k in Aparticle)
            end
            # 
            contacts[j.people] .+= Icontacts * j.θᵢ
            contacts[j.people] .+= Acontacts * j.θₐ
        end
    end
    
    # now we calculate the contacts on the global network
    contacts[Spop] .+= calcGlobalContacts(population, transitions[:, 2], t, Spop, Ipop, fKernel, dist) .* θᵢ
    contacts[Spop] .+= calcGlobalContacts(population, transitions[:, 2], t, Spop, Apop, fKernel, dist) .* θₐ

    # determining the newly exposed population
    Eprob = exp.(- δ .* contacts)
    newE = selectiveRand(Spop) .> Eprob
    transitions[newE, 1] .= t

    # determining the newly infected/asymptomatic people
    incubationEnd = BitArray(zeros(population.popSize))
    incubationEnd[Epop] .= (t .- transitions[Epop, 1]) .>= population.preInfectionPeriod[Epop]
    transitions[incubationEnd, 2] .= t
    newA = incubationEnd .& population.asymptomatic
    newI = incubationEnd .& (.~newA)

    # determining the newly recovered population
    maxInfectionTime = size(population.viralCharge, 2)
    newR = BitArray(zeros(population.popSize))
    newR[Ipop] .= (t .- transitions[Ipop, 2]) .> maxInfectionTime
    newR[Apop] .= (t .- transitions[Apop, 2]) .> maxInfectionTime
    transitions[newR, 3] .= t

    # update all populations
    S = (Spop .& (.~newE))
    E = ((Epop .& (.~incubationEnd)) .| newE)
    I = ((Ipop .& (.~newR)) .| newI)
    A = ((Apop .& (.~newR)) .| newA)
    R = Rpop .| newR

    newState = ones(Int, population.popSize)
    newState[E] .= 2
    newState[A] .= 3
    newState[I] .= 4
    newState[R] .= 5
    return newState
end

function simulate(
    population::Population, times::AbstractArray{T} where T <: Number, fKernel, θᵢ, θₐ; dist=true)
    """
        Input:
            population:  variable with all required information about the population
            timeSteps:   array with the times to evaluate the infection
            fKernel:    
            θᵢ:          transmission rate of infected people in the global network
            θₐ:          transmission rate of asymptomatic people in the global network
            dist:        calculate the actual distance between people from the same square? (boolean)
        Output:
            transitions: array with the transition day for each individual as each transition (S -> E; E -> A/I; A/I -> R)
    """
    nT = length(times)
    timeSteps = times[2:end] - times[1:(end-1)]

    transitions = copy(population.initialTransitions)
    currentState = copy(population.initialState)

    for (t, δ) in zip(times[1:(end-1)], timeSteps)
        currentState .= timeStep(population, currentState, t, δ, transitions, fKernel, θᵢ, θₐ, dist)
    end
    return transitions
end

function map2matriz(a)
    nSim = length(a)
    transitions = zeros(nSim, size(a[1])...)
    
    for (i, j) in enumerate(a)
        transitions[i, :, :] .= j
    end
    return transitions
end

macro parallelSimulation(population, times, fKernel, θᵢ, θₐ, nSim, dist=true)
    return :(
        map2matriz(pmap((i) -> simulate($population, $times, $fKernel, $θᵢ, $θₐ, dist=$dist), 1:$nSim))
    )
end