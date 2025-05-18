import numpy as np
import matplotlib.pyplot as plt
from deap import base, benchmarks, \
        cma, creator, tools

# Função para criar uma toolbox
def create_toolbox(strategy):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    #Registrar a função de avaliação (função de otimização Rastrigin)
    toolbox.register("evaluate", benchmarks.rastrigin)

    # Configura a geração de números randômicos
    np.random.seed(7)

    #Registra as função de geração e atualizção
    #O método geração-atualização gera a população com base em uma estratégia e a atualiza com base na população
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox

if __name__ == "__main__":
    # Tamanho do "problema"
    num_individuals = 10
    num_generations = 125

    # Cria uma estratégia usando o algoritmo CMA-ES antes de iniciar o processo
    #Covariance Matrix Adaptation Evolution Strategy
    strategy = cma.Strategy(centroid=[5.0]*num_individuals, sigma=5.0, 
            lambda_=20*num_individuals)

    # Cria uma toolbox baseada na estratégia definida acima
    toolbox = create_toolbox(strategy)

    # Cria o objeto "hall da fama"
    #Ele armazena os melhores indivíduos (ordenados). O primeiro da lista é sempre o "melhor" indivíduo do processo
    hall_of_fame = tools.HallOfFame(1)

    # Registra as estatísticas relevantes
    stats = tools.Statistics(lambda x: x.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #lista (dicionário) com a cronologia da evolução
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    # Objetos que irão compilar os dados
    sigma = np.ndarray((num_generations, 1))
    axis_ratio = np.ndarray((num_generations, 1))
    diagD = np.ndarray((num_generations, num_individuals))
    fbest = np.ndarray((num_generations,1))
    best = np.ndarray((num_generations, num_individuals))
    std = np.ndarray((num_generations, num_individuals))

    for gen in range(num_generations):
        # Gerar uma populaçõ nova
        population = toolbox.generate()

        # Avaliar os indivíduos
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Atualizar a estratégia com os indivíduos avaliados
        toolbox.update(population)
        
        # Atualizar o hall da fama e as estatísticas
        # com a população atualmente avaliada
        hall_of_fame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)
        
        print(logbook.stream)
        
        # Salvar os dados ao longo da avaliação
        sigma[gen] = strategy.sigma
        axis_ratio[gen] = max(strategy.diagD)**2/min(strategy.diagD)**2
        diagD[gen, :num_individuals] = strategy.diagD**2
        fbest[gen] = hall_of_fame[0].fitness.values
        best[gen, :num_individuals] = hall_of_fame[0]
        std[gen, :num_individuals] = np.std(population, axis=0)

    # O eixo x será o número de avaliações
    x = list(range(0, strategy.lambda_ * num_generations, strategy.lambda_))
    avg, max_, min_ = logbook.select("avg", "max", "min")
    plt.figure()
    plt.semilogy(x, avg, "--b")
    plt.semilogy(x, max_, "--b")
    plt.semilogy(x, min_, "-b")
    plt.semilogy(x, fbest, "-c")
    plt.semilogy(x, sigma, "-g")
    plt.semilogy(x, axis_ratio, "-r")
    plt.grid(True)
    plt.title("azul: f-values, verde: sigma, vermelho: razão do eixo")

    plt.figure()
    plt.plot(x, best)
    plt.grid(True)
    plt.title("Variáveis objetos")

    plt.figure()
    plt.semilogy(x, diagD)
    plt.grid(True)
    plt.title("Dimensionamento (Eixos principais)")

    plt.figure()
    plt.semilogy(x, std)
    plt.grid(True)
    plt.title("Desvio padrão em todas as coordenadas")
    
    plt.show()

