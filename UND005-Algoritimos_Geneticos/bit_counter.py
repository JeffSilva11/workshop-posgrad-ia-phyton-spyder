import random

from deap import base, creator, tools

# Função de avaliação
def eval_func(individual):
    target_sum = 45
    return len(individual) - abs(sum(individual) - target_sum),

# Criar a toolbox com os parâmetros corretos
def create_toolbox(num_bits):
    #Classes abstratas para as funções de avaliação (fitness) e os indivíduos
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Inicializar a toolbox, objeto que armazena as funções principais do AG
    toolbox = base.Toolbox()
    
    #REGISTRAR AS FUNÇÕES
    # registrar o gerador de números randômicos, número entre 0 e 1, usado para gerar as strings de bits (genes)
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Registrar a função "individual", usada para criar os cromossomos: tools.initRepeat
    toolbox.register("individual", tools.initRepeat, 
                     creator.Individual, #genótipo (container dos indivíduos)
                     toolbox.attr_bool,  #função usada para preencher o genótipo
                     num_bits)           #tamanho do cromossomo (número de vezes que a função será executada)

    # Registrar a função que define a população como uma lista de indivíduos
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #REGISTRAR OS OPERADORES GENÉTICOS
    # Registrar a função de avaliação (fitness)
    toolbox.register("evaluate", eval_func)

    # Registrar o operador para a recombinação (crossover)
    toolbox.register("mate", tools.cxTwoPoint)

    # Registrar o operador para a mutação (indpb refere-se a probabilidade de um indivíduo sofrer mutação)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # Registrar o operador para a seleção (tournsize representa o tamanho da "escolha")
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

if __name__ == "__main__":
    # Definir o número de bits
    num_bits = 75

    # Criar a toolbox usando o parâmetro definido acima
    toolbox = create_toolbox(num_bits)

    # Configurar a geração de números randômicos
    random.seed(7)

    # Criar uma população inicial de 500 indivíduos
    population = toolbox.population(n=500)

    # Definir as probabilidades de cruzamento e mutação (alterá-los muda o resultado)
    probab_crossing, probab_mutating  = 0.5, 0.2

    # Definir o número de gerações
    num_generations = 60
    
    print('\nIniciando o processo de evolução')
    
    # Avaliar a população inicial
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    print('\nAvaliados', len(population), 'indivíduos')
    
    # Iterar através das  gerações
    for g in range(num_generations):
        print("\n===== Geração", g)
        
        # Selecionar a pŕoxima geração de indivíduos
        offspring = toolbox.select(population, len(population))

        # Clonar os indivíduos selecionados
        offspring = list(map(toolbox.clone, offspring))
    
        # Aplicar o cruzamento e a mutação sobre os descendentes (offspring)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Cruzar dois indivíduos
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)

                # "resetar" os valores de aptidão para os filhos
                del child1.fitness.values
                del child2.fitness.values

        # Aplicar mutação
        for mutant in offspring:
            # Mutação de um indivíduo
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Avaliar os indivíduos com uma aptidão inválida
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print('Avaliados', len(invalid_ind), 'indivíduos')
        
        # População é substituída pelos seus descendentes
        population[:] = offspring
        
        # Reunir todos os aptos em uma lista e imprimir as estatísticas
        fits = [ind.fitness.values[0] for ind in population]
        
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print('Min =', min(fits), ', Max =', max(fits))
        print('Média =', round(mean, 2), ', Desvio padrão =', 
                round(std, 2))
    
    print("\n==== Fim da evolução")
    
    best_ind = tools.selBest(population, 1)[0]
    print('\nMelhor indivíduo:\n', best_ind)
    print('\nNumber of 1s:', sum(best_ind))
