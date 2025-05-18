import copy
import random
from functools import partial

import numpy as np
from deap import algorithms, base, creator, tools, gp

#Classe controladora do robô
class RobotController(object):
    def __init__(self, max_moves):
        self.max_moves = max_moves
        self.moves = 0
        self.consumed = 0
        self.routine = None

        #Definição das direções e dos movimentos
        self.direction = ["north", "east", "south", "west"]
        self.direction_row = [1, 0, -1, 0]
        self.direction_col = [0, 1, 0, -1]
    
    #Funcionalidade de "reset"
    def _reset(self):
        self.row = self.row_start 
        self.col = self.col_start 
        self.direction = 1
        self.moves = 0  
        self.consumed = 0
        self.matrix_exc = copy.deepcopy(self.matrix)

    #Operador condicional
    def _conditional(self, condition, out1, out2):
        out1() if condition() else out2()

    #Virar à esquerda
    def turn_left(self): 
        if self.moves < self.max_moves:
            self.moves += 1
            self.direction = (self.direction - 1) % 4

    #Virar à direita
    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1    
            self.direction = (self.direction + 1) % 4
        
    #Seguir em frente
    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.row = (self.row + self.direction_row[self.direction]) % self.matrix_row
            self.col = (self.col + self.direction_col[self.direction]) % self.matrix_col

            #Ao encontrar um alvo, incrementa a variável
            if self.matrix_exc[self.row][self.col] == "target":
                self.consumed += 1

            self.matrix_exc[self.row][self.col] = "passed"

    #Ao enxergar um alvo, atualiza a matriz
    def sense_target(self):
        ahead_row = (self.row + self.direction_row[self.direction]) % self.matrix_row
        ahead_col = (self.col + self.direction_col[self.direction]) % self.matrix_col        
        return self.matrix_exc[ahead_row][ahead_col] == "target"
   
    #Identifica um alvo a sua frente
    def if_target_ahead(self, out1, out2):
        return partial(self._conditional, self.sense_target, out1, out2)
   
    #Método de execução
    def run(self,routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()
    
    # Define a função para atravessar o mapa
    # S é o ponto de partida
    # "#" são os alvos
    # . são células "vazias"
    def traverse_map(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())

            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("target")

                elif col == ".":
                    self.matrix[-1].append("empty")

                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.direction = 1

        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)

#Criar funções de acordo com o número de entradas
class Prog(object):
    def _progn(self, *args):
        for arg in args:
            arg()

    def prog2(self, out1, out2): 
        return partial(self._progn, out1, out2)

    def prog3(self, out1, out2, out3):     
        return partial(self._progn, out1, out2, out3)

#Função de avaliação
def eval_func(individual):
    global robot, pset

    # Transformar a expressão em árvore para código funcional em Python (programação genética)
    routine = gp.compile(individual, pset)

    # Executar a rotina gerada
    robot.run(routine)
    return robot.consumed,

def create_toolbox():
    global robot, pset

    #A programação genética visa construir programas de forma automática, independentemente do domínio
    #e procura resolvê-los. Uma estratégia comum é montar uma árvore com instruções que serão executadas
    
    #Cria um conjunto de primitivas
    pset = gp.PrimitiveSet("MAIN", 0)
    #Adiciona as primitivas (nós da árvore), com o nome da função que será executada e o número de argumentos
    pset.addPrimitive(robot.if_target_ahead, 2)
    pset.addPrimitive(Prog().prog2, 2)
    pset.addPrimitive(Prog().prog3, 3)
    #
    pset.addTerminal(robot.move_forward)
    pset.addTerminal(robot.turn_left)
    pset.addTerminal(robot.turn_right)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Atributo gerador
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

    # Estruturas de inicialização
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    return toolbox

if __name__ == "__main__":
    global robot

    # Inicializar a geração de números randômicos
    random.seed(7)

    # Definir o número máximo de movimentos
    max_moves = 750

    # Criar o objeto robô
    robot = RobotController(max_moves)

    # Criar a toolbox
    toolbox = create_toolbox()
    
    # Abrir o mapa de dados
    with open('target_map.txt', 'r') as f:
      robot.traverse_map(f)
    
    # Definir a população e o "hall da fama" de variáveis
    population = toolbox.population(n=400)
    hall_of_fame = tools.HallOfFame(1)

    # Registrar as estatísticas
    stats = tools.Statistics(lambda x: x.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Definir os parâmetros
    probab_crossover = 0.4
    probab_mutate = 0.3
    num_generations = 50
    
    # Executar o algoritmo genético para solucionar o problema
    algorithms.eaSimple(population, toolbox, probab_crossover, 
            probab_mutate, num_generations, stats, 
            halloffame=hall_of_fame)

