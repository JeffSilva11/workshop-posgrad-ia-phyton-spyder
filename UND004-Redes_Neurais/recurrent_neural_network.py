import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

def get_data(num_points):
    # Criar formas de onda baseadas em seno
    wave_1 = 0.5 * np.sin(np.arange(0, num_points))
    wave_2 = 3.6 * np.sin(np.arange(0, num_points))
    wave_3 = 1.1 * np.sin(np.arange(0, num_points))
    wave_4 = 4.7 * np.sin(np.arange(0, num_points))

    # Crie amplitudes variadas
    amp_1 = np.ones(num_points)
    amp_2 = 2.1 + np.zeros(num_points) 
    amp_3 = 3.2 * np.ones(num_points) 
    amp_4 = 0.8 + np.zeros(num_points) 

    wave = np.array([wave_1, wave_2, wave_3, wave_4]).reshape(num_points * 4, 1)
    amp = np.array([[amp_1, amp_2, amp_3, amp_4]]).reshape(num_points * 4, 1)

    return wave, amp 

# Visualiza a saída
def visualize_output(nn, num_points_test):
    wave, amp = get_data(num_points_test)
    output = nn.sim(wave)
    plt.plot(amp.reshape(num_points_test * 4))
    plt.plot(output.reshape(num_points_test * 4))

if __name__=='__main__':
    # Cria um conjunto de dados de exemplo
    num_points = 40
    wave, amp = get_data(num_points)

    # Cria uma rede neural recorrente com duas camadas
    nn = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

    # Alterar os pesos iniciais e "bias"
    nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.init()

    # Treina a rede neural recorrente
    error_progress = nn.train(wave, amp, epochs=1200, show=100, goal=0.01)

    # Executa a rede neural sobre os dados de treinamento
    output = nn.sim(wave)

    # Plota os resultados
    plt.subplot(211)
    plt.plot(error_progress)
    plt.xlabel('Número de épocas')
    plt.ylabel('Erro (MSE)')

    plt.subplot(212)
    plt.plot(amp.reshape(num_points * 4))
    plt.plot(output.reshape(num_points * 4))
    plt.legend(['Original', 'Previsto'])

    # Testa o desempenho da rede sobre dados desconhecidos
    plt.figure()

    plt.subplot(211)
    visualize_output(nn, 80)
    plt.xlim([0, 300])

    plt.subplot(212)
    visualize_output(nn, 50)
    plt.xlim([0, 300])

    plt.show()
