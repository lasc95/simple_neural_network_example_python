# importamos la librería numpy
import numpy as np

# para la función de activación, utilizaré la función sigmoide, es común usarlas en casos donde sea un resultado binario (1 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# creamos la función de la derivada de sigmoide, la cual se utiliza en la retropropagación (backpropagation) para ajustar los pesos de la red
def sigmoid_derivate(x):
    return x * (1 - x)

# generamos semilla en numpy para nuestros procesos random
np.random.seed(1)

# generamos nuestros pesos, en este caso 2. Los pesos son aquellos valores que la red neuronal aprende durante el proceso.
# Se utilizan para calcular la salida de la red a partir de la entrada
weight0 = 2 * np.random.random((3, 4)) - 1
weight1 = 2 * np.random.random((4, 1)) - 1

# creamos nuestros datos de entrada y de salida. Son nuestros datos de prueba para que la red vaya aprendiendo. El input y el label
input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_data = np.array([[0, 1, 1, 0]]).T

# iteramos 1000 veces para poder entrenar nuestra red neuronal
for iteration in range(1000):
    # la capa 0 contendrá nuestros datos de entrada
    layer0 = input_data

    # nuestra red solo tiene una capa oculta, por lo general en redes más grandes, las capas ocultas son cientas o miles
    # aquí tomamos nuestra data de la primera capa y la multiplicamos por nuestro peso, a esto le aplicamos nuestra función de activación
    # le aplicamos la función para que nuestros valores sean 1 o 0, recordar eso.
    layer1 = sigmoid(np.dot(layer0, weight0))

    # Capa 2 es la capa de salida, tomamos la salida de la capa oculta (layer1) hacemos el mimsmo proceso con los pesos correspondientes y 
    # terminamos el proceso de "propagación hacia adelante"
    layer2 = sigmoid(np.dot(layer1, weight1))

    # calculo del error: es la diferencia entre las salidas esperadas (output_data) y las salidas de la red neuronal en entrenamiento. 
    # nos señala cuánto se desvió de lo esperado
    layer2_error = output_data - layer2

    if (iteration % 1000) == 0:
        print("Error:" + str(np.mean(np.abs(layer2_error))))

    # la capa delta el error de la capa de salida y se multiplica por la derivada de la función que utilizamos, en esta fue sigmoide.
    # este valor nos sirve para determinar cuánto debe ajustarse los pesos durante la retropropagación 
    layer2_delta = layer2_error * sigmoid_derivate(layer2)

    # obtener el error de la capa oculta nos dice cuánto determinó en el resultado final
    layer1_error = layer2_delta.dot(weight1.T)
    # el delta de la capa oculta nos dice cuánto debería ajustarse en la retropropagación
    layer1_delta = layer1_error * sigmoid_derivate(layer1)

    # se actualiza el peso del a capa de salida y de la capa oculta
    weight1 += layer1.T.dot(layer2_delta)
    weight0 += layer0.T.dot(layer1_delta)

