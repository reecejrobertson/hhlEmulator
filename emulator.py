import numpy as np
import matplotlib.pyplot as plt

def emulateHHL(A, b, shots=8192):
    '''
    input parameters :
    A : The matrix A for which we need to calculate the eigen values and eigen vectors
    b : the vector from the system of equeation Ax = b.
    lamda0_tilde  : The value of the first Eigen value.
    lamda1_tilde : The value for the second Eigen value.

    return parameters:
    count : Dictionary where key is the quantum states |00>, |01>, |10>, |11> 
            and the values are the corresponding counts, hence giving a probablity distribution
    '''

    # Get the solution to the system of linear equations
    evals, evecs = np.linalg.eig(A)

    beta = np.linalg.solve(evecs, b)

    c = min(evals)

    a = np.array([np.sqrt(1 - c**2/evals**2), c/evals])

    psi = np.kron(beta * evecs, a)

    psi = psi[:,1] + psi[:,2]

    # Normalise the amplitudes
    psi = psi /np.linalg.norm(psi)

    # Create a Dictionary for the measure count of the states |00>, |01>, |10>, |11>  
    measure_count = {'00': 0, '01': 0, '10': 0, '11': 0}

    for i in range(shots):
        r = np.random.rand()
        if r < psi[0]**2:
            measure_count['00'] += 1
        elif r < (psi[0]**2 + psi[1]**2):
            measure_count['01'] += 1
        elif r < (psi[0]**2 + psi[1]**2 + psi[2]**2):
            measure_count['10'] += 1
        else:
            measure_count['11'] += 1

    # Extracting keys and values from the dictionary
    x_values = list(measure_count.keys())
    y_values = list(measure_count.values())

    # Calculating probabilities
    total_counts = sum(y_values)
    probabilities = [count / total_counts for count in y_values]

    # Plotting the histogram with exact y-values on top of the bars
    fig, ax = plt.subplots()
    bars = ax.bar(x_values, probabilities, color='blue', alpha=0.7)

    # Adding exact y-values on top of the bars
    for bar, prob in zip(bars, probabilities):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')


    # Plotting the histogram
    # plt.bar(x_values, probabilities, color='blue', alpha=0.7)
    plt.xlabel('Measurement Result')
    plt.ylabel('Probability')
    plt.title('Histogram of Probabilities')
    plt.show()

    return np.array([measure_count['01'] / (measure_count['01'] + measure_count['11']), measure_count['11'] / (measure_count['01'] + measure_count['11'])])

A = [[1 , -1/3],[-1/3 , 1]]
b = [0, 1]
count = emulateHHL(A, b, 1000)
print(count)
total = np.abs(3./8.)**2 + np.abs(9./8.)**2
print(np.array([np.abs(3./8.)**2 / total, np.abs(9./8.)**2 / total]))
