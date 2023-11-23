import sys
sys.path.insert(0, '../intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import numpy as np
from matplotlib import pyplot as plt
import time

def computeQPERotations(A, t, numQubits):
    '''
    Compute the single qubit gates needed in the QPE algorithm.
    Parameters:
        A (np.ndarray)  : The A matrix.
        t (float)       : A variable used in the computation.
            For details, see https://arxiv.org/pdf/2108.09004.pdf eqs (17)-(21).
        numQubits (int) : The total number of qubits to use.
    Returns:
        A lsit of the QPE rotation matrices, and a list of the IQPE matrices.
    '''

    # Get the eigenvalues and eigenvectors and sort them by eigenvalue size.
    evals, evecs = np.linalg.eig(A)
    idx = evals.argsort() 
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the diagonal matrix hamiltonian.
    Udiag = np.diag(np.exp(1j * evals * t))

    # Create a list to hold the QPE and IQPE rotation matrices.
    qpeRotations = []
    qpeInvRotations = []

    # Populate the matrices with successive powers of the diagonal matrix.
    for i in range(numQubits-2):
        Ui = evecs.T @ np.linalg.matrix_power(Udiag, 2**i) @ evecs
        qpeRotations.append(Ui)
        qpeInvRotations.append(np.linalg.inv(Ui))

    # Return both lists.
    return qpeRotations, qpeInvRotations

def computeQFTRotations(numQubits):
    '''
    A helper function to get the QFT and IQFT rotation matrices.
    Parameters:
        numQubits (int) : The number of qubits involved in the QFT.
    Returns:
        The list of QFT rotation matrices, followed by that of IQFT matrices.
    '''
    return QFTRotationsHelper(numQubits), QFTRotationsHelper(numQubits, True)

def QFTRotationsHelper(numQubits, inverse=False):
    '''
    Computes the QFT (or IQFT) rotation matrices.
    Parameters:
        numQubits (int) : The number of qubits involved in the QFT.
        inverse (bool)  : If true, compute IQFT matrices, else QFT matrices.
    Returns:
        The list of QFT (or IQFT) rotation matrices.
    '''

    # Create a list to hold the matrices.
    rotations = []

    # Instantiate an inverse oscillating factor if necessary.
    sign = -1 if inverse else 1

    # Compute each matrix using the QFT formula and append it to the list.
    for n in range(2, numQubits-1):
        rotations.append(
            np.array(
                [[1, 0], [0, np.exp(sign*1j*2*np.pi/2**n)]],
                dtype=complex
            )
        )

    # Return the list.
    return rotations

def simulateIQFT(psi, numQubits, rotations):
    '''
    Simulate the Inverse Quantum Fourier Transform.
    Parameters:
        psi (QuantumRegister)   : A quantum state.
        numQubits (int)         : The number of qubits in the circuit.
        rotations (list)        : The list of matrices needed for the algoithm.
    Returns:
        The quantum state after the IQFT has been applied.
    '''

    # Apply a Hadamard gate and then the needed rotations to each qubit.
    for i in range(numQubits-2, 0, -1):
        psi.ApplyHadamard(i)
        for j in range(1, i):
            psi.ApplyControlled1QubitGate(j, i, rotations[i-j-1])

    # Apply swaps as needed.
    for i in range(1, numQubits//2):
        psi.ApplySwap(i, (numQubits-1)-i)

    # Return psi.
    return psi

def simulateQFT(psi, numQubits, invRotations):
    '''
    Simulate the Quantum Fourier Transform.
    Parameters:
        psi (QuantumRegister)   : A quantum state.
        numQubits (int)         : The number of qubits in the circuit.
        invRotations (list)     : The list of matrices needed for the algoithm.
    Returns:
        The quantum state after the IQFT has been applied.
    '''

    # Apply swaps as needed.
    for i in range(numQubits//2-1, 0, -1):
        psi.ApplySwap(i, (numQubits-1)-i)

    # Apply a Hadamard gate and then the needed rotations to each qubit.
    for i in range(1, numQubits-1):
        psi.ApplyHadamard(i)
        for j in range(numQubits-2, i, -1):
            psi.ApplyControlled1QubitGate(j, i, invRotations[j-i-1])

    # Return psi.
    return psi

def simulateQPE(psi, numQubits, qpeRotations, qftInvRotations):
    '''
    Simulate the Quantum Phase Estimation.
    Parameters:
        psi (QuantumRegister)   : A quantum state.
        numQubits (int)         : The number of qubits in the circuit.
        qpeRotations (list)     : The list of matrices needed for the algoithm.
        invRotations (list)     : The list of matrices needed for the IQFT.
    Returns:
        The quantum state after the QPE has been applied.
    '''

    # Apply a hadamard transformation to psi.
    for i in range(1, numQubits-1):
        psi.ApplyHadamard(i)

    # Apply the controlled-U**i operations to psi.
    for i in range(1, numQubits-1):
        psi.ApplyControlled1QubitGate(i, numQubits-1, qpeRotations[i-1])

    # Apply the IQFT to psi.
    psi = simulateIQFT(psi, numQubits, qftInvRotations)

    # Return psi.
    return psi

def simulateIQPE(psi, numQubits, qpeInvRotations, qftRotations):
    '''
    Simulate the Inverse Quantum Phase Estimation.
    Parameters:
        psi (QuantumRegister)   : A quantum state.
        numQubits (int)         : The number of qubits in the circuit.
        qpeInvRotations (list)  : The list of matrices needed for the algoithm.
        qftRotations (list)     : The list of matrices needed for the QFT.
    Returns:
        The quantum state after the IQPE has been applied.
    '''

    # Apply the QFT to psi.
    psi = simulateQFT(psi, numQubits, qftRotations)

    # Apply the inverse controlled-U**i operations to psi.
    for i in range(numQubits-2, 0, -1):
        psi.ApplyControlled1QubitGate(i, numQubits-1, qpeInvRotations[i-1])

    # Apply a hadamard transformation to psi.
    for i in range(1, numQubits-1):
        psi.ApplyHadamard(i)

    # Return psi.
    return psi

def measure(psi, target):
    '''
    Measure a qubit of a given quantum state.
    Parameters:
        psi (QuantumRegister)   : A quantum state.
        target (int)            : The index of the qubit to measure.
    Returns:
        The QuantumRegister post measurement, as well as the measurement result.
    '''

    # Compute the probability of qubit 0 being in state |1>.
    prob = psi.GetProbability(target)
    result = None

    # Draw random number in [0,1)
    r = np.random.rand()

    # If the random number is less than the probability, collapse to |1>.
    if r < prob:
        psi.CollapseQubit(target, True)
        result = '1'

    # Otherwise collapse to |0>.
    else:
        psi.CollapseQubit(target, False)
        result = '0'

    # In both cases we renormalize the wavefunction and return.
    psi.Normalize()
    return psi, result

def simulateHHL(A, b, applyRYs, t, numQubits=4, extraAncillas=0, shots=2048, figfile='simulator.png', dpi=300):
    '''
    A function that simulates the HHL algorithm using the process explained in
    section II of https://arxiv.org/pdf/2108.09004.pdf.
    Parameters:
        A (np.ndarray)      : The matrix A from the system Ax = b.
        b (np.ndarray)      : The vector b from the system Ax = b.
        applyRYs (func)     : A function to apply the controlled y rotations.
        t (float)           : A value needed for the QPE matrices.
        numQubits (int)     : The number of qubits used in the algorithm.
        extraAncillas (int) : The number of extra ancillas (used in applyRYs).
        shots (int)         : The number of shots to run.
        figfile (string)    : The file name to save the histogram under.
        dpi (int)           : The dpi of the histogram.
    Returns:
        An estimation of x as given by the HHL algorith.
        Also returns the time taken to perform the emulation.
        Finally saves a plot of the distribution generated by the algorithm.
    '''

    # Define the initial state of psi and get the QPE and QFT matrices.
    initialSate = 0
    qftGates, iqftGates = computeQFTRotations(numQubits)
    qpeGates, iqpeGates = computeQPERotations(A, t, numQubits)

    # Define a dictionary to hold the experimental results.
    probs = {'00': 0, '01': 0, '10': 0, '11': 0}

    # Compute the matrix to prepare the b state.
    bMatrix = np.array([[b[0], -b[1]], [b[1], b[0]]])

    # Note the time before starting the simulation proper.
    startTime = time.time()

    # For each shot:
    for i in range(shots):

        # Initialize a quantum system.
        psi = simulator.QubitRegister(numQubits + extraAncillas, 'base', initialSate, 0)

        # Intialize the target (last) qubit to the b state.
        psi.Apply1QubitGate(numQubits-1, bMatrix)

        # Apply the QPE to the target qubit (involving the c qubits).
        psi = simulateQPE(psi, numQubits, qpeGates, iqftGates)

        # Apply the controlled Y rotations to the anscilla (top) qubit.
        psi = applyRYs(psi)

        # Apply the IQPE to the target qubit.
        psi = simulateIQPE(psi, numQubits, iqpeGates, qftGates)

        # Measure the anscilla and target qubit, and track the result.
        psi, rightResult = measure(psi, target=0)
        psi, leftResult = measure(psi, target=numQubits-1)
        probs[leftResult + rightResult] += 1
    
    # Compute the total amount of time it took to perform the simulation.
    elapsedTime = time.time() - startTime

    # Compute the percentage of each measurement result.
    for key in probs:
        probs[key] /= shots

    # Plot the histogram with exact y-values on top of the bars.
    fig = plt.figure()
    plt.bar(range(len(probs)), list(probs.values()), align='center')
    plt.xticks(range(len(probs)), list(probs.keys()))
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(probs.values()):
        plt.text(xlocs[i] - 0.2, v + .007, str(round(v, 3)))
    plt.xlabel('Measurement Result')
    plt.ylabel('Probability')
    plt.title(
        'Simulation with A=['
        + np.array_str(A[0], precision=2,) + ','
        + np.array_str(A[1], precision=2) + '], b='
        + np.array_str(b, precision=2)
    )
    plt.savefig('plots/' + figfile, dpi=dpi)

    # Return the estimation of x.
    return np.array([
        probs['01'] / (probs['01'] + probs['11']),
        probs['11'] / (probs['01'] + probs['11'])
    ]), elapsedTime