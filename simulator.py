import sys
sys.path.insert(0, '../intel-qs/build/lib') # Change this to match your installation location.
import intelqs_py as simulator
import numpy as np
from matplotlib import pyplot as plt

# TODO: Validate that this scales properly
def computeQPERotations(A, numQubits):
    evals, evecs = np.linalg.eig(A)
    idx = evals.argsort() 
    evals = evals[idx]
    evecs = evecs[:, idx]
    lambdaHat = 1
    t = 2 * np.pi * lambdaHat / (2**(numQubits-2) * evals[0])
    Udiag = np.diag(np.exp(1j * evals * t))
    qpeRotations = []
    qpeInvRotations = []
    for i in range(numQubits-2):
        Ui = evecs.T @ np.linalg.matrix_power(Udiag, 2**i) @ evecs
        qpeRotations.append(Ui)
        qpeInvRotations.append(np.linalg.inv(Ui))
    return qpeRotations, qpeInvRotations

def computeQFTRotations(numQubits):
    return QFTRotationsHelper(numQubits), QFTRotationsHelper(numQubits, True)

def QFTRotationsHelper(numQubits, inverse=False):
    rotations = []
    sign = -1 if inverse else 1
    for n in range(2, numQubits-1):
        rotations.append(
            np.array(
                [[1, 0], [0, np.exp(sign*1j*2*np.pi/2**n)]],
                dtype=complex
            )
        )
    return rotations

def simulateIQFT(psi, numQubits, rotations):
    # print('BEGIN IQFT')
    for i in range(numQubits-2, 0, -1):
        psi.ApplyHadamard(i)
        # print('Hadamard on', i)
        for j in range(1, i):
            psi.ApplyControlled1QubitGate(j, i, rotations[i-j-1])
            # print('CRotation on', j, i)
            # print(rotations[i-j-1])
    for i in range(1, numQubits//2):
        psi.ApplySwap(i, (numQubits-1)-i)
    #     print('Swap on', i, (numQubits-1)-i)
    # print('END IQFT')
    return psi

def simulateQFT(psi, numQubits, invRotations):
    # print('BEGIN QFT')
    for i in range(numQubits//2-1, 0, -1):
        psi.ApplySwap(i, (numQubits-1)-i)
        # print('Swap on', i, (numQubits-1)-i)
    for i in range(1, numQubits-1):
        psi.ApplyHadamard(i)
        # print('Hadamard on', i)
        for j in range(numQubits-2, i, -1):
            psi.ApplyControlled1QubitGate(j, i, invRotations[j-i-1])
    #         print('CRotation on', j, i)
    #         print(invRotations[j-i-1])
    # print('END QFT')
    return psi

def simulateQPE(psi, numQubits, qpeRotations, qftInvRotations):
    # print('BEGIN QPE')
    for i in range(1, numQubits-1):
        psi.ApplyHadamard(i)
        # print('Hadamard on', i)

    # Apply controlled-U**i operations
    for i in range(1, numQubits-1):
        psi.ApplyControlled1QubitGate(i, numQubits-1, qpeRotations[i-1])
        # print('CRotation on', i, numQubits-1)
        # print(qpeRotations[i-1])

    # IQFT
    psi = simulateIQFT(psi, numQubits, qftInvRotations)
    
    # print('End QPE')
    return psi

def simulateIQPE(psi, numQubits, qpeInvRotations, qftRotations):
    # print('BEGIN IQPE')
    # IQFT
    psi = simulateQFT(psi, numQubits, qftRotations)

    # Apply controlled-U**i operations
    for i in range(numQubits-2, 0, -1):
        psi.ApplyControlled1QubitGate(i, numQubits-1, qpeInvRotations[i-1])
        # print('CRotaion on', i, numQubits-1)
        # print(qpeInvRotations[i-1])

    for i in range(1, numQubits-1):
        psi.ApplyHadamard(i)
    #     print('Hadamard on', i)
    
    # print('END IQPE')
    return psi

# TODO: FIXME
def simulateYRotations(psi, numQubits):
    for i in range(1, numQubits-1):
        psi.ApplyCRotationY(i, 0, 2*np.arcsin(1/2**(i-1)))
    return psi

def measure(psi, target):
    # Compute the probability of qubit 0 being in state |1>.
    prob = psi.GetProbability(target)
    result = None

    # Draw random number in [0,1)
    r = np.random.rand()
    if r < prob:
        # Collapse the wavefunction according to qubit 0 being in |1>.
        psi.CollapseQubit(target, True)
        result = '1'
    else:
        # Collapse the wavefunction according to qubit 0 being in |0>
        psi.CollapseQubit(target, False)
        result = '0'

    # In both cases one needs to re-normalize the wavefunction:
    psi.Normalize()

    return psi, result

def simulateHHL(A, b, numQubits=4, shots=1000):
    initialSate = 0
    qftGates, iqftGates = computeQFTRotations(numQubits)
    qpeGates, iqpeGates = computeQPERotations(A, numQubits)

    probs = {'00': 0, '01': 0, '10': 0, '11': 0}

    for i in range(shots):

        # Initialize a quantum system
        psi = simulator.QubitRegister(numQubits, 'base', initialSate, 0)

        # FIXME: This should use b!
        psi.ApplyPauliX(numQubits-1)

        psi = simulateQPE(psi, numQubits, qpeGates, iqftGates)

        # Controlled Y rotations
        # for i in range(1, numQubits-1):
        # psi = simulateYRotations(psi, numQubits)
        psi.ApplyCRotationY(1, 0, np.pi)
        psi.ApplyCRotationY(2, 0, np.pi/3.)

        psi = simulateIQPE(psi, numQubits, iqpeGates, qftGates)

        psi, rightResult = measure(psi, target=0)
        psi, leftResult = measure(psi, target=numQubits-1)

        probs[leftResult + rightResult] += 1

    for key in probs:
        probs[key] /= shots

    plt.bar(range(len(probs)), list(probs.values()), align='center')
    plt.xticks(range(len(probs)), list(probs.keys()))
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(probs.values()):
        plt.text(xlocs[i] - 0.2, v + .007, str(v))
    plt.xlabel('Measurement Result')
    plt.ylabel('Probability')
    plt.show()

    return np.array([probs['01'] / (probs['01'] + probs['11']), probs['11'] / (probs['01'] + probs['11'])])

A = np.array([[1, -1/3.], [-1/3., 1]])
b = np.array([0, 1])
estimate = simulateHHL(A, b, numQubits=4, shots=1000)
print(estimate)
total = np.abs(3./8.)**2 + np.abs(9./8.)**2
print(np.array([np.abs(3./8.)**2 / total, np.abs(9./8.)**2 / total]))




# def toffoli(psi, a, b, c):
#     psi.ApplyHadamard(c)
#     psi.ApplyCPauliX(b, c)
#     psi.ApplyRotationZ(c, -np.pi/4.)
#     psi.ApplyCPauliX(a, c)
#     psi.ApplyRotationZ(c, np.pi/4.)
#     psi.ApplyCPauliX(b, c)
#     psi.ApplyRotationZ(c, -np.pi/4.)
#     psi.ApplyCPauliX(a, c)
#     psi.ApplyRotationZ(b, np.pi/4.)
#     psi.ApplyRotationZ(c, np.pi/4.)
#     psi.ApplyCPauliX(a, b)
#     psi.ApplyHadamard(c)
#     psi.ApplyRotationZ(a, np.pi/4.)
#     psi.ApplyRotationZ(b, -np.pi/4.)
#     psi.ApplyCPauliX(a, b)
#     return psi




# def simulateHHL(A, b, numQubits=4, shots=1000):
#     initialSate = 0
#     qftGates, iqftGates = computeQFTRotations(numQubits)
#     qpeGates, iqpeGates = computeQPERotations(A, numQubits)

#     probs = {'00': 0, '01': 0, '10': 0, '11': 0}

#     for i in range(shots):

#         # Initialize a quantum system
#         psi = simulator.QubitRegister(5, 'base', initialSate, 0)

#         # FIXME: This should use b!
#         psi.ApplyPauliX(3)

#         psi = simulateQPE(psi, numQubits, qpeGates, iqftGates)

#         # Controlled Y rotations
#         # for i in range(1, numQubits-1):
#         # psi = simulateYRotations(psi, numQubits)
#         psi.ApplyCRotationY(1, 0, np.pi)#2 * np.arcsin(1/2.))
#         psi = toffoli(psi, 1, 2, 4)
#         psi.ApplyCRotationY(4, 0, np.pi/3.)#2 * np.arcsin(1/3.))
#         psi = toffoli(psi, 1, 2, 4)

#         psi = simulateIQPE(psi, numQubits, iqpeGates, qftGates)

#         psi, rightResult = measure(psi, target=0)
#         psi, leftResult = measure(psi, target=numQubits-1)

#         probs[leftResult + rightResult] += 1

#     for key in probs:
#         probs[key] /= shots

#     plt.bar(range(len(probs)), list(probs.values()), align='center')
#     plt.xticks(range(len(probs)), list(probs.keys()))
#     xlocs, xlabs = plt.xticks()
#     for i, v in enumerate(probs.values()):
#         plt.text(xlocs[i] - 0.2, v + .007, str(v))
#     plt.xlabel('Measurement Result')
#     plt.ylabel('Probability')
#     plt.show()

#     return np.array([probs['01'] / (probs['01'] + probs['11']), probs['11'] / (probs['01'] + probs['11'])])

# A = np.array([[13/2., -1/2.], [-1/2., 13/2]])
# b = np.array([0, 1])
# estimate = simulateHHL(A, b, numQubits=4, shots=5000)
# print(estimate)
# total = np.abs(1./84.)**2 + np.abs(13./84.)**2
# print(np.array([np.abs(1./84.)**2 / total, np.abs(13./84.)**2 / total]))

# # TODO: cry(2, 0, pi)
# # TODO: toffoliry(1,2, 0, pi/3)
# A = np.array([[5/2., -1/2.], [-1/2., 5/2]])
# b = np.array([0, 1])
# estimate = simulateHHL(A, b, numQubits=4, shots=5000)
# print(estimate)
# total = np.abs(1./12.)**2 + np.abs(5./12.)**2
# print(np.array([np.abs(1./12.)**2 / total, np.abs(5./12.)**2 / total]))







# Begin old code...

# u = np.array([[-1/2. + 1j/2., 1/2. + 1j/2.], [1/2. + 1j/2., -1/2. + 1j/2.]])
# u2 = np.array([[0. + 0j, -1. + 0j], [-1. + 0j, 0. + 0j]])
# uList = [u, u2]

# uInv = np.array([[-1/2. - 1j/2., 1/2. - 1j/2.], [1/2. - 1j/2., -1/2. - 1j/2.]])
# u2Inv = np.array([[0. + 0j, -1. + 0j], [-1. + 0j, 0. + 0j]])
# uInvList = [uInv, u2Inv]

# estimate = simulateHHL(uList, uInvList)
# print(estimate)
# total = np.abs(3./8.)**2 + np.abs(9./8.)**2
# print(np.array([np.abs(3./8.)**2 / total, np.abs(9./8.)**2 / total]))








# import sys
# sys.path.append('../')
# sys.path.insert(0, '../../intel-qs/build/lib') # Change this to match your installation location.
# import intelqs_py as simulator
# import emulator
# import numpy as np
# import time
# from matplotlib import pyplot as plt


# def simulateHHL(uList, uInvList):   #TODO Pass in b operation, num clock qubits
#     N = 4   # number of qubits
#     M = 1000 # number of shots
#     initialSate = 0
#     ZI = np.array([[1, 0], [0, np.exp(-1j*np.pi/2.)]], dtype=complex)
#     Z = np.array([[1, 0], [0, np.exp(1j*np.pi/2.)]], dtype=complex)

#     probs = {'00': 0, '01': 0, '10': 0, '11': 0}

#     for i in range(M):

#         # Initialize a quantum system
#         psi = simulator.QubitRegister(N, 'base', initialSate, 0)

#         # Prepare b register
#         psi.ApplyPauliX(3)

#         # Begin QPE
#         # Prepare clock register
#         psi.ApplyHadamard(1)
#         psi.ApplyHadamard(2)

#         # Apply controlled-U**i operations
#         psi.ApplyControlled1QubitGate(1, 3, uList[0])
#         psi.ApplyControlled1QubitGate(2, 3, uList[1])

#         # IQFT
#         psi.ApplyHadamard(2)
#         psi.ApplyControlled1QubitGate(1, 2, ZI)
#         # psi.ApplyCPhaseRotation(1, 2, -np.pi/2)
#         psi.ApplyHadamard(1)
#         psi.ApplySwap(1, 2)
#         # End QPE

#         # Controlled Y rotations
#         psi.ApplyCRotationY(1, 0, np.pi)
#         psi.ApplyCRotationY(2, 0, np.pi/3.)

#         # Begin IQPE
#         # QFT
#         psi.ApplySwap(1, 2)
#         psi.ApplyHadamard(1)
#         psi.ApplyControlled1QubitGate(1, 2, Z)
#         # psi.ApplyCPhaseRotation(1, 2, np.pi/2)
#         psi.ApplyHadamard(2)

#         # Apply controlled-U**i operations
#         psi.ApplyControlled1QubitGate(2, 3, uInvList[1])
#         psi.ApplyControlled1QubitGate(1, 3, uInvList[0])

#         # Reset clock register
#         psi.ApplyHadamard(2)
#         psi.ApplyHadamard(1)

#         # Compute the probability of qubit 0 being in state |1>.
#         measured_qubit = 0
#         prob = psi.GetProbability(measured_qubit)
#         right = None

#         # Draw random number in [0,1)
#         r = np.random.rand()
#         if r < prob:
#             # Collapse the wavefunction according to qubit 0 being in |1>.
#             psi.CollapseQubit(measured_qubit, True)
#             right = '1'
#         else:
#             # Collapse the wavefunction according to qubit 0 being in |0>
#             psi.CollapseQubit(measured_qubit, False)
#             right = '0'

#         # In both cases one needs to re-normalize the wavefunction:
#         psi.Normalize()

#         # Compute the probability of qubit 3 being in state |1>.
#         measured_qubit = 3
#         prob = psi.GetProbability( measured_qubit )
#         left = None

#         # Draw random number in [0,1)
#         r = np.random.rand()
#         if r < prob:
#             # Collapse the wavefunction according to qubit 3 being in |1>.
#             psi.CollapseQubit(measured_qubit,True)
#             left = '1'
#         else:
#             # Collapse the wavefunction according to qubit 3 being in |0>
#             psi.CollapseQubit(measured_qubit,False)
#             left = '0'

#         probs[left + right] += 1

#     for key in probs:
#         probs[key] /= M

#     plt.bar(range(len(probs)), list(probs.values()), align='center')
#     plt.xticks(range(len(probs)), list(probs.keys()))
#     xlocs, xlabs = plt.xticks()
#     for i, v in enumerate(probs.values()):
#         plt.text(xlocs[i] - 0.2, v + .007, str(v))
#     plt.xlabel('Measurement Result')
#     plt.ylabel('Probability')
#     plt.show()

#     return np.array([probs['01'] / (probs['01'] + probs['11']), probs['11'] / (probs['01'] + probs['11'])])

# u = np.array([[-1/2. + 1j/2., 1/2. + 1j/2.], [1/2. + 1j/2., -1/2. + 1j/2.]])
# u2 = np.array([[0. + 0j, -1. + 0j], [-1. + 0j, 0. + 0j]])
# uList = [u, u2]

# uInv = np.array([[-1/2. - 1j/2., 1/2. - 1j/2.], [1/2. - 1j/2., -1/2. - 1j/2.]])
# u2Inv = np.array([[0. + 0j, -1. + 0j], [-1. + 0j, 0. + 0j]])
# uInvList = [uInv, u2Inv]

# estimate = simulateHHL(uList, uInvList)
# print(estimate)
# total = np.abs(3./8.)**2 + np.abs(9./8.)**2
# print(np.array([np.abs(3./8.)**2 / total, np.abs(9./8.)**2 / total]))




























# import sys
# sys.path.append('../')
# sys.path.insert(0, '../../intel-qs/build/lib') # Change this to match your installation location.
# import intelqs_py as simulator
# import emulator
# import numpy as np
# import time
# from matplotlib import pyplot as plt

# # Define the number of times to repeat each following experiment.
# M = 10

# # Define the number of qubits to simulate for each experiment.
# N_QFT = 6
# N_QPE = 12
# N_EM_QFT = 6
# N_EM_QPE = 12

# # ---------------------------------------------------------------------------- #
# #                                      QFT                                     #
# # ---------------------------------------------------------------------------- #

# # Set the maximum number of qubits to simulate.
# N = N_QFT

# # Define the z-rotations needed for the simulation method.
# Z = []
# for n in range(2, N+1):
#    Z.append(np.array([[1, 0], [0, np.exp(1j*2*np.pi/2**n)]], dtype=complex)) 

# # Create a list of various numbers of qubits <= N to simulate.
# num_qubits = np.arange(2, N+1, 2)

# # Define a list to hold the emulator and simulator times respectively.
# em_times = []
# sim_times = []

# # For each iteration of the experiment:
# for m in range(M):

#     # Define arrays to hold the results of this iteration (batch).
#     em_batch = []
#     sim_batch = []
        
#     # For each number of qubits:
#     for n in num_qubits:
        
#         # Initialize the simulator to a random state.
#         psi = simulator.QubitRegister(n, 'base', 0, 0)
#         rng = simulator.RandomNumberGenerator()
#         seed = np.random.randint(0, 10000, 1)
#         rng.SetSeedStreamPtrs(seed)
#         psi.SetRngPtr(rng)
#         psi.Initialize('rand', 0)
        
#         # Extract the simulator state to pass it into the emulator.
#         state = []
#         for i in range(2**n):
#             state.append(psi[i])
#         state = np.array(state, dtype=complex)
        
#         # Perform the QFT with the emulator and time how long it takes.
#         start_time = time.time()
#         em_state = emulator.qft(state)
#         em_batch.append(time.time() - start_time)

#         # Perform the QFT with the simulator and time how long it takes.
#         start_time = time.time()
#         for i in range(n-1, -1, -1):
#             psi.ApplyHadamard(i)
#             for j in range(0, i, 1):
#                 psi.ApplyControlled1QubitGate(j, i, Z[i-j-1])
#         for i in range(n//2):
#             psi.ApplySwap(i, n-i-1)
#         sim_batch.append(time.time() - start_time)
        
#         # Extract the resultant state from the simulator.
#         sim_state = []
#         for i in range(2**n):
#             sim_state.append(psi[i])
#         sim_state = np.array(sim_state, dtype=complex)
        
#         # Ensure that the final states of the emulator and simulator agree.
#         if not np.allclose(sim_state, em_state):
#             print('Output differed for ' + str(n) + ' qubits.')
#             print(sim_state)
#             print(em_state)
#             print()

#     # Append the batch results to the main array.     
#     em_times.append(em_batch)
#     sim_times.append(sim_batch)
        
# # Average the times over each batch.
# em_times = np.array(em_times)
# sim_times = np.array(sim_times)
# em_times = np.sum(em_times, axis=0)/M
# sim_times = np.sum(sim_times, axis=0)/M

# # Plot the times for each QFT operation.
# fig = plt.figure()
# plt.plot(num_qubits, em_times, 'o-k', label='Emulator')   
# plt.plot(num_qubits, sim_times, 'o-r', label='Simulator')
# plt.title('Speed Comparison for QFT')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.legend(loc='best')
# plt.savefig('Plots/qft.png', dpi=600)

# # Plot the times for each QFT operation on a log plot.
# fig = plt.figure()
# plt.semilogy(num_qubits, em_times, 'o-k', label='Emulator')   
# plt.semilogy(num_qubits, sim_times, 'o-r', label='Simulator')
# plt.title('Speed Comparison for QFT on Log Plot')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.legend(loc='best')
# plt.savefig('Plots/qft_log.png', dpi=600)

# # ---------------------------------------------------------------------------- #
# #                                      QPE                                     #
# # ---------------------------------------------------------------------------- #

# # Set the maximum number of qubits to simulate.
# N = N_QPE

# # Define a matrix (U) and an eigenvector.
# z = np.random.uniform(0, 1)
# U = np.array([[1, 0], [0, np.exp(1j*z)]])
# phi = np.array([0, 1])

# # Define the z-rotations needed for the simulation method.
# Z = []
# for n in range(1, N):
#    Z.append(np.array([[1, 0], [0, np.exp(-1j*np.pi/2**n)]], dtype=complex))

# # Create a list of various numbers of qubits <= N to simulate.
# num_qubits = np.arange(2, N+1, 2)

# # Define a list to hold the emulator and simulator times respectively.
# em_times = []
# sim_times = []

# # For each iteration of the experiment:
# for m in range(M):

#     # Define arrays to hold the results of this iteration (batch).
#     em_batch = []
#     sim_batch = []

#     # For each number of qubits:
#     for n in num_qubits:
        
#         # Initialize the simulator to the 0 state.
#         psi = simulator.QubitRegister(n+1, 'base', 0, 0)
        
#         # Extract the simulator state to pass it into the emulator.
#         state = []
#         for i in range(2**n):
#             state.append(psi[i])
#         state = np.array(state, dtype=complex)
        
#         # Perform the QPE with the emulator and time how long it takes.
#         start_time = time.time()
#         em_state = emulator.qpe(U, phi, n)
#         em_batch.append(time.time() - start_time)

#         # Perform the QPE with the simulator and time how long it takes.
#         start_time = time.time()
#         for i in range(n):
#             psi.ApplyHadamard(i)
#         psi.ApplyPauliX(n)
#         for i in range(0, n):
#             for j in range(2**i):
#                 psi.ApplyControlled1QubitGate(i, n, U)
#         for i in range(n//2):
#             psi.ApplySwap(i, n-i-1)
#         for j in range(n):
#             for m in range(j):
#                 psi.ApplyControlled1QubitGate(m, j, Z[j-m-1])
#             psi.ApplyHadamard(j)
#         for i in range(n):
#             prob = psi.GetProbability(i)
#             if prob < 0.5:
#                 psi.CollapseQubit(i, False)
#             else:
#                 psi.CollapseQubit(i, True)
#         sim_batch.append(time.time() - start_time)
        
#         # Extract the resultant state from the simulator.
#         sim_state = []
#         for i in range(2**n, 2**(n+1)):
#             sim_state.append(psi[i])
#         sim_state = np.array(sim_state, dtype=complex)
        
#         # Ensure that the final states of the emulator and simulator agree.
#         if not np.allclose(np.nonzero(sim_state), np.nonzero(em_state)):
#             print('Output differed for ' + str(n) + ' qubits.')
#             print(sim_state)
#             print(em_state)
#             print()
        
#     # Append the batch results to the main array.     
#     em_times.append(em_batch)
#     sim_times.append(sim_batch)

# # Average the times over each batch.
# em_times = np.array(em_times)
# sim_times = np.array(sim_times)
# em_times = np.sum(em_times, axis=0)/M
# sim_times = np.sum(sim_times, axis=0)/M

# # Plot the times for each QPE operation.
# fig = plt.figure()
# plt.plot(num_qubits, em_times, 'o-k', label='Emulator')   
# plt.plot(num_qubits, sim_times, 'o-r', label='Simulator')
# plt.title('Speed Comparison for QPE')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.legend(loc='best')
# plt.savefig('Plots/qpe.png', dpi=600)

# # Plot the times for each QPE operation on a log plot.
# fig = plt.figure()
# plt.semilogy(num_qubits, em_times, 'o-k', label='Emulator')   
# plt.semilogy(num_qubits, sim_times, 'o-r', label='Simulator')
# plt.title('Speed Comparison for QPE on Log Plot')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.legend(loc='best')
# plt.savefig('Plots/qpe_log.png', dpi=600)

# # ---------------------------------------------------------------------------- #
# #                                 Emulator QFT                                 #
# # ---------------------------------------------------------------------------- #

# # Set the maximum number of qubits to simulate.
# N = N_EM_QFT

# # Create a list of various numbers of qubits <= N to simulate.
# num_qubits = np.arange(2, N+1, 2)

# # Define a list to hold the emulator and simulator times respectively.
# em_times = []

# # For each iteration of the experiment:
# for m in range(M):
    
#     # Define arrays to hold the results of this iteration (batch).
#     em_batch = []

#     # For each number of qubits:
#     for n in num_qubits:
        
#         # Define a random initial state.
#         state = np.random.uniform(0, 1, 2**n).astype(complex)
        
#         # Perform the QFT with the emulator and time how long it takes.
#         start_time = time.time()
#         emulator.qft(state)
#         em_batch.append(time.time() - start_time)

#     # Append the batch results to the main array.
#     em_times.append(em_batch)
      
# # Average the times over each batch to get the average time for each operation.
# em_times = np.array(em_times)
# em_times = np.sum(em_times, axis=0)/M

# # Plot the times for each QFT operation.
# fig = plt.figure()
# plt.plot(num_qubits, em_times, 'o-k')   
# plt.title('Emulator Speed for QFT')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.savefig('Plots/em_qft.png', dpi=600)

# # Plot the times for each QFT operation on a log plot.
# fig = plt.figure()
# plt.semilogy(num_qubits, em_times, 'o-k')   
# plt.title('Emulator Speed for QFT on Log Plot')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.savefig('Plots/em_qft_log.png', dpi=600)

# # ---------------------------------------------------------------------------- #
# #                                 Emulator QPE                                 #
# # ---------------------------------------------------------------------------- #

# # Perform this experiment 10 times the amount of the others.
# M_hat = 10*M

# # Set the maximum number of qubits to simulate.
# N = N_EM_QPE

# # Define a matrix (the T gate) and an eigenvector.
# T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
# phi = np.array([0, 1])

# # Create a list of various numbers of qubits <= N to simulate.
# num_qubits = np.arange(2, N+1, 2)

# # Define a list to hold the emulator and simulator times respectively.
# em_times = []

# # For each iteration of the experiment:
# for m in range(M_hat):
    
#     # Define arrays to hold the results of this iteration (batch).
#     em_batch = []

#     # For each number of qubits:
#     for n in num_qubits:
        
#         # Perform the QFT with the emulator and time how long it takes.
#         start_time = time.time()
#         emulator.qpe(T, phi, n)
#         em_batch.append(time.time() - start_time)

#     # Append the batch results to the main array.
#     em_times.append(em_batch)
      
# # Average the times over each batch to get the average time for each operation.
# em_times = np.array(em_times)
# em_times = np.sum(em_times, axis=0)/M_hat

# # Plot the times for each QPE operation.
# fig = plt.figure()
# plt.plot(num_qubits, em_times, 'o-k')   
# plt.title('Emulator Speed for QPE')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.savefig('Plots/em_qpe.png', dpi=600)

# # Plot the times for each QPE operation on a log plot.
# fig = plt.figure()
# plt.semilogy(num_qubits, em_times, 'o-k')   
# plt.title('Emulator Speed for QPE on Log Plot')
# plt.xlabel('Number of Qubits')
# plt.ylabel('Time (seconds)')
# plt.savefig('Plots/em_qpe_log.png', dpi=600)
