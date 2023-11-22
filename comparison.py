import numpy as np
import emulator as em
import simulator as sim

def runExperiment(A, b, shots=2048, emFig='emulator.png', simFig='simulator.png'):
    print('Begin Experiment')
    print('A=['
        + np.array_str(A[0], precision=2,) + ','
        + np.array_str(A[1], precision=2) + '], b='
        + np.array_str(b, precision=2)
    )
    x = np.linalg.solve(A, b)
    actual = (x / np.linalg.norm(x))**2
    print('Actual x\t:', actual)
    simEstimate, simTime = sim.simulateHHL(A, b, numQubits=4, shots=shots, figfile=simFig)
    print('Simulated x\t:', simEstimate)
    emEstimate, emTime = em.emulateHHL(A, b, shots=shots, figfile=emFig)
    print('Emulated x\t:', emEstimate)
    print('Sim Time\t:', simTime)
    print('Em Time \t:', emTime)
    print('Sim Avg Time\t:', simTime/shots)
    print('Em Avg Time\t:', emTime/shots)
    print('End Experiment')
    print()

runExperiment(np.array([[1 , -1/3], [-1/3 , 1]]), np.array([0, 1]), emFig='em1.png', simFig='sim1.png')
runExperiment(np.array([[5/2., -1/2.], [-1/2., 5/2]]), np.array([0, 1]), emFig='em2.png', simFig='sim2.png')