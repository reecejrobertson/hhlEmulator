import numpy as np
import emulator as em
import simulator as sim

def runExperiment(A, b, applyRYs, t, numQubits, extraAncillas, shots=2048, emFig='emulator.png', simFig='simulator.png'):
    
    print('Begin Experiment')

    # Compute the solution, the simulator estimate, and the emulator estimate.
    x = np.linalg.solve(A, b)
    actual = (x / np.linalg.norm(x))**2
    simEstimate, simTime = sim.simulateHHL(A, b, applyRYs, t, numQubits, extraAncillas, shots=shots, figfile=simFig)
    emEstimate, emTime = em.emulateHHL(A, b, shots=shots, figfile=emFig)

    # Print results.
    print('A=['
        + np.array_str(A[0], precision=2,) + ','
        + np.array_str(A[1], precision=2) + '], b='
        + np.array_str(b, precision=2)
    )
    print('Actual x\t:', actual)
    print('Simulated x\t:', simEstimate)
    print('Emulated x\t:', emEstimate)
    print('Sim Time\t:', simTime)
    print('Em Time \t:', emTime)
    print('Sim Avg Time\t:', simTime/shots)
    print('Em Avg Time\t:', emTime/shots)
    print('End Experiment')
    print()

def applyToffoli(psi, a, b, c):
    psi.ApplyHadamard(c)
    psi.ApplyCPauliX(b, c)
    psi.ApplyRotationZ(c, -np.pi/4.)
    psi.ApplyCPauliX(a, c)
    psi.ApplyRotationZ(c, np.pi/4.)
    psi.ApplyCPauliX(b, c)
    psi.ApplyRotationZ(c, -np.pi/4.)
    psi.ApplyCPauliX(a, c)
    psi.ApplyRotationZ(b, -np.pi/4.)
    psi.ApplyRotationZ(c, np.pi/4.)
    psi.ApplyCPauliX(a, b)
    psi.ApplyHadamard(c)
    psi.ApplyRotationZ(b, -np.pi/4.)
    psi.ApplyCPauliX(a, b)
    psi.ApplyRotationZ(a, np.pi/4.)
    psi.ApplyRotationZ(b, np.pi/2.)
    return psi

def RY1(psi):
    psi.ApplyCRotationY(1, 0, np.pi)
    psi.ApplyCRotationY(2, 0, np.pi/3.)
    return psi

def RY2(psi):
    psi = applyToffoli(psi, 2, 3, 5)
    psi.ApplyCRotationY(5, 0, np.pi)
    psi = applyToffoli(psi, 2, 3, 5)
    psi = applyToffoli(psi, 1, 2, 5)
    psi = applyToffoli(psi, 3, 5, 6)
    psi.ApplyCRotationY(6, 0, np.pi/3.)
    psi = applyToffoli(psi, 3, 5, 6)
    psi = applyToffoli(psi, 1, 2, 5)
    return psi

runExperiment(np.array([[1 , -1/3], [-1/3 , 1]]), np.array([0, 1]), RY1, 3 * np.pi / 4, 4, 0, emFig='em1.png', simFig='sim1.png')
runExperiment(np.array([[13/2., -1/2.], [-1/2., 13/2]]), np.array([0, 1]), RY2, np.pi/4, 5, 2, emFig='em2.png', simFig='sim2.png')