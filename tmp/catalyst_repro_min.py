import pennylane as qml
from catalyst import grad

dev = qml.device('lightning.qubit', wires=1)

@qml.qjit
@qml.qnode(dev, diff_method='adjoint')
def f(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.Z(0))

@qml.qjit
def g(x):
    return grad(f)(x)

print('f', float(f(0.123)))
print('g', float(g(0.123)))
