from qiskit import QuantumRegister ,QuantumCircuit, Aer,execute

q = QuantumRegister(1)
hello_q = QuantumCircuit(q)

hello_q.iden(q[0])

job = execute(hello_q,'statevector_simulator')
result = job.result()
result.get_statevector()




