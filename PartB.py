
import numpy as np
import matplotlib.pyplot as plt
import scipy

Mr = Mt = 4
P = 5
max_N = 5*Mr
M = 5000
variance = 0.5

def generate_symbols(N):
    symbols = np.random.choice(range(-4, 5), (Mt, N))
    symbols = symbols / np.linalg.norm(symbols)  # normalize the entire matrix
    power_allocated_symbols = np.sqrt(P/Mt) * symbols
    return power_allocated_symbols

def channel_matrix():
    return np.random.normal(0, np.sqrt(1), (Mr, Mt))

def noise(N):
    return np.random.normal(0, np.sqrt(variance), (Mr, N))

def transmit(symbols, H):
    return H @ symbols

def receive(y, n):
    return y + n

def LS_estimate(S, R):
    R_pinv = np.linalg.pinv(R, 0.0001)
    return S @ R_pinv

def MSE(H, H_hat):
    return (np.linalg.norm((H - H_hat), 'fro')**2)/(Mr*Mt)


mse_values = []
cnt = 0
dividor = 0
H = channel_matrix()
for N in range(Mr, max_N + 1):
    temp_mse = []
    for _ in range(M):
        S = generate_symbols(N)
        n = noise(N)
        Y = transmit(S, H)
        R = receive(S, n)
        H_hat = LS_estimate(S, R)
        mse = MSE(H, H_hat)
        temp_mse.append(mse)
    for i in temp_mse:
        cnt = cnt + i
        dividor = dividor + 1

    mse_values.append(cnt/dividor)
    dividor = 0
    cnt = 0
# Convert MSE to dB
mse_db = 20 * np.log10(mse_values)

plt.plot(range(Mr, max_N + 1), mse_db)
plt.xlabel('Number of training symbols')
plt.ylabel('MMSE (in dB)')
plt.savefig('MMSE.png')
plt.grid(True)
plt.show()