import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import itertools
import random

#   -------------------------------------- global variables -----------------------------------
iters = 5000
k = 30  # number of receiving users
t = 2  # number of transmitting antennae
square_Sig = 0.5  # variance for each complex gaussian
P = 5  # the total power allowed
num_bins = 100
num_processes = 10
GAUSS = 1
CHAOTIC = 2
CORR = 3
colors = ["blue", "green", "red", "cyan", "magenta", "pink", "olive"]

ch_mean = np.random.uniform(0, 3, k)
ch_variance = np.random.uniform(0, 1, k)
iid_mean = np.zeros(t)
iid_variance = square_Sig * np.eye(t)
corr_mean = np.zeros(t)
corr_variance = 1 * np.eye(t)


def correlated_channel():  # Create corr channel with expected value of 0 and a variance of 1
    H = np.zeros((k, t), dtype=complex)
    first_row_real = np.random.multivariate_normal(corr_mean, (1 / 2) * corr_variance)
    first_row_imaginary = 1j * np.random.multivariate_normal(corr_mean, (1 / 2) * corr_variance)
    first_row = first_row_real + first_row_imaginary
    for i in range(k):
        H[i] = first_row
    return H  #


def chaotic_channel():  # Create chaotic channel expected value uniformly distributed 𝑈[0,1] and a variance
    # uniformly distributed 𝑈[0,3]
    H = np.zeros((k, t), dtype=complex)
    for i in range(k):
        H_row_real = np.random.multivariate_normal(ch_mean[i] * np.ones(t),
                                                   (1 / 2) * ch_variance[i] * np.eye(t))
        H_row_imaginary = 1j * np.random.multivariate_normal(ch_mean[i] * np.ones(t),
                                                             (1 / 2) * ch_variance[i] * np.eye(t))
        H[i] = H_row_real + H_row_imaginary
    return H


def iid_gaussian_channel():  # Create gaussian channel with expected value of 0 and a variance of 0.5
    H = np.zeros((k, t), dtype=complex)
    for i in range(k):
        H_row_real = np.random.multivariate_normal(iid_mean, (1 / 2) * iid_variance)
        H_row_imaginary = 1j * np.random.multivariate_normal(iid_mean, (1 / 2) * iid_variance)
        H[i] = H_row_real + H_row_imaginary
    return H


class BaseStation:

    def __init__(self, num_users, num_antennas, power, H_val):
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.power = power
        if H_val == 1:
            self.H = iid_gaussian_channel()
        elif H_val == 2:
            self.H = chaotic_channel()
        else:
            self.H = correlated_channel()

    def select_random_user(self):  # Pick one random user gives all P
        res = np.zeros(k)
        random_number = random.randint(0, 29)
        res[random_number] = P
        return res

    def select_strongest_user(self):  # check which user is strongest by norm gives all P

        max_norm = 0
        best_user = 0
        for i in range(k):
            row_norm = np.linalg.norm(self.H[i])
            if row_norm > max_norm:
                max_norm = row_norm
                best_user = i
        res = np.zeros(k)
        res[best_user] = P
        return res

    def select_random_t_users(self):  # Selecrt random t useres and uniform power allocation between the selected users.
        user_indices = np.random.choice(np.arange(k), size=t, replace=False)
        res = np.zeros(k)
        for index in user_indices:
            res[index] = P / t
        return res

    def select_strongest_t_users(
            self):  # Selecrt strongest t useres and uniform power allocation between the selected users.
        norms = np.linalg.norm(self.H, axis=1)
        indices = np.argpartition(norms, -t)[-t:]
        powers = np.zeros(k)
        for index in indices:
            powers[index] = P / t
        return powers

    def optimal_t_users(
            self):  # by check all possible subgroups, check which subgroup is optimal and uniform power allocation between the selected users
        best_users = None
        best_rate = 0
        all_users = list(range(k))
        combinations = itertools.combinations(all_users, t)
        for comb in combinations:
            powers = np.zeros(k)
            for ele in comb:
                powers[ele] = P / t
            res_rate = calc_rates(self.H, powers)
            if res_rate > best_rate:
                best_rate, best_users = res_rate, powers
        return best_users

    def select_SUS_t_users(self):  # select by using sus algorithm we say in the presentaion and the link we provided
        best_users = np.zeros(k)
        users = []
        ind = self.num_users
        remaining_channels = self.H.copy()
        for _ in range(t):
            rates = np.linalg.norm(remaining_channels, axis=1) ** 2
            for j in users:
                rates = np.delete(rates, j)
            user = np.argmax(rates)
            users.append(user)
            selected_channel = remaining_channels[user]
            selected_channel_normalized = selected_channel / np.linalg.norm(selected_channel)

            for i in range(ind):
                if i != user and i not in users:
                    proj = np.dot(remaining_channels[i],
                                  selected_channel_normalized.conj()) * selected_channel_normalized
                    remaining_channels[i] -= proj
        for i in users:
            best_users[i] = P / t
        return best_users

    def select_orthonormal_basis_users(self):  # Demos algorithm
        res = np.zeros(k)
        user = np.argmax(np.linalg.norm(self.H, axis=1))  # normalize the vector of strongest user
        max_user_vector = self.H[user]
        basis_orthonormal = generate_orthonormal_basis(max_user_vector)  # create random orthonormal basis
        max_val = 0
        curr_user = 0
        users = []
        for i in range(k):
            if i == user:
                continue
            for j in range(t):
                rate = calculate_SINR(self.H, basis_orthonormal, P, i, j)
                if rate > max_val:
                    max_val = rate
                    curr_user = i
            users.append(curr_user)
        users = list(set(users))
        index = np.array(users)
        P_val = P / len(users)  # seperate the power for the optimal subgroup.
        res = np.insert(res, index, P_val)
        return res


def generate_orthonormal_basis(vector):
    t = vector.shape[0]
    basis = []
    for i in range(t):
        # Create a random complex vector of the same size
        random_vector = np.random.randn(t) + 1j * np.random.randn(t)
        # Orthogonalize the random vector with respect to the previously generated basis vectors
        for basis_vector in basis:
            random_vector -= np.dot(random_vector, basis_vector.conj()) * basis_vector
        # Normalize the orthogonalized vector
        normalized_vector = random_vector / np.linalg.norm(random_vector)
        # Append the normalized vector to the basis list
        basis.append(normalized_vector)
    # Return the orthonormal basis vectors as a single 2D array
    return np.stack(basis, axis=0)


def calculate_SINR(H, V, P, i, j):
    hi = H[i]  # Get the i-th row of H matrix
    wj = V[:, j]  # Get the j-th column of V matrix
    numerator = np.abs(np.dot(hi, wj.conj())) ** 2  # Calculate the numerator
    denominator = P + np.sum(
        np.abs(np.dot(hi, V[:, k].conj())) ** 2 for k in range(V.shape[1]) if k != j)  # Calculate the denominator
    SINR = numerator / denominator  # Calculate SINR
    return SINR


def calc_rates(H, users_powers):  # Calc for denominator
    V = np.linalg.pinv(H)
    sum_rates = 0
    for i in range(k):
        sum_rates = sum_rates + calc_single_rate(H, V, users_powers, i)
    return sum_rates


def calc_single_rate(H, V, users_powers, n):  # calc each rate seperate
    sum = 1
    for j in range(k):
        if j == n:
            continue
        else:
            sum += users_powers[j] * (abs((np.vdot(H[n], V[:, j] / np.linalg.norm(V[:, j])))) ** 2) / sum
    numerator = np.vdot(H[n], V[:, n] / np.linalg.norm(V[:, n]))
    rate = np.log1p(users_powers[n] * abs(numerator) ** 2)

    return rate


def simulation_run(method, num_simulations):
    gauss_results = []
    chaot_results = []
    corr_results = []
    for _ in range(num_simulations):
        base_station_gauss = BaseStation(k, t, P, GAUSS)  # crate gaussian channel
        base_station_chaotic = BaseStation(k, t, P, CHAOTIC)  # crate chaotic channel
        base_station_corr = BaseStation(k, t, P, CORR)  # crate corr channel

        rates_Gauss = getattr(base_station_gauss, method)()  # crate list of powers
        iid_rate = calc_rates(base_station_gauss.H, rates_Gauss)  # calculate rate
        gauss_results.append(iid_rate)

        rates_chaotic = getattr(base_station_chaotic, method)()
        chaot_rate = calc_rates(base_station_chaotic.H, rates_chaotic)
        chaot_results.append(chaot_rate)

        rates_corr = getattr(base_station_corr, method)()
        corr_rate = calc_rates(base_station_corr.H, rates_corr)
        corr_results.append(corr_rate)

    return gauss_results, chaot_results, corr_results


def simulation_run_optimal(num_simulations):  # seperate for better running time
    global iter
    gauss_results = []
    chaot_results = []
    corr_results = []

    base_station_gauss = BaseStation(k, t, P, GAUSS)
    base_station_chaotic = BaseStation(k, t, P, CHAOTIC)
    base_station_corr = BaseStation(k, t, P, CORR)

    rates_Gauss = base_station_gauss.optimal_t_users()
    iid_rate = calc_rates(base_station_gauss.H, rates_Gauss)
    gauss_results.append(iid_rate)

    rates_chaotic = base_station_chaotic.optimal_t_users()
    chaot_rate = calc_rates(base_station_chaotic.H, rates_chaotic)
    chaot_results.append(chaot_rate)

    rates_corr = base_station_corr.optimal_t_users()
    corr_rate = calc_rates(base_station_corr.H, rates_corr)
    corr_results.append(corr_rate)

    return gauss_results, chaot_results, corr_results


def simulate(num_simulations=iters, num_processes=num_processes):
    total_rates = {method: [] for method in BaseStation.__dict__ if method.startswith('select')}
    colors_plot = colors[0:len(total_rates.keys()) + 1]
    pool = mp.Pool(processes=num_processes)
    results = [pool.apply_async(simulation_run, args=(method, num_simulations)) for method in
               total_rates.keys()]
    pool.close()
    pool.join()

    num_processes = mp.cpu_count()  # Get the number of available CPU cores

    pool = mp.Pool(processes=num_processes)

    results_opt = pool.map(simulation_run_optimal, range(num_simulations))

    pool.close()
    pool.join()

    results_opt_array = np.array(results_opt)
    transposed_array_opt = np.transpose(results_opt_array)
    results_opt = transposed_array_opt.tolist()

    for result, method in zip(results, total_rates.keys()):
        total_rates[method] = result.get()

    total_rates.update({"optimal_t_users": tuple(results_opt[0])})
    total_iid_rates = np.zeros(0)
    total_chaotic_rates = np.zeros(0)
    total_correlated_rates = np.zeros(0)

    for key, value in total_rates.items():
        total_iid_rates = np.concatenate((total_iid_rates, value[0]))
        a = plt.hist(total_iid_rates, alpha=0.3, bins=num_bins, label=f"" + str(key).replace('select_', ''),
                     density=True, color='orange')
        plt.legend()
        plt.xlabel("Transmission rates")
        plt.title(f"IID Channel (k={k} and t={t})")
        plt.savefig('Gauss' + str(key) + '.png')
        plt.show()

        total_chaotic_rates = np.concatenate((total_chaotic_rates, value[1]))
        a = plt.hist(total_chaotic_rates, alpha=0.3, bins=num_bins, label=f"" + str(key).replace('select_', ''),
                     density=True, color='blue')
        plt.legend()
        plt.xlabel("Transmission rates")
        plt.title(f"Chaotic Channel (k={k} and t={t})")
        plt.savefig('Chaotic' + str(key) + '.png')
        plt.show()

        total_correlated_rates = np.concatenate((total_correlated_rates, value[2]))
        a = plt.hist(total_correlated_rates, alpha=0.3, bins=num_bins, label=f"" + str(key).replace('select_', ''),
                     density=True, color='red')
        plt.legend()
        plt.xlabel("Transmission rates")
        plt.title(f"Correlated Channel (k={k} and t={t})")
        plt.savefig('Correlated' + str(key) + '.png')
        plt.show()

    pllt_gauss = np.zeros(0)
    pllt_chaot = np.zeros(0)
    pllt_corr = np.zeros(0)

    for key, value in total_rates.items():
        pllt_gauss = np.append(pllt_gauss, value[0])
        pllt_chaot = np.append(pllt_chaot, value[1])
        pllt_corr = np.append(pllt_corr, value[2])

    pllt_gauss = np.array_split(pllt_gauss, len(total_rates.keys()))
    pllt_chaot = np.array_split(pllt_chaot, len(total_rates.keys()))
    pllt_corr = np.array_split(pllt_corr, len(total_rates.keys()))

    a = plt.hist(pllt_gauss, alpha=0.3, bins=num_bins,
                 label=[f"{str(i).replace('select_', '')}" for i in total_rates.keys()],
                 density=True, color=colors_plot, stacked=True)
    plt.xlabel("Transmission rates")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f"IID Channel (k={k} and t={t})")
    plt.savefig('Gauss.png')
    plt.show()

    a = plt.hist(pllt_chaot, alpha=0.3, bins=num_bins,
                 label=[f"{str(i).replace('select_', '')}" for i in total_rates.keys()],
                 density=True, color=colors_plot, stacked=True)
    plt.xlabel("Transmission rates")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f"Chaotic Channel (k={k} and t={t})")
    plt.savefig('Chaotic.png')
    plt.show()

    a = plt.hist(pllt_corr, alpha=0.3, bins=num_bins,
                 label=[f"{str(i).replace('select_', '')}" for i in total_rates.keys()],
                 density=True, color=colors_plot, stacked=True)
    plt.xlabel("Transmission rates")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f"Correlated Channel (k={k} and t={t})")
    plt.savefig('Correlated.png')
    plt.show()

    return


if __name__ == '__main__':
    start_time = time.time()
    simulate()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
