import numpy as np
import matplotlib.pyplot as plt

def generate_complex_Gaussian_channel(K, t):
    mean = 0.0
    variance = 0.5
    channel_quality = np.random.normal(mean, np.sqrt(variance), size=(K, t))
    return channel_quality

def generate_chaotic_channel(K, t):
    means = np.random.uniform(0, 1, size=(K,))
    variances = np.random.uniform(0, 3, size=(K,))
    channel_quality = np.zeros((K, t), dtype=np.complex128)
    for i in range(K):
        channel_quality[i, :] = np.random.normal(means[i], np.sqrt(variances[i]), size=(1, t))
    return channel_quality

def generate_correlative_channel(K, t):
    mean = 0.0
    variance = 1.0
    channel_quality = np.random.normal(mean, np.sqrt(variance), size=(K, t))
    return channel_quality

def channel_quality_based_greedy_user_selection(channel_quality, t, P):
    K = channel_quality.shape[0]
    users = np.arange(K)
    selected_users = []

    while len(selected_users) < t:
        remaining_users = np.setdiff1d(users, selected_users)
        selected_user = remaining_users[np.argmax(np.linalg.norm(channel_quality[remaining_users], axis=1))]
        selected_users.append(selected_user)

    power_allocation = P / t

    rates = np.log2(1 + power_allocation * np.abs(channel_quality[selected_users]) ** 2)

    total_rate = np.sum(rates)

    return total_rate

def simulate_channel_quality_based_greedy_user_selection(channel_generator, K, t, P, num_simulations):
    total_rates = []

    for _ in range(num_simulations):
        channel_quality = channel_generator(K, t)
        total_rate = channel_quality_based_greedy_user_selection(channel_quality, t, P)
        total_rates.append(total_rate)

    return total_rates

def plot_total_rates(total_rates, channel_name):
    fig, ax = plt.subplots()
    ax.hist(total_rates, bins=100, density=True, alpha=0.7)
    ax.set_xlabel('Transmission Rate (bits/s/Hz)')
    ax.set_ylabel('Probability')
    ax.set_title('Total Rates per Transmission - {}'.format(channel_name))
    plt.savefig('{}.png'.format(channel_name))
    plt.show()

if __name__ == '__main__':
    K = 30  # Number of users
    P = 5  # Transmission power (Watts)
    t = 5  # Number of users to select
    num_simulations = 5000  # Number of simulations

    # Simulate channel 1 (complex Gaussian channel)
    total_rates_channel1 = simulate_channel_quality_based_greedy_user_selection(generate_complex_Gaussian_channel, K, t, P, num_simulations)
    plot_total_rates(total_rates_channel1, 'Channel 1')

    # Simulate channel 2 (chaotic channel model)
    total_rates_channel2 = simulate_channel_quality_based_greedy_user_selection(generate_chaotic_channel, K, t, P, num_simulations)
    plot_total_rates(total_rates_channel2, 'Channel 2')

    # Simulate channel 3 (complete correlative channel)
    total_rates_channel3 = simulate_channel_quality_based_greedy_user_selection(generate_correlative_channel, K, t, P, num_simulations)
    plot_total_rates(total_rates_channel3, 'Channel 3')
