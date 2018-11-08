import numpy as np
import matplotlib.pyplot as plt

def gen_channel(parameters, fc=5E9, fs=2E9, dynamic_range=30):

    # Calculate samples/nanosec ratio
    nanosec_to_samples = int(1E-9 * fs)

    #####################################
    # Unpack parameters and convert units

    cluster_rate = parameters['cluster_rate'] / nanosec_to_samples
    inter_cluster_rate_1 = parameters['inter_cluster_rate_1'] / nanosec_to_samples
    inter_cluster_rate_2 = parameters['inter_cluster_rate_2'] / nanosec_to_samples
    beta = parameters['beta']
    cluster_decay = parameters['cluster_decay'] * nanosec_to_samples
    inter_cluster_decay = parameters['inter_cluster_decay'] * nanosec_to_samples
    mean_m = parameters['mean_m']
    std_m = parameters['std_m']
    std_cluster_shadowing = parameters['std_cluster_shadowing']
    kf = parameters['kf']

    #########################
    # Obtain impulse response

    if inter_cluster_decay > cluster_decay:
        raise ValueError("Inter cluster decay cannot be larger than cluster decay.")

    max_t = int(dynamic_range * cluster_decay * np.log(10) / 10)

    h = np.zeros(max_t, dtype=complex)

    t = 0

    while t < max_t:
        tau = 0

        max_tau = int((max_t - t) * inter_cluster_decay / cluster_decay)

        cluster_power = np.exp(-t / cluster_decay) * np.random.lognormal(mean=0, sigma=std_cluster_shadowing)

        while tau < max_tau:

            # Mean power for this ray
            mean_power = cluster_power * np.exp(-tau / inter_cluster_decay)

            # Nakagami m-factor is log normally distributed
            m = np.random.lognormal(mean_m, std_m)

            # Compute amplitude as Nakagami distributed
            a = np.sqrt(np.random.gamma(shape=m, scale=mean_power / m))

            # Compute phase as uniformly distributed
            phi = np.random.uniform(0, 2 * np.pi)

            h[t + tau] = np.array([a * np.exp(-1j * phi)])[0]

            if np.random.uniform(0, 1) < beta:
                inter_cluster_rate = inter_cluster_rate_1
            else:
                inter_cluster_rate = inter_cluster_rate_2

            tau += round(np.random.exponential(1 / inter_cluster_rate))

        t += round(np.random.exponential(1 / cluster_rate))

    ##########################
    # Add frequency dependency

    # Zero padding before FFT to avoid artifacts
    h = np.append(h, np.zeros(h.size, dtype=complex))

    H = np.fft.fft(h, norm='ortho')

    # Get frequency array in the same order as produced by the FFT
    freq = np.linspace(fc - fs / 2, fc + fs / 2, num=h.size)
    freq = np.append(freq[freq.size // 2:], freq[:freq.size // 2])

    # Calculate frequency dependency and apply
    Gf = np.power(freq, -2 * kf)
    H = np.multiply(Gf, H)

    # Inverse FFT
    h = np.fft.ifft(H, norm='ortho')

    # Remove padding
    h = h[:h.size // 2]

    ###############
    # Normalization

    h = normalize(h)

    return h


def normalize(s):
    return s / np.sqrt(energy(s))


def energy(s):
    return np.sum(np.square(np.abs(s)))


if __name__ == '__main__':
    parameters_cm1 = {
        'cluster_rate': 0.047,
        'inter_cluster_rate_1': 1.54,
        'inter_cluster_rate_2': 0.15,
        'beta': 0.095,
        'cluster_decay': 22.61,
        'inter_cluster_decay': 12.53,
        'mean_m': 0.67,
        'std_m': 0.28,
        'std_cluster_shadowing': 2.75,
        'kf': 1.12,
        'kd': 1.79,
        'std_path_shadowing': 2.22

    }

    h = gen_channel(parameters=parameters_cm1,
                          fc=(10.6E9 + 3.1E9) / 2,
                          fs=6E9,
                          dynamic_range=30)

    plt.plot(np.abs(h))
    plt.show()
