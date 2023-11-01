import numpy as np
from scipy.stats import dirichlet


def generate_TPM(input_dim, n_contexts_init, h_gamma_c, h_alpha_c, h_kappa_c):
    """
    Initialize a transition probability matrix given the hyperparam
    
    :param input_dim: Number of possible distinct cues (length of the cue vector)
    :param n_contexts_init: Number of initial contexts
    :param h_gamma_c: Controls the decay rate of the global context transition probabilities
    :param h_alpha_c: Concentration parameter for the local context transition probabilities
    :param h_kappa_c: Context self-transition bias

    :return: Transition probability matrix 'Pi_c'
    """

    # Beta_c: Global context probabilities
    Beta_c = np.random.dirichlet([h_gamma_c] * n_contexts_init)
    
    # TPM: Context transition probability matrix
    TPM = np.zeros((n_contexts_init, n_contexts_init))
    for i in range(n_contexts_init):
        row_parameters = (h_alpha_c*Beta_c + h_kappa_c*(np.arange(n_contexts_init) == i)) / (h_alpha_c + h_kappa_c)
        TPM[i, :] = dirichlet.rvs(row_parameters)[0]

    print("Global context probabilities:")
    print(Beta_c)
    print("True context transition probability matrix:")
    print(TPM)

    return TPM
    

def generate_contexts_cues(Pi_c, Phi_q, t_steps):
    """
    Generate data using the specified generative model.

    :param Pi_c: Global context transition probability matrix
    :param Phi_q: Cue probability matrix
    :param t_steps: Number of timesteps to generate data for

    :return: contexts: Array of context states over time
    :return: cues: Matrix of generated sensory cues over time
    """
    n, input_dim = Phi_q.shape
    contexts = np.zeros(t_steps, dtype=np.int64)
    cues = np.zeros((t_steps, input_dim), dtype=np.int64)
    
    # Initialize the first context randomly
    contexts[0] = np.random.choice(n)
    
    for t in range(t_steps):
        # Generate the cue for the current context
        cue = Phi_q[contexts[t]]
        cues[t, :] = cue
        
        # Transition to the next context according to the relevant row of the TPM
        if t < t_steps - 1:
            trans_prob = Pi_c[contexts[t], :]
            contexts[t + 1] = np.random.choice(n, p=trans_prob)
    

    return contexts, cues