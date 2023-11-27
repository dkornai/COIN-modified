import numpy as np
from scipy.stats import dirichlet


def generate_TPM(n_contexts_init, hyp_gamma, hyp_alpha, hyp_kappa):
    """
    Initialize a transition probability matrix given the hyperparam
    
    :param input_dim: Number of possible distinct cues (length of the cue vector)
    :param n_contexts_init: Number of initial contexts
    :param hyp_gamma: Controls the decay rate of the global context transition probabilities
    :param hyp_alpha: Concentration parameter for the local context transition probabilities
    :param hyp_kappa: Context self-transition bias

    :return: Transition probability matrix 'Pi_c'
    """

    # theta_beta_c: Global context probabilities
    theta_beta_c = np.random.dirichlet([hyp_gamma] * n_contexts_init)
    
    # TPM: Context transition probability matrix
    TPM = np.zeros((n_contexts_init, n_contexts_init))
    for i in range(n_contexts_init):
        row_parameters = (hyp_alpha*theta_beta_c + hyp_kappa*(np.arange(n_contexts_init) == i)) / (hyp_alpha + hyp_kappa)
        TPM[i, :] = dirichlet.rvs(row_parameters)[0]

    print("Global context probabilities:")
    print(theta_beta_c)
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