import numpy as np
from scipy.stats import dirichlet
from copy import deepcopy

def initialize_TPM(n_contexts_init, hyp_gamma, hyp_alpha, hyp_kappa):
    """
    Initialize the model parameters.
    
    :param n_contexts_init: Number of initial contexts
    :param hyp_gamma: Controls the decay rate of the global context transition probabilities
    :param hyp_alpha: Concentration parameter for the local context transition probabilities
    :param hyp_kappa: Context self-transition bias
    :return: Initialized parameters (theta_beta_c, Pi_c)
    """
    # theta_beta_c: Global context probabilities
    theta_beta_c = np.random.dirichlet([hyp_gamma] * n_contexts_init)
    
    # Pi_c: Context transition probability matrix
    Pi_c = np.zeros((n_contexts_init, n_contexts_init))
    for i in range(n_contexts_init):
        Pi_c[i, :] = dirichlet.rvs((hyp_alpha * theta_beta_c + hyp_kappa * (np.arange(n_contexts_init) == i)) / (hyp_alpha + hyp_kappa))[0]
    
    return theta_beta_c, Pi_c


def extend_theta_beta_c(theta_beta_c, hyp_gamma):
    b = np.random.beta(1, hyp_gamma)
    theta_beta_c = np.append(theta_beta_c, theta_beta_c[-1]*(1-b))
    theta_beta_c[-2] = theta_beta_c[-2]*b

    return theta_beta_c

def extend_ss_n_t(ss_n_t):
    n_contexts = ss_n_t.shape[0]
    new_ss_n_t = np.zeros((n_contexts+1, n_contexts+1),dtype=np.int64)
    new_ss_n_t[0:n_contexts,0:n_contexts] = ss_n_t
    return new_ss_n_t

def extend_ss_n_q(ss_n_q):
    n_contexts, n_cues = ss_n_q.shape
    new_ss_n_q = np.zeros((n_contexts+1,n_cues),dtype=np.int64)
    new_ss_n_q[0:n_contexts,:] = ss_n_q
    return new_ss_n_q

def extend_theta_CEM(cue_dim, theta_CEM, hyp_lambda):
    if theta_CEM.size == 0:
        new_theta_CEM = np.full((1, cue_dim), hyp_lambda)
    else:
        n_contexts, n_cues = theta_CEM.shape
        new_theta_CEM = np.zeros((n_contexts+1,n_cues),dtype=np.float64)
        new_theta_CEM[0:n_contexts,:] = theta_CEM
        new_theta_CEM[n_contexts,:] = hyp_lambda
        
    return new_theta_CEM

'''
FUNCTIONS USED TO CALCULATE THE POSTERIOR PROPORTIONAL
'''

def exp_local_transition_prob(
        context_labels, 
        context_tm1, 
        ss_n_t, 
        theta_beta_c, 
        hyp_alpha, 
        hyp_kappa
        ):
    '''
    Calculate the expected local transition probability for each context (this acts as a prior)
    '''

    n_contexts = len(context_labels)
    delta_func = np.identity(n_contexts+1)
    
    # Calculate for contexts with established memories
    p_contexts = np.zeros(n_contexts)
    for context in range(n_contexts):
        num = (hyp_alpha*theta_beta_c[context]) + (hyp_kappa*delta_func[context, context_tm1]) + (ss_n_t[context_tm1, context])
        den = (hyp_alpha + hyp_kappa + np.sum(ss_n_t[context_tm1]))
        p_contexts[context] = num/den
    
    # Calculate for the novel context, and append
    p_novel_context = ((hyp_alpha*theta_beta_c[-1]) + (hyp_kappa*delta_func[-1, context_tm1]))/(hyp_alpha + hyp_kappa)
    p_contexts = np.append(p_contexts, p_novel_context)

    return p_contexts

def binary_vector_probability(cue, probabilities):
    '''
    Given a parameter array of probabilities for each element of the vector, and a vector, 
    return the probability of observing that vector

    p(q|theta_CEM)
    '''

    if cue.shape[0] != probabilities.shape[0]:
        raise ValueError("cue and probabilities must have the same length")
    
    if not np.all((cue == 0) | (cue == 1)):
        raise ValueError("cue must be a binary vector")
    
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("All probability values must be between 0 and 1")
    
    probability = np.prod(probabilities**cue * (1 - probabilities)**(1 - cue))
    return probability

def exp_local_cue_prob(
        context_labels, 
        cue, 
        theta_CEM, 
        hyp_lambda,
        ):
    '''
    Calculate the expected local cue probability for each context, given the parameters
    
    p(q_t|c,z_t-1)
    '''

    n_contexts = len(context_labels)

    # Calculate cue probability for contexts with established memories
    p_cues = np.zeros(n_contexts)
    for context in range(n_contexts):
        p_cues[context] = binary_vector_probability(cue, theta_CEM[context])

    # Calculate cue probability for the novel context
    p_novel_cue = binary_vector_probability(cue, np.full(len(cue), hyp_lambda))
    p_cues = np.append(p_cues, p_novel_cue)    

    return p_cues

def calc_posterior_prop(
        context_labels, 
        cue, 
        context_tm1, 
        ss_n_t,
        theta_CEM,
        theta_beta_c,
        hyp_alpha, 
        hyp_kappa,
        hyp_lambda,
        ):
    
    '''
    Main function for calculating the posterior proportional

    p(q_t|z_t-1)
    '''

    eLTP = exp_local_transition_prob(
        context_labels, context_tm1, ss_n_t, theta_beta_c, hyp_alpha, hyp_kappa)
    eLCP = exp_local_cue_prob(
        context_labels, cue, theta_CEM, hyp_lambda)
    
    posterior_prop_dist = eLTP*eLCP

    return posterior_prop_dist

'''
FUNCTIONS USED FOR RESAMPLING PARTICLES
'''

def resample_particles(particles, weights):
    ''' 
    Resample the particles according to their weights
    '''

    weights = weights/np.sum(weights) # Normalize weights
    random_indices = np.random.choice(len(weights), size=len(particles), p=weights) # Choose particles

    # Make list with new particles
    new_particles = []
    for index in random_indices:
        new_particles.append(deepcopy(particles[index]))

    return new_particles

'''
FUNCTIONS USED TO SAMPLE PARAMETERS
'''

def sample_theta_CEM(context_labels, theta_CEM, ss_n_q, ss_n_t):
    '''
    Sample each element of the CEM
    '''

    n_contexts = len(context_labels)

    for context in range(n_contexts):
        # Calculate updated parameters for the Beta distribution
        alpha = ss_n_q[context] + theta_CEM[context] + 0.1
        beta = np.sum(ss_n_t, axis = 0)[context] - ss_n_q[context] + (1 - theta_CEM[context]) + 0.1
        
        # Sample the updated probabilities from the Beta distribution
        theta_CEM[context] = np.random.beta(alpha, beta)
    
    return theta_CEM

def sample_theta_beta_c(context_labels, theta_beta_c, hyp_alpha, hyp_gamma, hyp_kappa):
    '''
    Sample the global context frequencies Beta
    '''

    n_contexts = len(context_labels)
    total_n = n_contexts*n_contexts

    # Get normalized rho parameter
    h_rho_c = hyp_kappa/(hyp_alpha+hyp_kappa)    

    valid_sample = False
    while valid_sample == False:
        try:
            # Simulate CRP for transitions based on beta parameter, as well as hyperparameters (and average over multiple runs)
            m = np.zeros((n_contexts, n_contexts))
            
            for j in range(n_contexts):
                for k in range(n_contexts):
                    fac = hyp_alpha*theta_beta_c[j] + hyp_kappa*(j == k)
                    p = np.array([(fac)/(n + fac) for n in range(1, total_n+1)])
                    m[j,k] += np.sum(np.random.binomial(n=100, p=p))/100

            # Subtract cases where it is overridden by the specialy dish (which again is generated from multiple runs)
            w = np.zeros((n_contexts, n_contexts))
            
            for j in range(n_contexts):
                p = h_rho_c/(h_rho_c + theta_beta_c[j]*(1-h_rho_c))
                for _ in range(10):
                    w[j,j] += np.sum(np.random.binomial(n = m[j,j], p = p))/10
            m -= w
        
            # Sum across rows to get alpha parameters
            m_hat = np.sum(m, axis = 1)
            m_hat = np.append(m_hat, hyp_gamma) # add parameter for novel context

            # Get new alpha values from the expected value of the dirichlet distribution
            new_theta_beta_c = m_hat/np.sum(m_hat) #np.random.dirichlet(m_hat))
            valid_sample = True

        except:
            pass
        
    return new_theta_beta_c

#theta_beta_c = np.array([0.26373571, 0.07677936, 0.65948493])
#print(sample_theta_beta_c(theta_beta_c, 3, hyp_alpha, hyp_gamma, hyp_kappa))

class Particle:
    def __init__(self, cue_dim, n_contexts_init, hyperparam):
        self.cue_dim = cue_dim
        
        # The hyperparam of the TPM prior
        self.hyperparam = hyperparam

        # The labels of currently active contexts
        self.context_labels = np.array([c for c in range(n_contexts_init)], dtype = np.int64)

        # Sufficient statistics for the parameters of the model, which are all counts initiated at 0
        self.sufficient_statistics = {
            'ss_n_t': np.zeros((n_contexts_init, n_contexts_init), dtype=np.int64),
            'ss_n_q': np.full((n_contexts_init, cue_dim), 0, dtype=np.int64),
        }

        # The parameters of the model (global context frequencies)
        if n_contexts_init > 0:
            init_theta_beta_c, init_Pi_c = initialize_TPM(
                cue_dim, 
                n_contexts_init, 
                hyperparam['hyp_gamma'],
                hyperparam['hyp_alpha'],
                hyperparam['hyp_kappa']
                )
        else:
            init_theta_beta_c = np.array([1.0])
        
        if n_contexts_init > 0:
            print('undefined behaviour')
        else:
            init_Phi_q = np.array([])

        self.parameters = {
            'theta_beta_c': init_theta_beta_c,
            'theta_CEM': init_Phi_q
        }

        # The a list of contexts the particle has been in, with the the last element being the most recent
        self.context = [int(np.random.choice(init_theta_beta_c.size, p=init_theta_beta_c))] # initialize according to global context parameters

        # 
        self.posterior_prop = None

        # The responsibilities of each context, given the cue, sufficient statistics, and model parameters
        self.weight = None


    def print_statevector(self):
        print('\nhyperparameters:')
        print(self.hyperparam)
        print('context labels:')
        print(self.context_labels)
        print('sufficient statistics:')
        print('\ttransition counts:\n', self.sufficient_statistics['ss_n_t'])
        print('\tcue counts:\n', self.sufficient_statistics['ss_n_q'])
        print('parameters:')
        print('\tglobal context prob.:\n', np.round(self.parameters['theta_beta_c'],2))
        print('\tcue emission matrix:\n', np.round(self.parameters['theta_CEM'], 2))
        print('posterior_prop:')
        print(self.posterior_prop)
        print('context:')
        print(self.context)
        print('responsibility:')
        print(self.responsibility)

    def get_hypar(self, paramname):
        return self.hyperparam[paramname] 

    def get_context(self):
        return self.context

    def set_context(self, new_context):
        self.context.append(new_context)
        
    def get_ss_n_t(self):
        return self.sufficient_statistics['ss_n_t']

    def set_ss_n_t(self, new_ss_n_t):
        self.sufficient_statistics['ss_n_t'] = np.array(new_ss_n_t, dtype=int)

    def get_ss_n_q(self):
        return self.sufficient_statistics['ss_n_q']

    def set_ss_n_q(self, new_ss_n_q):
        self.sufficient_statistics['ss_n_q'] = np.array(new_ss_n_q, dtype=int)

    def get_ss_n_c(self):
        return np.sum(self.sufficient_statistics['ss_n_t'], axis = 0)

    def get_theta_beta_c(self):
        return self.parameters['theta_beta_c']

    def set_theta_beta_c(self, new_theta_beta_c):
        self.parameters['theta_beta_c'] = np.array(new_theta_beta_c)

    def get_theta_CEM(self):
        return self.parameters['theta_CEM']

    def set_theta_CEM(self, new_theta_CEM):
        self.parameters['theta_CEM'] = new_theta_CEM

    def calculate_posterior_prop(self, cue):
        '''
        Calculate the posterior proportional distribution P(cue | state vector)
        '''
        
        self.posterior_prop = calc_posterior_prop(
            context_labels  = self.context_labels, 
            cue             = cue, 
            context_tm1     = self.context[-1], 
            ss_n_t          = self.get_ss_n_t(), 
            theta_CEM       = self.get_theta_CEM(),
            theta_beta_c    = self.get_theta_beta_c(),
            hyp_alpha       = self.get_hypar('hyp_alpha'), 
            hyp_kappa       = self.get_hypar('hyp_kappa'),
            hyp_lambda      = self.get_hypar('hyp_lambda'),
            )

    def calculate_weight(self):
        '''
        Calculate the responsibility of the given particle. This is used to weight during resampling
        '''
        self.weight = np.sum(self.posterior_prop)

    def propagate_context(self):
        '''
        Propagate the context by sampling from the posterior_prop.
        '''
        
        # Sample the context from the posterior_prop
        context_probability = self.posterior_prop
        context_probability = context_probability/np.sum(context_probability)
        predicted_context   = int(np.random.choice(context_probability.size, p = context_probability))
        
        # If the novel context was selected:
        if predicted_context not in self.context_labels:
            # Add a new context label
            self.context_labels = np.append(self.context_labels, predicted_context)

            # Add a new element to the global context probabilities
            self.set_theta_beta_c(
                extend_theta_beta_c(self.get_theta_beta_c(),self.get_hypar('hyp_gamma')))
            
            # Add a new element to the cue emission matrix
            self.set_theta_CEM(
                extend_theta_CEM(self.cue_dim, self.get_theta_CEM(), self.get_hypar('hyp_lambda')))

            # Extend the sufficient statistics
            self.set_ss_n_q(
                extend_ss_n_q(self.get_ss_n_q()))
            self.set_ss_n_t(
                extend_ss_n_t(self.get_ss_n_t()))

        # Set the context to the predicted context
        self.set_context(predicted_context)

    def propagate_sufficient_statistics(self, cue):
        '''
        Propagate the sufficient statistics (counts) to the relevant arrays.
        '''

        curr_context = self.context[-1]
        prev_context = self.context[-2]

        # Add the context transition count n_t
        self.sufficient_statistics['ss_n_t'][prev_context,curr_context] += 1

        # Add the cue counts n_q
        self.sufficient_statistics['ss_n_q'][curr_context] += cue

    def sample_parameters(self):
        '''
        Sample the parameters of the TPM and CEM
        '''

        # Sample theta_beta_c
        self.set_theta_beta_c(
            sample_theta_beta_c(
                context_labels  = self.context_labels, 
                theta_beta_c    = self.parameters['theta_beta_c'],
                hyp_alpha       = self.get_hypar('hyp_alpha'), 
                hyp_gamma       = self.get_hypar('hyp_gamma'), 
                hyp_kappa       = self.get_hypar('hyp_kappa')
                )
            )

        # Sample theta_CEM
        self.set_theta_CEM(
            sample_theta_CEM(
                context_labels  = self.context_labels, 
                theta_CEM       = self.get_theta_CEM(),
                ss_n_q          = self.get_ss_n_q(),
                ss_n_t          = self.get_ss_n_t()
                )
            )
        

# p0 = Particle(cue_dim=cue_dim, n_contexts_init=0, hyperparam=model_hyperparam)

# print('initial state:')
# p0.print_statevector()
# print('---')