import numpy as np
from copy import deepcopy

def sample_global_context_prob(Beta_c, n_contexts, h_alpha_c, h_gamma_c, h_kappa_c):
    #print('\n sampling Beta c...')
    
    valid_sample = False
    while valid_sample == False:
        try:
            # Get normalized rho parameter
            h_rho_c = h_kappa_c/(h_alpha_c+h_kappa_c)
            
            # Simulate CRP for transitions based on beta parameter, as well as hyperparameters (and average over multiple runs)
            total_n = n_contexts*n_contexts
            m = np.zeros((n_contexts, n_contexts))
            for _ in range(100):
                for j in range(n_contexts):
                    for k in range(n_contexts):
                        fac = h_alpha_c*Beta_c[j] + h_kappa_c*(j == k)
                        for n in range(1, total_n+1):
                            m[j,k] += np.random.binomial(
                                n = 1, 
                                p = (fac)/(n + fac) 
                                )
            
            m /= 100
            #print(m)

            # Subtract cases where it is overridden by the specialy dish (which again is generated from multiple runs)
            w = np.zeros((n_contexts, n_contexts))
            for _ in range(10):
                for i in range(n_contexts):
                    w[i,i] += np.sum(np.random.binomial(
                        n = m[i,i], 
                        p = h_rho_c/(h_rho_c + Beta_c[i]*(1-h_rho_c)))
                        )
            w /= 10
            m -= w
            
            #print(w)
            #print(m)
            # Sum across rows to get alpha parameters
            m_hat = np.sum(m, axis = 1)
            m_hat = np.append(m_hat, h_gamma_c) # add parameter for novel context
            #print(m_hat)

            # Get new alpha values from the expected value of the dirichlet distribution
            new_Beta_c = m_hat/np.sum(m_hat)
            valid_sample = True

        except:
            pass
        
    return new_Beta_c

#Beta_c = np.array([0.26373571, 0.07677936, 0.65948493])
#print(sample_global_context_prob(Beta_c, 3, h_alpha_c, h_gamma_c, h_kappa_c))

def extend_Beta_c(Beta_c, h_gamma_c):
    b = np.random.beta(1, h_gamma_c)
    Beta_c = np.append(Beta_c, Beta_c[-1]*(1-b))
    Beta_c[-2] = Beta_c[-2]*b

    return Beta_c

#print(extend_Beta_c(Beta_c, h_gamma_c))

def extend_transition_counts(transition_counts):
    n_contexts = transition_counts.shape[0]
    new_transition_counts = np.zeros((n_contexts+1, n_contexts+1),dtype=np.int64)
    new_transition_counts[0:n_contexts,0:n_contexts] = transition_counts
    return new_transition_counts

def extend_cue_counts(cue_counts):
    n_contexts, n_cues = cue_counts.shape
    new_cue_counts = np.zeros((n_contexts+1,n_cues),dtype=np.int64)
    new_cue_counts[0:n_contexts,:] = cue_counts
    return new_cue_counts

def extend_context_counts(context_counts):
    return np.append(context_counts, 0)

def initialize_TPM(input_dim, n_contexts_init, h_gamma_c, h_alpha_c, h_kappa_c):
    """
    Initialize the model parameters.
    
    :param input_dim: Number of possible distinct cues (length of the cue vector)
    :param n_contexts_init: Number of initial contexts
    :param h_gamma_c: Controls the decay rate of the global context transition probabilities
    :param h_alpha_c: Concentration parameter for the local context transition probabilities
    :param h_kappa_c: Context self-transition bias
    :return: Initialized parameters (Beta_c, Pi_c)
    """
    # Beta_c: Global context probabilities
    Beta_c = np.random.dirichlet([h_gamma_c] * n_contexts_init)
    
    # Pi_c: Context transition probability matrix
    Pi_c = np.zeros((n_contexts_init, n_contexts_init))
    for i in range(n_contexts_init):
        Pi_c[i, :] = dirichlet.rvs((h_alpha_c * Beta_c + h_kappa_c * (np.arange(n_contexts_init) == i)) / (h_alpha_c + h_kappa_c))[0]
    
    return Beta_c, Pi_c

def extend_cue_emission_matrix(input_dim, cue_emission_matrix, h_prior_cue = 0.5):
    if cue_emission_matrix.size == 0:
        new_cue_emission_matrix = np.full((1, input_dim), h_prior_cue)
    else:
        n_contexts, n_cues = cue_emission_matrix.shape
        new_cue_emission_matrix = np.zeros((n_contexts+1,n_cues),dtype=np.float64)
        new_cue_emission_matrix[0:n_contexts,:] = cue_emission_matrix
        new_cue_emission_matrix[n_contexts,:] = h_prior_cue
        
    return new_cue_emission_matrix

def sample_cue_emission_matrix(context_labels, cue_emission_matrix, cue_counts, context_counts):
    n_contexts = len(context_labels)

    for context in range(n_contexts):
        # Calculate updated parameters for the Beta distribution
        alpha_prior = cue_counts[context] + cue_emission_matrix[context] + 0.1
        beta_prior = context_counts[context] - cue_counts[context] + (1 - cue_emission_matrix[context]) + 0.1
        
        # Sample the updated probabilities from the Beta distribution
        cue_emission_matrix[context] = np.random.beta(alpha_prior, beta_prior)
    
    return cue_emission_matrix

def exp_local_transition_prob(context_labels, context_tm1, Pi_count, Beta_c, h_alpha_c, h_kappa_c):
    n_contexts = len(context_labels)
    delta_func = np.identity(n_contexts+1)
    
    # Calculate for contexts with established memories
    p_contexts = np.zeros(n_contexts)
    for context in range(n_contexts):
        num = (h_alpha_c*Beta_c[context]) + (h_kappa_c*delta_func[context, context_tm1]) + (Pi_count[context_tm1, context])
        den = (h_alpha_c + h_kappa_c + np.sum(Pi_count[context_tm1]))
        p_contexts[context] = num/den
    
    # Calculate for the novel context
    p_novel_context = ((h_alpha_c*Beta_c[-1]) + (h_kappa_c*delta_func[-1, context_tm1]))/(h_alpha_c + h_kappa_c)
    p_contexts = np.append(p_contexts, p_novel_context)

    return p_contexts


def binary_vector_probability(X, probabilities):
    if X.shape[0] != probabilities.shape[0]:
        raise ValueError("X and probabilities must have the same length")
    
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("X must be a binary vector")
    
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("All probability values must be between 0 and 1")
    
    probability = np.prod(probabilities**X * (1 - probabilities)**(1 - X))
    return probability

def exp_local_cue_prob(context_labels, cue, cue_emission_matrix, h_prior_cue = 0.5):
    n_contexts = len(context_labels)

    # Calculate cue probability for contexts with established memories
    p_cues = np.zeros(n_contexts)
    for context in range(n_contexts):
        p_cues[context] = binary_vector_probability(cue, cue_emission_matrix[context])

    # Calculate cue probability for the novel context
    p_cues = np.append(p_cues, binary_vector_probability(cue, np.full(len(cue), h_prior_cue)))    

    return p_cues

def get_responsibility(
        context_labels, 
        cue, 
        most_recent_context, 
        transition_counts, 
        cue_counts, 
        context_counts,
        global_context_prob,
        h_alpha_c, 
        h_kappa_c,
        cue_emission_matrix
        ):

    p_contexts = exp_local_transition_prob(context_labels, most_recent_context, transition_counts, global_context_prob, h_alpha_c, h_kappa_c)
    p_cues = exp_local_cue_prob(context_labels, cue, cue_emission_matrix)
    joint_dist = p_contexts*p_cues

    return joint_dist

def resample(particles, weights):
    weights = weights/np.sum(weights)
    random_indices = np.random.choice(len(weights), size=len(particles), p=weights)
    #print(np.round(weights, 4))
    #print(random_indices)
    new_particles = []
    for index in random_indices:
        new_particles.append(deepcopy(particles[index]))

    return new_particles

class Particle:
    def __init__(self, input_dim, n_contexts_init, hyperparam):
        self.cue_dim = input_dim
        
        # The hyperparam of the TPM prior
        self.hyperparam = hyperparam

        # The labels of currently active contexts
        self.context_labels = np.array([c for c in range(n_contexts_init)], dtype = np.int64)

        # Sufficient statistics for the parameters of the model, which are all counts initiated at 0
        self.sufficient_statistics = {
            'transition_counts': np.zeros((n_contexts_init, n_contexts_init), dtype=np.int64),
            'cue_counts': np.full((n_contexts_init, input_dim), 0, dtype=np.int64),
            'context_counts': np.full(n_contexts_init, 0, dtype=np.int64)
        }

        # The parameters of the model (global context frequencies)
        if n_contexts_init > 0:
            init_Beta_c, init_Pi_c = initialize_TPM(
                input_dim, 
                n_contexts_init, 
                hyperparam['h_gamma_c'],
                hyperparam['h_alpha_c'],
                hyperparam['h_kappa_c']
                )
        else:
            init_Beta_c = np.array([1.0])
        
        if n_contexts_init > 0:
            print('undefined behaviour')
        else:
            init_Phi_q = np.array([])

        self.parameters = {
            'global_context_prob': init_Beta_c,
            'cue_emission_matrix': init_Phi_q
        }

        # The a list of contexts the particle has been in, with the the last element being the most recent
        self.context = [int(np.random.choice(init_Beta_c.size, p=init_Beta_c))] # initialize according to global context parameters

        # 
        self.joint = None

        # The responsibilities of each context, given the cue, sufficient statistics, and model parameters
        self.responsibility = None


    def print_statevector(self):
        print('\nhyperparameters:')
        print(self.hyperparam)
        print('context labels:')
        print(self.context_labels)
        print('sufficient statistics:')
        print('\ttransition counts:\n', self.sufficient_statistics['transition_counts'])
        print('\tcue counts:\n', self.sufficient_statistics['cue_counts'])
        print('\tcontext counts:\n', self.sufficient_statistics['context_counts'])
        print('parameters:')
        print('\tglobal context prob.:\n', np.round(self.parameters['global_context_prob'],2))
        print('\tcue emission matrix:\n', np.round(self.parameters['cue_emission_matrix'], 2))
        print('joint:')
        print(self.joint)
        print('context:')
        print(self.context)
        print('responsibility:')
        print(self.responsibility)

    def get_hyperparameter(self, paramname):
        return self.hyperparam[paramname] 

    def get_context(self):
        return self.context

    def set_context(self, new_context):
        self.context.append(new_context)
        
    def get_transition_counts(self):
        return self.sufficient_statistics['transition_counts']

    def set_transition_counts(self, new_transition_counts):
        self.sufficient_statistics['transition_counts'] = np.array(new_transition_counts, dtype=int)

    def get_cue_counts(self):
        return self.sufficient_statistics['cue_counts']

    def set_cue_counts(self, new_cue_counts):
        self.sufficient_statistics['cue_counts'] = np.array(new_cue_counts, dtype=int)

    def get_context_counts(self):
        return self.sufficient_statistics['context_counts']

    def set_context_counts(self, new_context_counts):
        self.sufficient_statistics['context_counts'] = np.array(new_context_counts, dtype=int)

    def get_global_context_prob(self):
        return self.parameters['global_context_prob']

    def set_global_context_prob(self, new_global_context_prob):
        self.parameters['global_context_prob'] = np.array(new_global_context_prob)

    def get_cue_emission_matrix(self):
        return self.parameters['cue_emission_matrix']

    def set_cue_emission_matrix(self, new_cue_emission_matrix):
        self.parameters['cue_emission_matrix'] = new_cue_emission_matrix

    def calculate_joint(self, cue):
        '''
        Calculate the joint distribution P(cue | state vector) which yields a weight for each context
        '''

        if not self.context:
            raise ValueError("Context list is empty")
        
        self.joint = get_responsibility(
            context_labels = self.context_labels, 
            cue = cue, 
            most_recent_context = self.context[-1], 
            transition_counts = self.get_transition_counts(), 
            cue_counts = self.get_cue_counts(), 
            context_counts = self.get_context_counts(),
            global_context_prob = self.get_global_context_prob(),
            h_alpha_c = self.get_hyperparameter('h_alpha_c'), 
            h_kappa_c = self.get_hyperparameter('h_kappa_c'),
            cue_emission_matrix = self.get_cue_emission_matrix()
            )

    def calculate_responsibility(self):
        '''
        Calculate the responsibility of the given particle. This is used to weight during resampling
        '''
        self.responsibility = np.sum(self.joint)

    def get_responsibility(self):
        return self.responsibility

    def propagate_context(self):
        '''
        Propagate the context by sampling from the joint.
        '''
        
        # Sample the context from the joint
        context_probability = self.joint
        context_probability = context_probability/np.sum(context_probability)
        predicted_context = int(np.random.choice(context_probability.size, p = context_probability))
        
        # If the novel context was selected:
        if predicted_context not in self.context_labels:
            # Add a new context label
            self.context_labels = np.append(self.context_labels, predicted_context)

            # Add a new element to the global context probabilities
            self.set_global_context_prob(extend_Beta_c(self.get_global_context_prob(),self.get_hyperparameter('h_gamma_c')))
            
            # Add a new element to the cue emission matrix
            self.set_cue_emission_matrix(extend_cue_emission_matrix(self.cue_dim, self.get_cue_emission_matrix()))

            # Extend the sufficient statistics
            self.set_context_counts(extend_context_counts(self.get_context_counts()))
            self.set_cue_counts(extend_cue_counts(self.get_cue_counts()))
            self.set_transition_counts(extend_transition_counts(self.get_transition_counts()))

        # Set the context to the predicted context
        self.set_context(predicted_context)

    def propagate_sufficient_statistics(self, cue):
        '''
        Propagate the sufficient statistics (counts) to the relevant arrays.
        '''

        curr_context = self.context[-1]
        prev_context = self.context[-2]
        
        # Add the context count
        self.sufficient_statistics['context_counts'][curr_context] += 1

        # Add the context transition count
        self.sufficient_statistics['transition_counts'][prev_context,curr_context] += 1

        # Add the cue counts
        self.sufficient_statistics['cue_counts'][curr_context] += cue

    def sample_parameters(self):
        '''
        Sample the parameters of the TPM and CEM
        '''

        # Sample global_context_prob
        self.set_global_context_prob(
            sample_global_context_prob(
                Beta_c = self.parameters['global_context_prob'],
                n_contexts = len(self.context_labels), 
                h_alpha_c = self.get_hyperparameter('h_alpha_c'), 
                h_gamma_c = self.get_hyperparameter('h_gamma_c'), 
                h_kappa_c = self.get_hyperparameter('h_kappa_c')
                )
            )

        # Sample cue_emission_matrix
        self.set_cue_emission_matrix(
            sample_cue_emission_matrix(
                context_labels = self.context_labels, 
                cue_emission_matrix = self.get_cue_emission_matrix(),
                cue_counts = self.get_cue_counts(),
                context_counts = self.get_context_counts()
                )
            )
        

# p0 = Particle(input_dim=input_dim, n_contexts_init=0, hyperparam=model_hyperparam)

# print('initial state:')
# p0.print_statevector()
# print('---')