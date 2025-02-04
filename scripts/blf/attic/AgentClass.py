"""Each agent has coherence matrix and neighbors which are agents as well"""
import utilities
import numpy as np
import networkx
from scipy.stats import logistic
from config import contagion_mode, scale,number_of_bits
import utilities
from const import Constants

class Agent:

    def __init__(self, name='', tau=-1, knowledge_dim = 3, initial_state=None):
        self.neighbors = {} # neighbors will also be dictionary of agents
        self.name = name
        self.tau = tau # agent's threshold defined in initialization

        if initial_state: # if provided with initial knowledge state
            if isinstance(initial_state, list):
                self.initial_state = initial_state
                self.knowledge_state = initial_state
            else:
                raise ValueError('list expected')
        self.dissonance_lst = None
        self.next_state_probs = None
        self.next_state_onehot = None
        self.next_state = None
        self.soc_probs = None
        self.state_disagreements = None

    def reset_state(self,nstate = None):
        if nstate is not None:
            self.initial_state = nstate
        self.knowledge_state = self.initial_state

    def add_neighbors(self, neighbor_agent):
        """add neigbhor agent to the dictionary of neighbors
        which has neighbor agent name for key and agent itself as value"""
        self.neighbors[neighbor_agent.name] = neighbor_agent
        neighbor_agent.neighbors[self.name] = self # neighbors also get new neighbors

    def get_neighbors(self):
        """return all neighbors for the agent"""
        return self.neighbors

    def get_neighbors_name(self):
        return [n.name for k, n in self.neighbors.items()]

    def remove_neighbors(self, name):
        """remove neighbor for the agent with name"""
        if name in self.neighbors:
            del self.neighbors[name]
            print('removed neighbor')
        else:
            print('neighbor not found')


    def update_knowledge(self, alpha, txn, bit_matrix):
        """Takes alpha and transition matrix (expects a 2d array)"""
        # first convert state binary to int to get the row in coherence matrix
        row_ptr = utilities.bool2int(self.knowledge_state)
        # get the corresponding probabilites from the matrix
        coh_prob_tx = txn[row_ptr]
        ones_list = np.zeros(number_of_bits)
        dissonance_list = []
        disagreements = []
        
        for index, curr_bit_state in enumerate(self.knowledge_state):
            # now look for neighbors who disagree in this bit value

            neigh_disagreement_count = self.count_dissimilar_neighbors(index)

            # compute d as (# of neighbors disagree on bit/# of neighbors)
            if len(self.neighbors) > 0:
                d = neigh_disagreement_count/len(self.neighbors)
            else:
                d = 0

            #TODO: Handle the viral parameter - in general, if d = 0 and viral is set,
            #TODO: it should not be possible to make that transition
            
            if d > 0:
                dissonance = utilities.sigmoid(d, self.tau)
                
            else:
                dissonance = 0
                
            dissonance_list.append(dissonance)
            
            # keeping track of disagreement of bits/total neighbors
            disagreements.append(d)
            # transition probabilities given social pressure for moving to a state
            # with a '1' at this bit
            ones_list[index] = (1-dissonance if curr_bit_state else dissonance)

        zeros_list = 1-ones_list
        tmp_soc_mat = ones_list * bit_matrix  + zeros_list * (1-bit_matrix)

        # Probabilities for each state given social pressure
        soc_prob_tx = np.prod(tmp_soc_mat,1)
        #TODO logs soc_prob_tx for each agent at each time step
            
        probs = alpha * soc_prob_tx + (1-alpha)*coh_prob_tx
        self.next_state_probs = probs
        self.soc_probs = soc_prob_tx
        self.next_state = utilities.int2bool(np.random.choice(range(2**number_of_bits),1,p=probs)[0],number_of_bits)
        self.dissonance_lst = dissonance_list
        self.state_disagreements = disagreements
        
        return soc_prob_tx
    
    def count_dissimilar_neighbors(self, kbit):
        count = 0
        for name, agent in self.neighbors.items():
            if agent.knowledge_state[kbit] != self.knowledge_state[kbit]:
                count += 1

        return count


    def __str__(self):
        print('-'*30)
        s = 'agent name : ' + self.name
        s += '\nknowledge_state: [' + ', '.join(map(str, self.knowledge_state)) + ']'
        for n,v in self.neighbors.items():
            s += '\n'
            s += self.name + ' <-> ' + v.name

        return s
