import numpy as np
import pulp as p
import time
import math
import sys

class utils:
    def __init__(self,eps, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,ACTIONS,CONSTRAINT,Cb):
        self.P = P.copy()
        self.R = R.copy()
        self.C = C.copy()
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.N_STATES = N_STATES
        self.ACTIONS = ACTIONS
        self.eps = eps
        self.delta = delta
        self.M = M
        self.Cb = Cb
        #self.ENV_Q_VALUES = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        
        self.P_hat = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        #self.P_tilde = np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        
        #self.R_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.C_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.R_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.C_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        
        
        self.NUMBER_OF_OCCURANCES = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.NUMBER_OF_OCCURANCES_p = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob_1 = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_2 = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_T = {}
        self.Psparse = [[[] for i in self.ACTIONS] for j in range(self.N_STATES)]
        
        self.mu = np.zeros(self.N_STATES)
        self.mu[0] = 1.0
        self.CONSTRAINT = CONSTRAINT
        
        self.R_Tao = {}
        for s in range(self.N_STATES):
            l = len(self.ACTIONS[s])
            self.R_Tao[s] = np.zeros(l)
            
        self.C_Tao = {}
        for s in range(self.N_STATES):
            l = len(self.ACTIONS[s])
            self.C_Tao[s] = np.zeros(l)
        
        for s in range(self.N_STATES):
            self.P_hat[s] = {} # estimated transition probabilities
            l = len(self.ACTIONS[s])
            self.NUMBER_OF_OCCURANCES[s] = np.zeros(l) # initialize the number of occurences of [s][a]
            self.beta_prob_1[s] = np.zeros(l)
            self.beta_prob_2[s] = np.zeros(l)
            self.beta_prob_T[s] = np.zeros(l)
            self.NUMBER_OF_OCCURANCES_p[s] = np.zeros((l, N_STATES)) # initialize the number of occurences of [s][a, s']
            self.beta_prob[s] = np.zeros((l, N_STATES)) # [s][a, s']
            
            for a in self.ACTIONS[s]:
                self.P_hat[s][a] = np.zeros(self.N_STATES) 
                for s_1 in range(self.N_STATES):
                    if self.P[s][a][s_1] > 0:
                        self.Psparse[s][a].append(s_1)
    

    def step(self,s, a, h):
        probs = np.zeros((self.N_STATES))
        for next_s in range(self.N_STATES):
            probs[next_s] = self.P[s][a][next_s]
        next_state = int(np.random.choice(np.arange(self.N_STATES),1,replace=True,p=probs)) # find next_state based on the transition probabilities
        rew = self.R[s][a]
        cost = self.C[s][a]
        return next_state,rew, cost


    def setCounts(self,ep_count_p,ep_count):
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                self.NUMBER_OF_OCCURANCES[s][a] += ep_count[s, a]
                for s_ in range(self.N_STATES):
                    self.NUMBER_OF_OCCURANCES_p[s][a, s_] += ep_count_p[s, a, s_]


    def compute_confidence_intervals(self,ep, mode):
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.beta_prob[s][a, :] = np.ones(self.N_STATES)
                    self.beta_prob_T[s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a],1)) # not sure what is beta_prob_T used for?
                else:
                    if mode == 2:
                        self.beta_prob[s][a, :] = min(np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a], 1)), 1)*np.ones(self.N_STATES)
                    elif mode == 3:
                        self.beta_prob_T[s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a],1))
                        
                    for s_1 in range(self.N_STATES):
                        if mode == 0:
                            # DOPE policy, which equation?
                            self.beta_prob[s][a,s_1] = min(np.sqrt(ep*self.P_hat[s][a][s_1]*(1-self.P_hat[s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[s][a],1)) + ep/(max(self.NUMBER_OF_OCCURANCES[s][a],1)), ep/(max(np.sqrt(self.NUMBER_OF_OCCURANCES[s][a]),1)), 1)
                        
                        elif mode == 1:
                            # safe base policy 
                            # equation (5) in the paper to calculate the confidence interval for P
                            self.beta_prob[s][a, s_1] = min(2*np.sqrt(ep*self.P_hat[s][a][s_1]*(1-self.P_hat[s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[s][a],1)) + 14*ep/(3*max(self.NUMBER_OF_OCCURANCES[s][a],1)), 1)
                        
                self.beta_prob_1[s][a] = max(self.beta_prob[s][a, :])
                self.beta_prob_2[s][a] = sum(self.beta_prob[s][a, :])


    def update_empirical_model(self,ep):
        # ep is not used here

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.P_hat[s][a] = 1/self.N_STATES*np.ones(self.N_STATES) # uniform distribution for unvisited state-action pairs
                else:
                    for s_1 in range(self.N_STATES):
                        self.P_hat[s][a][s_1] = self.NUMBER_OF_OCCURANCES_p[s][a,s_1]/(max(self.NUMBER_OF_OCCURANCES[s][a],1)) #calculate the estimated/empirical probabilities
                    self.P_hat[s][a] /= np.sum(self.P_hat[s][a]) # normalize the probabilities

                if abs(sum(self.P_hat[s][a]) - 1)  >  0.001: # sanity check  after updating the probabilities
                    print("empirical is wrong")
                    print(self.P_hat)
                    
                    
    def update_costs(self):
        alpha_r = (self.N_STATES*self.EPISODE_LENGTH) + 4*self.EPISODE_LENGTH*(self.N_STATES*self.EPISODE_LENGTH)/(self.CONSTRAINT-self.Cb)
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                self.R_Tao[s][a] = self.R[s][a] + alpha_r * self.beta_prob_T[s][a]
                self.C_Tao[s][a] = self.C[s][a] + (self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]





    def compute_opt_LP_Unconstrained(self, ep):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a]
    
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')
        
        #Objective function
        list_1 = [self.R[s][a] for s in range(self.N_STATES) for a in self.ACTIONS[s]] * self.EPISODE_LENGTH
        list_2 = [q[(h,s,a)] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]

        opt_prob += p.lpDot(list_1,list_2)
                  
        #opt_prob += p.lpSum([q[(h,s,a)]*self.R[s,a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS)])
                  
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0
        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
                
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001, msg = 0))
        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        print("printing best value")
        print(p.value(opt_prob.objective))
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                          
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]:
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                    probs = opt_policy[s,h,:]
                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0
                                                                          
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
        
        val_policy = 0
        con_policy = 0
       
        for h in range(self.EPISODE_LENGTH):
         for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                con_policy  += opt_q[h,s,a]*self.C[s][a]
                if opt_q[h,s,a] < 0:
                        opt_q[h,s,a] = 0
                elif opt_q[h,s,a] > 1:
                    opt_q[h,s,a] = 1.0
                
                val_policy += opt_q[h,s,a]*self.R[s][a]
                    
        print("value from the UnconLPsolver")
        print(val_policy,con_policy)
                                                                          
        return opt_policy, value_of_policy, cost_of_policy, q_policy
                                                                                  
                                                                                  
    def compute_opt_LP_Constrained(self, ep):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a], note here the action dimension is N_STATES
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a]
                                                                                  
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
                                                                                          
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')

        opt_prob += p.lpSum([q[(h,s,a)]*self.R[s][a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]])
            
        opt_prob += p.lpSum([q[(h,s,a)]*self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0
            
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0

        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001, msg = 0))
        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        print("printing best value constrained")
        print(p.value(opt_prob.objective))
                                                                                                                  
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]: # for actions that are not available in the current state, their probability is 0
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                        if math.isnan(opt_policy[s,h,a]):
                            opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                        elif opt_policy[s,h,a] > 1.0:
                            print("invalid value printing")
                            print(opt_policy[s,h,a])
                #probs = opt_policy[s,h,:]
                #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0
        
        val_policy = 0
        con_policy = 0
        for h in range(self.EPISODE_LENGTH):
         for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if opt_q[h,s,a] < 0:
                        opt_q[h,s,a] = 0
                elif opt_q[h,s,a] > 1:
                    opt_q[h,s,a] = 1.0
                    
                con_policy  += opt_q[h,s,a]*self.C[s][a]
                val_policy += opt_q[h,s,a]*self.R[s][a]
        print("value from the conLPsolver")
        print(val_policy,con_policy)

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                          
        return opt_policy, value_of_policy, cost_of_policy, q_policy
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  

    def compute_extended_LP(self,ep, cb):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES,self.N_STATES)) #[h,s,a,s_]
        #create problem variables
        
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous')
            
        # r_k = {}
        # for s in range(self.N_STATES):
        #     l = len(self.ACTIONS[s])
        #     r_k[s] = np.zeros(l)
        #     for a in self.ACTIONS[s]:
        #         r_k[s][a] = self.R[s][a] + self.EPISODE_LENGTH**2/(self.CONSTRAINT - cb)* self.beta_prob_2[s][a]

        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])
        #Constraints
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*(self.C[s][a] + self.EPISODE_LENGTH*self.beta_prob_2[s][a]) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0

        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
                                                                                                                                                                                                              
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
                                                                                                                                                                                                                      
                                                                                                                                                                                                                      #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                                                                                                                                                                                                                                        
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01, msg = 0))
                                                                                                                                                                                                                                      
        if p.LpStatus[status] != 'Optimal':
            print(p.LpStatus[status])
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))
                                                                                                                                                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001:
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] < -0.001:
                            print("invalid value")
                            sys.exit()
                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                  
        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)
                                                                                                                                                                                                                                                                                  
                                                                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                    sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                if math.isnan(sum_prob):
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                  
        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy
    
    
    def compute_LP_Tao(self, ep, cb):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a]
                                                                                  
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
                                                                                          
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')
        
        
        
        
        # alpha_r = 1 + self.N_STATES*self.EPISODE_LENGTH + 4*self.EPISODE_LENGTH*(1+self.N_STATES*self.EPISODE_LENGTH)/(self.CONSTRAINT-cb)
        # for s in range(self.N_STATES):
        #     l = len(self.ACTIONS[s])
        #     self.R_Tao[s] = np.zeros(l)
        #     for a in self.ACTIONS[s]:
        #         self.R_Tao[s][a] = self.R[s][a] - alpha_r * self.beta_prob_T[s][a]
                
       
        
        # for s in range(self.N_STATES):
        #     l = len(self.ACTIONS[s])
        #     self.C_Tao[s] = np.zeros(l)
        #     for a in self.ACTIONS[s]:
        #         self.C_Tao[s][a] = self.C[s][a] + (1 + self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]
                
                
        
        #print(alpha_r)
        
        # for s in range(self.N_STATES):
        #     for a in range(self.N_ACTIONS):
        #         self.R_Tao[s][a] = self.R[s][a] - alpha_r * self.beta_prob_T[s][a]
        #         self.C_Tao[s][a] = self.C[s][a] + (self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]
        
        

        opt_prob += p.lpSum([q[(h,s,a)]*(self.R_Tao[s][a]) for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]])
            
        opt_prob += p.lpSum([q[(h,s,a)]*(self.C_Tao[s][a]) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0
            
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P_hat[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0

        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001, msg = 0))
        #if p.LpStatus[status] != 'Optimal':
            #return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status]
        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        # print("printing best value constrained")
        # print(p.value(opt_prob.objective))
                                                                                                                  
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]:
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                        if math.isnan(opt_policy[s,h,a]):
                            opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                        elif opt_policy[s,h,a] > 1.0 and opt_policy[s,h,a]<1.1:
                            opt_policy[s,h,a] = 1.0
                        elif opt_policy[s,h,a]>1.1:
                            print("invalid value printing",opt_policy[s,h,a])
                            #print(opt_policy[s,h,a])
                #probs = opt_policy[s,h,:]
                #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0
        
        
        for h in range(self.EPISODE_LENGTH):
         for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if opt_q[h,s,a] < 0:
                        opt_q[h,s,a] = 0
                elif opt_q[h,s,a] > 1:
                    opt_q[h,s,a] = 1.0
                    

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                          
        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status]
                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                          
    def compute_extended_LP1(self,ep,alg):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES,self.N_STATES)) #[h,s,a,s_]

        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                  
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous')
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])
                                                                                                                                                                                                                                                                                                                                                                      
        #Constraints
        if alg == 1:
            opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0
                                                                                                                                                                                                                                                                                                                                                                      
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
                                                                                                                                                                                                                                                                                                                                                                                      
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                              #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0

        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01, msg = 0))
                                                                                                                                                                                                                                                                                                                                                                                                              
        if p.LpStatus[status] != 'Optimal':
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001:
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] < -0.001:
                            print("invalid value")
                            sys.exit()

        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                    sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                if math.isnan(sum_prob):
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)

        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy

    def compute_extended_ucrl2(self,ep):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES,self.N_STATES)) #[h,s,a,s_]
        
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
        
        
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous')
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])
        
        #Constraints
        #opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0
        
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
        
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0

        #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0

        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01, msg = 0))

        if p.LpStatus[status] != 'Optimal':
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001:
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] < -0.001:
                            print("invalid value")
                            sys.exit()
    
        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                    sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                if math.isnan(sum_prob):
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob
        
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy)

        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy

    def FiniteHorizon_Policy_evaluation(self,Px,policy,R,C):
        q = np.zeros((self.N_STATES,self.EPISODE_LENGTH, self.N_STATES))
        v = np.zeros((self.N_STATES, self.EPISODE_LENGTH))
        c = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        P_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES))
        R_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        C_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))

        for s in range(self.N_STATES):
            x = 0
            for a in self.ACTIONS[s]:
                x += policy[s, self.EPISODE_LENGTH - 1, a]*C[s][a]
            c[s,self.EPISODE_LENGTH-1] = x #np.dot(policy[s,self.EPISODE_LENGTH-1,:], self.C[s])

            for a in self.ACTIONS[s]:
                q[s, self.EPISODE_LENGTH-1, a] = R[s][a]
            v[s,self.EPISODE_LENGTH-1] = np.dot(q[s, self.EPISODE_LENGTH-1, :], policy[s, self.EPISODE_LENGTH-1, :])

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                x = 0
                y = 0
                for a in self.ACTIONS[s]:
                    x += policy[s,h,a]*R[s][a]
                    y += policy[s,h,a]*C[s][a]
                R_policy[s,h] = x
                C_policy[s,h] = y
                for s_1 in range(self.N_STATES):
                    z = 0
                    for a in self.ACTIONS[s]:
                        z += policy[s,h,a]*Px[s][a][s_1]
                    P_policy[s,h,s_1] = z #np.dot(policy[s,h,:],Px[s,:,s_1])

        for h in range(self.EPISODE_LENGTH-2,-1,-1):
            for s in range(self.N_STATES):
                c[s,h] = C_policy[s,h] + np.dot(P_policy[s,h,:],c[:,h+1])
                for a in self.ACTIONS[s]:
                    z = 0
                    for s_ in range(self.N_STATES):
                        z += Px[s][a][s_] * v[s_, h+1]
                    q[s, h, a] = R[s][a] + z
                v[s,h] = np.dot(q[s, h, :],policy[s, h, :])
        #print("evaluation",v)
                

        return q, v, c

    def compute_qVals_EVI(self, Rx):
        # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.EPISODE_LENGTH] = np.zeros(self.N_STATES)
        p_tilde = {}
        for h in range(self.EPISODE_LENGTH):
            j = self.EPISODE_LENGTH - h - 1
            qMax[j] = np.zeros(self.N_STATES)
            for s in range(self.N_STATES):
                qVals[s, j] = np.zeros(len(self.ACTIONS[s]))
                p_tilde[s] = {}
                for a in self.ACTIONS[s]:
                    #rOpt = R[s, a] + R_slack[s, a]
                    p_tilde[s][a] = np.zeros(self.N_STATES)
                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = self.P_hat[s][a].copy()
                    if pOpt[pInd[self.N_STATES - 1]] + self.beta_prob_1[s][a] * 0.5 > 1:
                        pOpt = np.zeros(self.N_STATES)
                        pOpt[pInd[self.N_STATES - 1]] = 1
                    else:
                        pOpt[pInd[self.N_STATES - 1]] += self.beta_prob_1[s][a] * 0.5

                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    qVals[s, j][a] = Rx[s][a] + np.dot(pOpt, qMax[j + 1])
                    p_tilde[s][a] = pOpt.copy()

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax, p_tilde

    def  deterministic2stationary(self, policy):
        stationary_policy = np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))
        for s in range(self.N_STATES):
            for h in range(self.EPISODE_LENGTH):
                a = int(policy[s, h])
                stationary_policy[s, h, a] = 1

        return stationary_policy

    def update_policy_from_EVI(self, Rx):
        qVals, qMax, p_tilde = self.compute_qVals_EVI(Rx)
        policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                Q = qVals[s,h]
                policy[s,h] = np.random.choice(np.where(Q==Q.max())[0])

        self.P_tilde = p_tilde.copy()

        policy = self.deterministic2stationary(policy)

        q, v, c = self.FiniteHorizon_Policy_evaluation(self.P, policy)

        return policy, v, c, q
