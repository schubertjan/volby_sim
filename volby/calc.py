import numpy as np
import pandas as pd
from scipy import stats

class Calc():
    def __init__(self, n_seats, results):
        self.seats = n_seats
        self.results = results

    def dhont_(self, results, n_seats):
        # create a matrix where to store results
        results_rounds = np.arange(results.shape[0] * n_seats).reshape(results.shape[0], n_seats)
        for i in range(results_rounds.shape[1]):
            results_rounds[:, i] = results / (i+1)
        #decide who should get seats
        results_rounds_flat = results_rounds.flatten()
        for i in range(n_seats):
            aux_max = results_rounds_flat.argmax()
            results_rounds_flat[aux_max] = -99
        #count the seats for each party
        results_rounds_seats = results_rounds_flat.reshape(results.shape[0], n_seats)
        seats = np.zeros(results.shape[0])
        for i in range(results_rounds_seats.shape[0]):
            seats[i] = np.sum(results_rounds_seats[i,:] == -99)
        return seats.astype(int) #return as integers
    
    def assign_seats(self, threshold, inplace = True):
        seats_df = pd.DataFrame(index = self.seats.keys(), data = self.seats)
        results_df = pd.DataFrame(data = self.results)
        results_df.columns = seats_df.columns
        
        #first only use parties that have above the threshold
        total_votes = results_df.sum(axis = 1)
        prc_votes = total_votes / total_votes.sum()

        results_df = results_df.loc[prc_votes >= threshold, :]
        # crete a df where to store the results
        seats_final = pd.DataFrame(columns = self.seats.keys(), 
                                    data = np.zeros(results_df.shape[0] * results_df.shape[1]).reshape(results_df.shape[0], results_df.shape[1]), 
                                    index = results_df.index)
        
        #for each kraj allocate seats
        for j in range(results_df.shape[1]):
            aux_results = results_df.iloc[:, j].values
            aux_seats = seats_df.iloc[0, j]
            seats_final.iloc[:, j] = self.dhont_(n_seats = aux_seats, results = aux_results)
        
        self.seats_party = seats_final
        if inplace == False:
            return seats_final

class Sim():
    def __init__(self, n_seats, n_parties, threshold, alpha):
        self.n_seats = n_seats
        self.n_parties = n_parties
        self.threshold = threshold
        self.alpha = alpha
        
    def sample_votes(self):
        m = np.zeros(self.n_parties*len(self.n_seats)).reshape(self.n_parties, len(self.n_seats))
        for i in range(m.shape[1]):
            m[:, i] = stats.dirichlet.rvs(alpha = self.alpha, size = 1)
        m_sim = np.round(pd.DataFrame(m) * 10000)
        return m_sim
    
    def sim_one(self):
        #extract names of districts
        districts = list(self.n_seats.keys())
        n = len(districts)
        
        results_sim = self.sample_votes()
        # for i in range(n):
        #     aux_sim = self.sample_votes()
        #     results_sim[districts[i]] = dict(enumerate(aux_sim))
        
        self.votes = pd.DataFrame(data = results_sim)
        self.votes.columns = list(self.n_seats.keys())
        self.seats = Calc(n_seats = self.n_seats, results = results_sim).assign_seats(inplace=False, threshold=self.threshold)
    
    def sim_multi(self, N):
        S = np.zeros(N * self.n_parties).reshape(N,self.n_parties)
        V = np.zeros(N * self.n_parties).reshape(N,self.n_parties)
        for n in range(N):
            self.sim_one()
            v = self.votes.sum(axis = 1) / np.sum(self.votes.sum(axis = 1))
            s = self.seats.sum(axis = 1) / np.sum(self.seats.sum(axis = 1))
            S[n, s.index.values] = s
            V[n, v.index.values] = v
        self.sim = {"seats": S, "votes": V}