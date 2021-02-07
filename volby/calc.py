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
    
    def hb_(self, results, n_seats):
        # create a matrix for the first round where to store results
        seats_round1 = np.zeros(results.shape[0] * n_seats.shape[0]).reshape(results.shape[0], n_seats.shape[0])
        votes_left_from_r1 = np.zeros(results.shape[0] * n_seats.shape[0]).reshape(results.shape[0], n_seats.shape[0])
        seats_left_from_r1 = np.zeros(n_seats.shape[0])
        #allocate seats in the first round
        for i in range(seats_round1.shape[1]):
            v = results[:,i]
            s = n_seats[i]
            #calculate hb quota
            quota = np.sum(v) / (s+1)
            #allocate seats
            seats_round1[:, i] = np.floor(v / quota)
            #store votes left
            votes_left_from_r1[:, i] = v - (quota * seats_round1[:, i])
            #store seats left
            seats_left_from_r1[i] = s - np.sum(seats_round1[:, i])
        
        # sum all the votes
        votes_r2 = votes_left_from_r1.sum(axis=1)
        n_seats_r2 = seats_left_from_r1.sum()
        # total seats
        seats_total = seats_round1.sum(axis=1)
        #calculate quota
        quota = np.sum(votes_r2) / (n_seats_r2+1)
        # allocate seats
        seats_aux = np.floor(votes_r2 / quota)
        #update votes left
        votes_r2 = votes_r2 - (seats_aux * quota)
        #update seats
        n_seats_r2 = n_seats_r2 - seats_aux.sum()
        #add to total seats
        seats_total += seats_aux

        #if there are still seats to left allocate to the party with the highest remain
        if n_seats_r2 > 0:
            # allocate seats
            seats_aux = votes_r2 / quota
            for _ in range(int(n_seats_r2)):
                seats_aux[seats_aux.argmax()] = -99.0

            seats_aux = [1 if x==-99 else 0 for x in seats_aux]
            #add to total seats
            seats_total += seats_aux

        return seats_total.astype(int)


    def allocate_seats(self, threshold, method, inplace = True):
        seats_df = pd.DataFrame(index = self.seats.keys(), data = self.seats)
        results_df = pd.DataFrame(data = self.results)
        results_df.columns = seats_df.columns
        
        #first only use parties that have above the threshold
        total_votes = results_df.sum(axis = 1)
        prc_votes = total_votes / total_votes.sum()

        results_df = results_df.loc[prc_votes >= threshold, :]
        
        # dhont
        if method=="dhont":
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

        if method=="hb":
            aux_seats = seats_df.iloc[0,:].to_numpy()
            aux_votes = results_df.to_numpy()
            seats_final = self.hb_(results=aux_votes, n_seats=aux_seats)
            self.seats_party = pd.DataFrame(data=seats_final, index=results_df.index, columns=["Total"])

        if inplace == False:
            return self.seats_party

class Sim():
    def __init__(self, n_seats, n_parties, threshold, alpha, method):
        self.n_seats = n_seats
        self.n_parties = n_parties
        self.threshold = threshold
        self.alpha = alpha
        self.method = method
        
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
        self.seats = Calc(n_seats = self.n_seats, results = results_sim).allocate_seats(inplace=False, threshold=self.threshold, method=self.method)
    
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