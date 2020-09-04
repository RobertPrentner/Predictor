import time
import numpy
import itertools
from fractions import Fraction
import random
import copy
import os,sys

### This is an implementation of a predictive model environment to play the iterated prisoner's dilemma.
### Opponents are represented as probabilistic 4-vector of conditional probabilities to cooperate (C)
### after previous cooperation or defection (C or D); [ p(C|CC),p(C|CD),p(C|DC),p(C|DD)] with p(D|AB) = 1 - p(C|AB)
### Decisions in the present depend on joint actions in the past
### Parameter choice follows the initial setting of Axelrod. J. Conf. Res. 24(1), 1980, with 200 turns per match
### and competing strategies have been selected in accordance with the tournament of Stewart & Plotkin, PNAS 109 (26), 2012.
### Strategies were tested against the python library axelrod (https://axelrod.readthedocs.io/)

random_model = {}
for tup in itertools.product((1,-1),repeat=2): random_model[tup] = 0.5

payoff={(1,1): 3, (1,-1):0, (-1,1): 5, (-1,-1):1}

class History:

    def __init__(self,discount=1.,depth=2,thresh = 0.01):
        self.discount = discount
        self.depth = depth
        self.thresh = thresh
        self.h=numpy.zeros((4**depth,2*(depth+1)),dtype=int)
        self.payoffs = numpy.zeros(4**depth,dtype=float)

    def return_entry(self,row,col):
        d = {0: [1, 1], 1: [1, -1], 2: [-1, 1], 3: [-1, -1]}
        length = 4**(self.depth-col)
        d_key = int(row/length)%4
        e = d[d_key]
        return e

    def print_history(self):
        for j in range(len(self.h)): print(self.h[j], numpy.around(self.payoffs[j],2))

    def build_history(self,i,model):

        #predict possible histories
        self.h[:,0] = i[0]
        self.h[:,1] = i[1]
        for j in range(4**self.depth):
            col = 1
            while col <= self.depth:
                entry = self.return_entry(j,col)
                self.h[j,2*col] = entry[0]
                self.h[j,2*col+1] = entry[1]
                col += 1

        #calculate payoffs and probabilities
        remove_lines=[]
        for j in range(4**self.depth):
            col = 1
            self.payoffs[j] = payoff[tuple(self.h[j,0:2])]
            while col <= self.depth:
                alpha = self.discount ** col
                source = tuple(numpy.flip(self.h[j, 2*col-2:2*col]))
                state = tuple(self.h[j,2*col:2*col+2])
                pC = model[source]
                p = pC if state[1] == 1 else (1-pC)
                #print(j,col,":", source, "->", state, p)
                po = payoff[state]
                if p <= self.thresh:
                    remove_lines.append(j)
                    self.payoffs[j] = 0
                    break
                else: self.payoffs[j] +=  p * po * alpha
                col += 1

        #print(remove_lines)
        #self.h = numpy.delete(self.h,(0),axis=0)
        #for line in self.h: print(line)
        self.h = numpy.delete(self.h, remove_lines,axis=0)
        self.payoffs = numpy.delete(self.payoffs,remove_lines)
        #self.print_history()

    def find_max_history(self):
        index = numpy.argmax(self.payoffs)
        h = self.h[index]
        po = self.payoffs[index]
        return h,po

class Predictor:

    def __init__(self,model,explore,thresh,depth,discount,payoff):
        self.type = "PREDICTOR"
        self.model = model #model of the oponent's strategy
        self.model0 =  model #the initial opponent model
        self.explore = explore #how many learning steps (exploitation vs exploration)
        self.discount = discount  # discount for the future
        self.thresh = thresh #threshold value for events in history
        self.depth = depth  # the depth to look for
        self.payoff = payoff #the payoff model
        self.decision = random.choice((-1,1)) #initial decision
        self.predicted_history = History(self.discount,self.depth,self.thresh)
        self.max_history = ([1]*2*(self.depth+1),self.payoff[(1,1)])

    def calculate_p1(self,initial,print_flag=False):
        #P1(C) = PO(CC)p(C|initial) + PO(CD)*p(D|initial)
        #P1(D) = PO(DC)p(C|initial) + PO(DD)*p(D|initial)
        #print(self.model)
        #NOTE: Model keys are in reverse order (!)
        p0 = self.model[initial[1],initial[0]]
        P1C = payoff[(1,1)]*p0 + payoff[(1,-1)]*(1-p0)
        P1D = payoff[(-1,1)]*p0 + payoff[(-1,-1)]*(1-p0)
        if print_flag:
            print("probability of C:",p0)
            print("payoff for choosing C at first turn:", P1C)
            print("payoff for choosing D at first turn:", P1D)
        return P1C,P1D

    def calculate_p2(self,initial,print_flag=False):
        #P2(C;D) = PO(DC)[ p(C|CC) * p(C|initial) + p(C|CD)* p(D|initial)]
        #        + PO(DD)[ p(D|CC) * p(C|initial) + p(D|CD)* p(D|initial)]
        #P2(D;D) = PO(DC)[ p(C|DC) * p(C|initial) + p(C|DD)* p(D|initial)]
        #        + PO(DD)[ p(D|DC) * p(C|initial) + p(D|DD)* p(D|initial)]
        pC0 = self.model[initial[1],initial[0]]
        pD0 = 1-pC0
        pCCC = self.model[1,1]
        pCCD = self.model[-1,1]
        pCDC = self.model[1,-1]
        pCDD = self.model[-1,-1]
        pDCC = 1-pCCC
        pDCD = 1-pCCD
        pDDC = 1-pCDC
        pDDD = 1-pCDD

        P2CD = payoff[(-1,1)] * (pCCC*pC0 + pCCD*pD0) + payoff[(-1,-1)] * (pDCC*pC0 + pDCD*pD0)
        P2DD = payoff[(-1,1)] * (pCDC * pC0 + pCDD * pD0) + payoff[(-1, -1)] * (pDDC * pC0 + pDDD * pD0)
        if print_flag:
            print("payoff for choosing C, then D:", P2CD)
            print("payoff for choosing D, then D:", P2DD)
        return P2CD,P2DD

    def calculate_history_alt(self, i,print_flag=False):
        # drop last histories and replace with newest
        return 0
        # if print_flag: print(self.input)


    def decide_alt(self,initial,print_flag=True):
        #<PO>_1 = <PO>(C) + <PO>(CD)
        #<PO>_2 = <PO>(D) + <PO>(DD)
        poC,poD = self.calculate_p1(initial, print_flag=print_flag)
        poCD,poDD = self.calculate_p2(initial, print_flag=print_flag)
        if (poC + poCD) > (poD + poDD): self.decision = 1
        elif (poD + poDD) >= (poC + poCD): self.decision = -1
        else:
            if print_flag: print("Take random desicion")
            self.decision = random.choice((-1, 1))
        if print_flag: print("Decision:", self.decision)

    def calculate_history(self,initial,print_flag=False):
        #re-initzialize predicted histories
        self.predicted_history.__init__(self.discount,self.depth,self.thresh)
        #self.max_history.__init__(initial,0)
        # calculate possible histories
        self.predicted_history.build_history(initial,self.model)
        if print_flag: self.predicted_history.print_history()
        ####

    def decide(self,print_flag=False):
        try:
            h,po = self.predicted_history.find_max_history()
            self.max_history=(h,po)
            self.decision = h[2]
            if print_flag:  print("Best history:", self.max_history,"-> decision:", self.decision)
        except:
            print("Warning: Not predicted the future long enough")
            self.decision = random.choice((-1,1))
        ###

    def update_model(self,data,print_flag=False):
        if print_flag: print(" Update model")
        if data[1] in self.model:
            p_old = Fraction(self.model[data[1]]).limit_denominator(100)
            #if print_flag: print(" ",data[1],":",p_old.numerator/p_old.denominator)
            if data[0] == 1: p_new = (p_old.numerator + 1)/(p_old.denominator+1)
            elif data[0] == -1: p_new = p_old.numerator/(p_old.denominator+1)
                #print(" ",data[1],":",p_new)
            self.model[data[1]] = p_new
            if print_flag: print(" ->",data[1],":",numpy.around(p_new,3))
        else: print(" Warning: Data nor valid. Cannot update model")

class FixedStrategy:

    def __init__(self,strategy,name="strategy",memory_length=1):
        self.strategy = strategy
        self.memory_length = memory_length
        self.type = name
        self.explore = 0.
        self.input = [(1,1)]*self.memory_length #default input
        if 0 not in strategy: self.strategy[0] = random.choice((-1, 1))
        #print(self.type,strategy[0])
        self.decision = strategy[0] #initial decision

    def decide(self,print_flag=False):
        decision_input = [ item for sublist in self.input for item in sublist]
        pC = self.strategy[tuple(decision_input)]
        self.decision = random.choices((1,-1),weights=[pC,1-pC])[0]
        if print_flag: print(" Strategy decision:", self.decision)

    def calculate_history(self,i,print_flag=False):
        #drop last histories and replace with newest
        self.input.pop(self.memory_length-1)
        self.input.insert(0,tuple(i))
        #if print_flag: print(self.input)

class Tournament:

    def __init__(self,nturn,iter,discount):
        self.nturn = nturn
        self.iter = iter
        self.discount = discount
        self.comparison = numpy.zeros((1,10),dtype=float)
        self.predictor_score = [0]*self.nturn

    def round(self,player,opponent,initial,print_flag=False,alt_flag=False): #One round of the game with nturn turns

        if not initial: i = random.choices((-1, 1),k=2)
        else: i = initial
        game_history = numpy.array([0] + i)
        p0 = [payoff[tuple(i)], payoff[tuple(reversed(i))]]
        ave_p = numpy.array(p0, dtype=float) / nturn
        if print_flag:
            print("---", 0, "---")
            print("Initial decision:", game_history[1:],"leads to payoff:",ave_p)

        for nt in range(self.nturn):
            # infer decision:
            if print_flag: print("---", nt + 1, "---")

            if alt_flag and "PREDICTOR" in player.type:
                player.calculate_history_alt(i,print_flag=False)
                player.decide_alt(i,print_flag=print_flag)
            else:
                player.calculate_history(i, print_flag=print_flag)
                player.decide(print_flag=print_flag)

            #seed1 = numpy.random.rand()
            #if seed1 < player.explore: d1 = random.choice((1,-1))
            #else: d1 = player.decision
            d1 = player.decision if nt >= int(self.nturn*player.explore) else random.choice((1,-1))
            if alt_flag and "PREDICTOR" in opponent.type: opponent.decide_alt(i,print_flag=print_flag)
            else:
                opponent.calculate_history(list(reversed(i)), print_flag=print_flag)
                opponent.decide(print_flag=print_flag)

            #seed2 = numpy.random.rand()
            #if seed2 < opponent.explore: d2 = random.choice((1, -1))
            #else: d2 = opponent.decision
            d2 = opponent.decision if nt >= int(self.nturn*opponent.explore) else random.choice((1,-1))

            d = [d1,d2]

            if "PREDICTOR" in player.type:
                data = (d2, tuple(reversed(i)))
                player.update_model(data, print_flag)

            if "PREDICTOR" in opponent.type:
                data2 = (d1, tuple(i))
                opponent.update_model(data2,print_flag)

            game_history = numpy.vstack([game_history, numpy.array([nt + 1.] + d)])
            if nt != self.nturn - 1: # disregard last match
                alpha = (self.discount**(nt+1))/self.nturn
                ave_p += [ payoff[tuple(d)]*alpha, payoff[tuple(reversed(d))]*alpha]
                if "PREDICTOR" in player.type: self.predictor_score[nt] += payoff[tuple(d)]/self.nturn/self.iter
                if print_flag:
                    print("Decision:", d, "leads to payoff:", numpy.around(ave_p, 2), "after previous action:", i)

            elif nt == self.nturn-1 and print_flag:
                try:    print(" updated model after",nt,"steps:",{ (key,numpy.around(player.model[key],2)) for key in player.model})
                except: None
                try:    print(" updated model after",nt,"steps:", { (key,numpy.around(opponent.model[key], 2)) for key in opponent.model},)
                except: None

            i = d

        """ not necessary if only copy of objets are given to tournament.round
        #after game: reset player's and opponent's model
        if "PREDICTOR" in player.type:
            player.model = copy.deepcopy(player.model0)
            player.decision = 1#random.choice((-1,1))
        else: player.__init__(player.strategy,player.type,player.memory_length)

        if "PREDICTOR" in opponent.type:
            opponent.model = copy.deepcopy(opponent.model0)
            opponent.decision = 1#random.choice((-1, 1))
        else: opponent.__init__(opponent.strategy, opponent.type, opponent.memory_length)
        """

        return ave_p,game_history

    def match(self,player,opponent,initial=[],print_flag=False,alt_flag=False): #One match for iter iterations
        ave_p = numpy.zeros((self.iter, 2))
        #ave_game_history = numpy.zeros((nturn + 1, 3))
        # play against one particular strategy
        for j in range(self.iter):
            #try:
            i1 = [player.decision] if "PREDICTOR" not in player.type else random.choices((-1, 1), k=1)
            i2 = [player.decision] if "PREDICTOR" not in opponent.type else random.choices((-1, 1), k=1)
            if initial == []: i = i1 + i2
            else: i = initial
            #if j == 0: i = random.choices((-1, 1), k=2) #randomiz initial actions for ALL agents
            tup = tournament.round(copy.deepcopy(player), copy.deepcopy(opponent), i, print_flag,alt_flag)
            #print(ave_p[j], tup[0])
            ave_p[j] = tup[0]
            if print_flag:
                print(" Match history:")
                if self.iter <= 5 and self.nturn >= 15:
                    for line in numpy.transpose(tup[1])[1:]: print("  ", line[self.nturn-15:self.nturn])
                elif self.iter <= 5 and self.nturn < 15:
                    for line in numpy.transpose(tup[1])[1:]: print("  ", line)

                if j== self.iter-1:
                    print(" with payoff:", numpy.around(numpy.mean(ave_p,axis=0),3),numpy.around(numpy.std(ave_p,axis=0),2))
            #ave_game_history += tup[1] / iter
        return numpy.append(numpy.mean(ave_p,axis=0),numpy.std(ave_p,axis=0))

    def round_robin(self,players,print_flag=False,alt_flag=False):
        nplayers = len(players)
        self.payoffs = numpy.zeros((nplayers, nplayers), dtype=float)
        self.std = numpy.zeros((nplayers, nplayers), dtype=float)
        #for (i,j) in random_combination(range(nplayers),2): print(i,j)
        for (i,j) in itertools.combinations_with_replacement( range(nplayers),2):
            result = self.match(players[i],players[j],[],print_flag,alt_flag)
            print(players[i].type, "vs.",players[j].type, "payoffs:", numpy.around(result,3))
            if i == j:
                self.payoffs[i,j] = 0.5*(result[0]+result[1])
                self.std[i,j] = 0.5*(result[2]+result[3])
            else:
                self.payoffs[i,j] = result[0]
                self.payoffs[j,i] = result[1]
                self.std[i,j] = result[2]
                self.std[j,i] = result[3]

    def generate_results(self,players,print_flag=False):
        if print_flag: print("-- results of the round robin ---")
        self.results={}
        for i in range(len(players)):
            line = self.payoffs[i,:]
            std = self.std[i,:]
            column = self.payoffs[:,i]
            diff = numpy.around(line - column,2)
            number_of_wins = numpy.maximum(0,numpy.sign(diff))
            self.results[players[i].type] = [numpy.around(numpy.sum(line) / len(players), 3),                           #average payoff
                                             numpy.around(numpy.sum(std) / len(players) / numpy.sqrt(self.iter), 3),    #average standard error of the mean
                                             number_of_wins] #average number of winnings

            if "PREDICTOR" in players[i].type: self.comparison = self.payoffs[i]
            if print_flag: print(line, " |", self.results[players[i].type][0])

        occ = 1
        for key, value in sorted(self.results.items(), key=lambda item: item[1][0],reverse=True):
            self.results[key].append(occ)
            occ += 1
            if print_flag:
                print("%s: %s %s %s: %s" % (key, value[0],[value[1]],"number of wins",value[2]))
        if print_flag:
            print("std error of mean payoff:")
            for line in self.std: print(numpy.around(line,3))

def output_explore(players,plot,compare,places):

    print("explore/payoff/SEM/D(ZD-GTFT-2)/SEM(ZD-GTFT-2)/place/wins")
    f = open("../../manuscripts/IPD_Predictor/results0.txt", "w+")
    f2 = open("../../manuscripts/IPD_Predictor/results1.txt", "w+")
    f3 = open("../../manuscripts/IPD_Predictor/results2.txt", "w+")
    f.write("explore/payoff/SEM/D(ZD-GTFT-2)/SEM(ZD-GTFT-2)/place/wins" + "\n")
    for line in plot:
        print(line)
        for a in line: f.write(str(numpy.around(a, 3)) + ", ")
        f.write("\n")
    print(" " * 3, end=" ")

    f2.write("explore, ")
    for p in players:
        print(p.type, end=", ")
        f2.write(p.type + ", ")
    print("")
    f2.write("\n")
    for line in compare:
        print(line)
        for a in line: f2.write(str(numpy.around(a, 3)) + ", ")
        f2.write("\n")
    print("---")

    for p in players:
        print(p.type, end=", ")
        f3.write(p.type + ", ")
    print("")
    f3.write("\n")
    for line in places:
        print(line)
        for a in line: f3.write(str(numpy.around(a, 1)) + ", " )
        f3.write("\n")
    print("---")

    return

###MAIN; test a predicitve model for the IPD #####

start_time = time.time()
random.seed(1902) #set randoom seed
numpy.random.seed(1902)
#paramteres of the game
nturn = 200
iter = 5
discount = 1

#hyper-parameters of agent
thresh = 0.01
depth = 2#should be even and less than nturn
#nhis = 4**(depth+1)#make nhis large enough to record them all
#searches = 4**(depth+1) #search often enough
explore = 0.1 #fraction of exploration steps

##define fixed strategies
strategy = {(1,1): 1., (1,-1): 0., (-1,1): 1., (-1,-1): 0., 0: 1} #TFT
TFT = FixedStrategy(strategy,"TFT",memory_length=1)

strategy = {(1,1): 1., (1,-1): 0., (-1,1): 0., (-1,-1): 1., 0: 1} #WSLS (aka Pavlov)
WSLS =  FixedStrategy(strategy,"WSLS",memory_length=1)

strategy = {(1,1): 1., (1,-1): 1/3, (-1,1): 1., (-1,-1): 1/3, 0: 1} #Generous TFT
GTFT =  FixedStrategy(strategy,"GTFT",memory_length=1)

strategy = {(1,1): 0., (1,-1): 0., (-1,1): 0., (-1,-1): 0., 0: -1} #Defector
ALLD = FixedStrategy(strategy,"ALLD",memory_length=1)

strategy = {(1,1): 1., (1,-1): 1., (-1,1): 1., (-1,-1): 1., 0: 1} #Coperator
ALLC = FixedStrategy(strategy,"ALLC",memory_length=1)

strategy = {(1,1): 0., (1,-1): 0., (-1,1): 1., (-1,-1): 1., 0: 1} #Alternator
ALT = FixedStrategy(strategy,"ALT",memory_length=1)

strategy = {(1,1): 9/10, (1,-1): 0., (-1,1): 9/10, (-1,-1): 0., 0: 1} #JOSS
JOSS = FixedStrategy(strategy,"JOSS",memory_length=1)

strategy = {(1,1): 0., (1,-1): 0., (-1,1): 1., (-1,-1): 1.} #Trump (aka Bully)
TRUMP = FixedStrategy(strategy,"TRUMP",memory_length=1)

strategy = {(1,1): 1., (1,-1): 1/8, (-1,1): 1., (-1,-1): 1/4 } #ZD-GTFT-2
ZDGTFT =  FixedStrategy(strategy,"ZD-GTFT-2",memory_length=1)

strategy = {(1,1): 8/9, (1,-1): 1/2, (-1,1): 1/3, (-1,-1): 0.} #ZD-Extort-2
ZDExtort =  FixedStrategy(strategy,"ZD-Extort-2",memory_length=1)

for d in itertools.product((-1,1),repeat=4):
    if (d[1] == -1 and d[3] == -1): strategy[d] = 0.
    else: strategy[d] = 1.
strategy[0] = 1
strategy[1] = 1
TF2T = FixedStrategy(strategy,"TF2T",memory_length=2)
RANDOM =  FixedStrategy(copy.deepcopy(random_model),"RANDOM",memory_length=1)

agent = Predictor(random_model,explore,thresh,depth,discount,payoff)
players = [TFT,GTFT,WSLS,ZDGTFT,ZDExtort,JOSS,ALLD,ALLC,RANDOM] #ALT,TF2T,ALLC,ALLD,RANDOM
random.shuffle(players)
players = [agent] + players
tournament = Tournament(nturn,iter,discount)


#tournament.iter = 1
#tournament.nturn = 200
#initial = [1,-1]#random.choices((-1, 1),k=2)
#agent.model = {(1,1): 4/5, (1,-1): 1/2, (-1,1): 1/2, (-1,-1): 1/2}
#for key in agent.model: print(key,agent.model[key])
#agent.model = random_model
#agent.calculate_p1(initial,print_flag=True)
#agent.calculate_p2(initial,print_flag=True)
#print("initial value:", initial)
#agent.decide_alt(initial,print_flag=True)
#sys.exit()

#agent.explore=0.
#tournament.iter = 1
#tournament.nturn = 200
#tournament.match(agent,RANDOM,print_flag=True,alt_flag=True)
#tournament.match(agent,TFT,initial=[-1,-1],print_flag=True,alt_flag=True)
#tournament.match(agent,TF2T,print_flag=True,alt_flag=True)
#sys.exit()

#tournament.iter = 10
#tournament.nturn = 500
#agent.explore = 0.85
#tournament.round_robin(players, False)
#tournament.generate_results(players, print_flag=True)



frac = 1
plot = numpy.zeros((frac,7),dtype=float)
compare = numpy.zeros((frac,len(players)+1),dtype =float)
places = numpy.zeros((frac,len(players)),dtype =float)
for i in range(frac):
    if frac > 1:
        explore = i*1./(frac-1)
        agent.explore = explore
    #if explore > 0.1: break
    tournament.round_robin(players,print_flag=False,alt_flag=True)
    tournament.generate_results(players,print_flag=True)

    plot[i] = [ numpy.around(explore,2),
                tournament.results["PREDICTOR"][0],
                tournament.results["PREDICTOR"][1],
                tournament.results["PREDICTOR"][0]-tournament.results["ZD-GTFT-2"][0],
                tournament.results["ZD-GTFT-2"][1],
                tournament.results["PREDICTOR"][3],
                numpy.sum(tournament.results["PREDICTOR"][2])]

    compare[i] = numpy.append(numpy.around(explore,2),numpy.around(tournament.comparison,2))
    places[i,:] = numpy.array([tournament.results[p.type][3] for p in players])


#output exploration vs results
output_explore(players,plot,compare,places)

#ave_p = tournament.match(agent,TFT,print_flag=False)
print("PREDICTORs payoff after", nturn, "rounds and", iter, "iterations:", numpy.sum(tournament.predictor_score))
for a in numpy.around(tournament.predictor_score,3):
    print(str(a), end = ",")


#print("Learned model:", agent.model)#{(key, numpy.around(agent.model[key], 2)) for key in agent.model})
print("Running time: %s seconds" % numpy.around(time.time() - start_time,2))
