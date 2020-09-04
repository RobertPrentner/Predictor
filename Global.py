import numpy
import random
import itertools
from scipy import stats

#some global data structures and functions
numpy.set_printoptions(precision=3,suppress=True)
alphabet = {"0": 0, "1": 1} #"a":0,"b":1,"c":2,"d":3,"e":4} #Format: character: integer – must be invertible!
inv_alphabet = {v: k for k, v in alphabet.items()}
color_scheme = {1: "red", 2: "blue", 3: "green", 4: "cyan", 5: "magenta", 6: "yellow",
                        7: "grey"}
#
payoff={(1,1): 3, (1,-1):0, (-1,1): 5, (-1,-1):1}
#payoffs for indecisive behavior are expectations:
for d in itertools.permutations((-1,0,1),r=2):
    if d[0] == 0: payoff[d] = 0
    elif d[1] == 0: payoff[d] = 0.5*( payoff[(d[0],1)] + payoff[(d[0],-1)])
payoff[(0,0)]=0
for d in itertools.product((-1,1),repeat=2):
    payoff[(0,0)] += payoff[d]/4



def sign(i):
    if i == 0: return 0
    else: return 1 if i > 0 else -1


def hamming_distance(inp,state,size_input):
    #check whether they are of the same size:
    h = 0
    if len(inp) != len(state):
        #print("Error. Can not calculate distance between", inp,state)
        return -1
    else:
        h = 0
        for k in range(size_input):
            if inp[k] != state[k]: h += 1
    return h

def delta(x,y):
    if x == y: return 1
    else: return 0

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def random_string(length):
    message = ""
    seq = numpy.random.randint(0,len(alphabet),length) #random sequence of integers in range of alphabet
    #now replace integers in with characters to construct output, according to inverted alphabet
    for i in range(len(seq)):
        message += inv_alphabet[seq[i]]
    return message


#paste message to a certain size, either by by randomly slicing or adding random bits
def paste_message(m0,to_size):
        if len(m0) == to_size: m = m0
        elif len(m0) > to_size:
            #determine which to_size-long part of the message to take
            index= numpy.random.randint(0,len(m0)-to_size)
            m = m0[index:index+to_size]
        else: m=m0 + random_string(to_size-len(m0)) #add random sequenze of characters of size (to_size - len(m0) to message
        return m

def normalize(l0): #RETURNS a nornalized list
    norm = 0.
    for i in range(len(l0)):
        norm += l0[i]
    normalized_list = l0.copy()
    if norm != 1. and norm != 0.:
        for i in range(len(l0)): normalized_list[i] = normalized_list[i]/norm
    return normalized_list

def normalize_dic(d0,p,n_rows): #normalize dictionary; p = 0 for outflowing and p = 1 for inflowing probabilities
    norm = {}
    for j in range(n_rows): norm[j] = 0.
    for key in d0:
        j = key[p]  # sum of outflowing probability = 1
        norm[j] += d0[key]  # counts outgoing connectivity for all agents
    for key in d0:
        j = key[p]
        d0[key] = numpy.around(d0[key] / norm[j], 2)

#convert integer to list of binaries of size s
def int_to_bin(integer,s):
    bin_str = bin(integer)
    bin_rep = list(bin_str[2:])
    bin_rep = [0]*(s-len(bin_rep)) + list(map(int,bin_rep))
    return bin_rep

#onvert tuple with 0 and 1 to binary string
def tuple_to_bin(key):
    binkey = ""
    for i in range(len(key)): binkey += str(key[i])
    return binkey

#convert bit representation into color representation
def convert_rep(bit,color):
    rn=int(bin(bit),2)

#calculate mutual information induced by a Markovian kernel on an input distribution p_inp
#(Mutual information is equivalent to maximum channel capacity)
# p_k(x) = sum_i (p_ki(x|inp)*p_i(inp)) for all p_i(inp) that contribute to p_k(x)
# p_ik(mutual) = p_ik(output|input)p_i(input) <–––> p(x,inp) = p(x|inp)p(inp)
# MI(inp,x) = sum_{i,k} p_ik(mututal)* [ LOG2(p_ik(mututal)) - LOG2(p_i(inp)p_k(x))]
#           = sum_{i,k} p_ik(mututal)* LOG2[p_ki(x|inp)/p_k(x)]
def kernel_MI(p_inp,kernel,output_size):
    k_MI = 0
    p_out = [0] * (output_size)
    # p_k(x) = sum_i (p_ki(x|inp)*p_i(inp)) for all p_i(inp) that contribute to p_k(x)
    for key in kernel.keys():
        for k in range(output_size):
            if kernel[key][k] > 0:
                #key to integer
                if type(key) == str:
                    i = int(key,2)
                elif type(key) == tuple:
                    i = int(tuple_to_bin(key),2)
                p_out[k] += kernel[key][k] * p_inp[i]
    # MI(inp,x) = sum_{i,k} p_ik(mututal)* LOG2[p_ki(x|inp)/p_k(x)]
    for key in kernel.keys():
        for k in range(output_size):
            if kernel[key][k] > 0 and p_out[k] > 0:
                # key to integer
                if type(key) == str:
                    i = int(key, 2)
                elif type(key) == tuple:
                    i = int(tuple_to_bin(key), 2)
                k_MI += kernel[key][k] * p_inp[i] * \
                                numpy.log2(kernel[key][k] / p_out[k])
    return (k_MI,p_out)

#calculate mututal information from dictionary of the form {(i,j): N(i,j)}
def hist_MI(hist,norm=True):
    hist_MI = 0
    p_inp={}
    p_out={}
    for key in hist:
        p_inp[key[0]] = 0.
        p_out[key[1]] = 0.

    #normalize dictionary first
    if norm:
        normalize =  sum(hist.values())
        for key in hist: hist[key] = hist[key]/normalize
    #caluclate marginal probabilities
    for key in hist:
            p_inp[key[0]] += hist[key]
            p_out[key[1]] += hist[key]
    #calculate MI
    for key in hist:
        hist_MI += hist[key]*numpy.log2(hist[key]/(p_inp[key[0]]*p_out[key[1]]))
    return hist_MI

def entropy(p):
    e = 0.
    p = normalize(p)
    for i in range(len(p)):
        if p[i] > 0.: e -= p[i]*numpy.log2(p[i])
    return e

def dictionary_to_matrix(dict):
    j = 0
    xdim = len(dict)
    ydim = len(list( dict.values())[0] ) #assuming all values in dictionary have the same length
    m = [[0]*ydim]*xdim
    for key in dict:
        m[j] = dict[key]
        j += 1
    return m

def random_choose_dict(population,k):
    out_list = []
    popkeys = list(population.keys())
    numpy.random.shuffle(popkeys)
    #print(popkeys)
    #assume weights are normalized
    for j in range(k):
        r = numpy.random.random()
        for key in popkeys:
            #print(r,key)
            if r < population[key]:
                #print(r,population[key])
                out_list.append(key)
                break
            else:
                r = r - population[key]
    #print(r, population[key],out_list)
    return out_list

def signal_distribution(type, gbits=2, max_iter = 16):

    signal_population = numpy.random.randint(0, 1 + 1, size=gbits).tolist()
    # weight according to "type" distribution

    n = 2 ** gbits
    if type == "norm":
        distribution = getattr(stats, type)
        x = numpy.arange(n)
        xU, xL = x + 0.5, x - 0.5
        around = n / 2
        std_dev = 1
        prob = distribution.cdf(xU, loc=around, scale=std_dev) - distribution.cdf(xL, loc=around, scale=std_dev)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        signal_number = numpy.random.choice(x, p=prob, size=max_iter)

    else: #uniform distribution
        signal_number = numpy.random.randint(0, n, size=max_iter).tolist()

    return signal_number

def variance(vector):
    n = len(vector)
    if n == 1: return (sum(vector),0)
    ave = sum(vector) / n
    variance = 0.
    for i in range(n):
        variance += (ave - vector[i]) ** 2 / (n - 1)
    return(ave,variance)


def ideal_kernel(size_input,noise):
    #calculate an ideal kernel under condition that input is noisy (p for bit_flip = alpha)
    #assumption: 1-1 mapping from actual inputs to output -> p(o|i) = p(o|i_a)p(i_a|i)
    #noise between input i and actual input i_a -> p(i_a|i) = noise^H(i,i_a)*(1-noise)^(|I|-H(i,i_a)
    #prior probability p(i) = const

    ideal_kernel = {}
    for i in itertools.product({0,1},repeat=size_input):
        ideal_kernel[i] = []
        for ia in itertools.product({0,1}, repeat=size_input):
            distance = hamming_distance(i,ia,size_input)
            p = (1/2**size_input)* (noise ** distance ) * (1.-noise)**(size_input-distance)
            ideal_kernel[i].append(p)
        ideal_kernel[i] = normalize(ideal_kernel[i])
    return ideal_kernel

def dict_from_adj(a,s=0): #get a (sparse?) adjacency matrix and returns dictionary of the form source->target
    adj = numpy.nonzero(a)
    adj_dict = dict.fromkeys(list(adj)[1])
    for input in range(s):
        adj_dict[input]=[()]
    for tup in list(numpy.transpose(adj)):
        if adj_dict[tup[1]] == None:
            adj_dict[tup[1]] = [(tup[0],numpy.sign(a[tup[0],tup[1]]))]
        else:
            adj_dict[tup[1]].append( (tup[0], numpy.sign(a[tup[0],tup[1]])) )
    return adj_dict

def agent_list_from_adj(adj,offset=0):
    adj1 = adj.sum(axis=1)  # sum of rows
    adj0 = adj.sum(axis=0)  # sum of columns
    adjs = adj1 + adj0
    indices = [ia for ia in range(offset)] + [ia for ia in range(offset,len(adj)) if adjs[ia] != 0]
    return indices