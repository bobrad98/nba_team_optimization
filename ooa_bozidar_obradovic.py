import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import time
from func_timeout import func_timeout, FunctionTimedOut

plt.close('all')

####################### parameters, constants #######################
N_players = 12 #number of players on one team
N_teams = 100 #population size

#number of players per position on team
c_guards = 4
c_wings = 5
c_bigs = 3
c_pos = [c_guards, c_wings, c_bigs]
cum_pos = np.cumsum(c_pos)

sal_min = 0
sal_max = 136606000 #luxury tax threshold
cut = 10**6
low = 101173000 #90% of salary cap

#nsga
N_gen = 45
p_cross = 0.8
p_mut = 0.15

####################### class definitions ############################
@dataclass
class Player:
    ID: int
    name: str = ''
    surname: str = ''
    pos: str = ''
    ovr: int = 0
    salary: float = 0
   
@dataclass
class Team:
    members: list = np.full((N_players,), Player('default'))
    N_team: int = len(members)
    
    def overall(self):
        t_sum = 0
        for i in range(self.N_team):
            player = self.members[i]
            t_sum = t_sum + player.ovr
        t_ovr1 = t_sum / self.N_team
        for i in range(self.N_team):
            player = self.members[i]
            if player.ovr > t_ovr1:
                t_sum = t_sum + player.ovr * (player.ovr - t_ovr1) / t_ovr1
        t_ovr = t_sum / self.N_team
        return t_ovr

    def salary(self): #sum of all players's salaries
        salary = 0
        for i in range(self.N_team):
            player = self.members[i]
            salary = salary + player.salary
        return salary
    
    def ID(self): #list of all players's IDs
        ID = np.zeros(self.N_team,)
        for i in range(self.N_team):
            player = self.members[i]
            ID[i] = player.ID
        return ID
    
    def salary_list(self):
        salary_list = np.zeros(self.N_team,)
        for i in range(self.N_team):
            player = self.members[i]
            salary_list[i] = player.salary
        return salary_list

########### loading files and generating initial population ##########
def generate_team(guards, wings, bigs, cum_pos, cut):
    team = Team()
    g, w, b = guards.copy(), wings.copy(), bigs.copy()
     
    for j1 in range(cum_pos[0]): #selection of guards
        ksi = np.random.choice(len(g))
        tf = False
        while not tf:
            team.members[j1] = g[ksi]
            if sal_max - team.salary() > cut:
                tf = True
            else:
                ksi = np.random.choice(len(g))
        del g[ksi]
    
    for j2 in range(cum_pos[0], cum_pos[1]): #selection of wings
        ksi = np.random.choice(len(w))
        tf = False
        while not tf:
            team.members[j2] = w[ksi]
            if sal_max - team.salary() > cut:
                tf = True
            else:
                ksi = np.random.choice(len(w))
        del w[ksi]

    for j3 in range(cum_pos[1], cum_pos[2]): #selection of bigs
        ksi = np.random.choice(len(b))
        tf = False
        while not tf:
            team.members[j3] = b[ksi]
            if j3 == cum_pos[2]:
                cut = 0
            if sal_max - team.salary() > cut:
                tf = True
            else:
                ksi = np.random.choice(len(b))
        del b[ksi]
    
    if team.salary() > sal_min:
        check = True
    else:
         check = False
         
    while not check:
        salaries = team.salary_list()
        ind = np.argmin(salaries)
        
        if ind < cum_pos[0]: #guards
            ksi = np.random.choice(len(guards))
            team.members[ind] = guards[ksi]
        elif ind >= cum_pos[0] and ind < cum_pos[1]: #wings
            ksi = np.random.choice(len(wings))
            team.members[ind] = wings[ksi]
        else: #bigs
            ksi = np.random.choice(len(bigs))
            team.members[ind] = bigs[ksi]
        
        if team.salary() > sal_min and team.salary() < sal_max:
            check = True
            
    # print('team successfully generated')
    return team, team.members

def generate_pop(N_teams, N_players, guards, wings, bigs, cum_pos, cut):
    pop = np.full((N_teams, N_players), Player('default')) 
    for i in range(N_teams):
        _, members = generate_team(guards, wings, bigs, cum_pos, cut)
        pop[i, :] = members
    # print('population successfully generated\n')
    return pop

data = pd.read_csv('official 2k22 data.csv')
name = data["name"].astype(str).tolist()
surname = data["surname"].astype(str).tolist()
pos = data["type"].astype(str).tolist()
ovr = data["overall"].astype(int).to_numpy()
salary = data["salary"].astype(float).to_numpy()

players = []
guards = []
wings = []
bigs = []

for i in range(len(name)):
    p = Player(i, name[i], surname[i], pos[i], ovr[i], salary[i])
    players.append(p)
    if pos[i] == "guard":
        guards.append(p)
    elif pos[i] == "wing":
        wings.append(p)
    elif pos[i] == "big":
        bigs.append(p)
# N_pool = len(players)

start = time.time()
try:
    pop = func_timeout(0.1, generate_pop,args=(N_teams, N_players, guards,
                                               wings, bigs, cum_pos, cut))
    print('Initial population successfully generated')
except FunctionTimedOut:
    raise Exception('Initial population not successfully generated, run code again\n')

############################# NSGA ###############################
def pareto_sol(pop): #finding all non-dominated solutions in a population
    rows = []
    N_teams, _ = np.shape(pop)
    for i in range(N_teams):
        team1 = Team(pop[i, :])
        par = True
        c = 0
        while par and c < N_teams:
            team2 = Team(pop[c, :])
            if team2.salary() < team1.salary() and team1.overall() < team2.overall():
                par = False
            else:
                c = c + 1
        if par:
            rows.append(i)
            
    return pop[rows, :], rows

def pareto_sort(population):
    pop = population.copy()
    N_teams, N_players = np.shape(pop)
    pop_pareto = np.full((1,N_players), Player('default'))
    
    [ovr_u, ovr_l, sal_u, sal_l] = [0, 80, 0, sal_max] #upper and lower niche bounds
    for i in range(N_teams):
        team = Team(pop[i, :])
        ovr = team.overall()
        sal = team.salary()
        if ovr > ovr_u:
            ovr_u = ovr
        elif ovr < ovr_l:
            ovr_l = ovr
        if sal > sal_u:
            sal_u = sal
        elif sal < sal_l:
            sal_l = sal
            
    tt, _ = np.shape(pop)
    N_front = 0
    pareto_list = []
    while tt != 0:
        pareto, rows = pareto_sol(pop)
        pareto_list.append(np.split(pareto, len(rows), axis=0))
        pop_pareto = np.concatenate((pop_pareto, pareto))
        pop = np.delete(pop, rows, axis=0)
        tt, _ = np.shape(pop)
        N_front += 1
    pop_pareto = np.delete(pop_pareto, 0, axis=0)
    sigma = np.sqrt(np.power(ovr_u-ovr_l, 2) + np.power(sal_u-sal_l, 2))\
        / (2*np.sqrt(N_front))
    eps = 0.0001 #sigma - niche radius, eps - small positive number
 
    F_min = N_teams + eps
    loc_min = 2*N_teams
    fitness = [] 
    for i in range(N_front):           #selection of particular pareto front
        members = pareto_list[i]
        for j in range(len(members)):  #selection of particular team
            team1 = Team(members[j].flatten())
            ovr1, sal1 = team1.overall(), team1.salary()   
            F = F_min - eps
            nc = 0
            for k in range(len(members)):
                team2 = Team(members[k].flatten())
                ovr2, sal2 = team2.overall(), team2.salary()
                d = np.sqrt(np.power(ovr2 - ovr1, 2) + np.power(sal2 - sal1, 2))
                if d <= sigma:
                    nc += 1 - d/sigma
            f_j = F/nc
            fitness.append(f_j)
            if f_j < loc_min:
                loc_min = f_j
        F_min = loc_min
        
    return pop_pareto, fitness

def roulette(population, fitness):
    pop = population.copy()
    N_teams, N_players = np.shape(pop)
    new_pop = np.full((N_teams, N_players), Player('default'))
    
    roulette = np.zeros(N_teams,)
    for i in range(N_teams):
        roulette[i] = fitness[i]/np.sum(fitness) 
    roulette_cum = np.zeros(N_teams+1,)
    roulette_cum[0] = 0
    roulette_cum[1:] = np.cumsum(roulette)
     
    for i in range(N_teams):
        ksi = np.random.rand()
        for j in range(1, N_teams+1):
            if ksi > roulette_cum[j-1] and ksi < roulette_cum[j]:
                new_pop[i, :] = pop[j-1, :]
                break

    return new_pop

def crossover(population, p_cross):
    pop = population.copy()
    N_teams, N_players = np.shape(pop)
    new_pop = np.full((N_teams, N_players), Player('default'))
    
    count = 0
    while count < N_teams/2:
        ind1 = np.random.choice(N_teams)
        ind2 = np.random.choice(N_teams)
        while ind1 == ind2:
            ind2 = np.random.choice(N_teams)
        team1 = pop[ind1, :]
        team2 = pop[ind2, :]
        
        ksi = np.random.rand()
        if ksi < p_cross:
            size = 1 #number of players to be swapped
            check = False
            while not check:
                indices = np.random.choice(N_players, size) #which positions will swap
                        
                temp = team1[indices]
                team1[indices] = team2[indices]
                team2[indices] = temp
                
                t1, t2 = Team(team1), Team(team2)
                id_1, id_2 = t1.ID(), t2.ID()
                if len(np.unique(id_1)) == N_players and len(np.unique(id_2)) == N_players:
                    if t1.salary() < sal_max and t1.salary() > sal_min:
                        if t2.salary() < sal_max and t2.salary() > sal_min:
                            check = True
        else:
            t1, t2 = Team(team1), Team(team2)
            
        new_pop[2*count, :] = t1.members
        new_pop[2*count + 1, :] = t2.members   
        count += 1
    
    # print('crossover works')
    return new_pop

def mutation(population, p_mut, guards, wings, bigs):
    pop = population.copy()
    N_teams, N_players = np.shape(pop)
    new_pop = np.full((N_teams, N_players), Player('default'))
    
    for i in range(N_teams):
        team = pop[i, :]
        ksi = np.random.rand()
        if ksi < p_mut:
            size = 1
            indices = np.random.choice(N_players, size) ##which positions will swap
            
            for ind in indices:
                check = False
                while not check:
                    if ind < cum_pos[0]: #guards
                        ksi = np.random.choice(len(guards))
                        team[ind] = guards[ksi]
                    elif ind >= cum_pos[0] and ind < cum_pos[1]: #wings
                        ksi = np.random.choice(len(wings))
                        team[ind] = wings[ksi]
                    else: #bigs
                        ksi = np.random.choice(len(bigs))
                        team[ind] = bigs[ksi]     
                    t = Team(team)
                    if len(np.unique(t.ID())) == N_players:
                        if t.salary() < sal_max and t.salary() > sal_min:
                            check = True
        
            new_pop[i, :] = t.members
        else:
            new_pop[i, :] = pop[i, :]
    
    # print('mutation works\n')
    return new_pop

list_pop = []
for i in range(N_gen):
    pop_pareto, fitness = pareto_sort(pop)
    pop_roulette = roulette(pop_pareto, fitness)
    pop_crossover = crossover(pop_roulette, p_cross)
    pop_mutation = mutation(pop_crossover, p_mut, guards, wings, bigs)
    pop = pop_mutation
    list_pop.append(pop)

end = time.time()
print("Execution time: " + str(end - start) + " seconds")

######################## visualization and writing to files #########################
def write_and_visual(pop, low, high, name):
    N_teams, N_players = np.shape(pop)
    plt.figure()
    
    file1 = name + '.txt'
    f = open(file1, 'w')
    sal_pop, ovr_pop = np.zeros(N_teams,), np.zeros(N_teams) #salary, overall
    for i in range(N_teams):
        team = Team(pop[i, :])
        f.write('team ' + str(i+1) + '\n')
        sal_pop[i] = team.salary()
        ovr_pop[i] = team.overall()
        pct = np.round(100 * sal_pop[i]/high, decimals = 2)
        for j in range(N_players):
            player = team.members[j]
            data = str(player.ID) + ' ' + str(player.name) + ' ' +  str(player.surname) +\
                ' ' + str(player.ovr) + ' ' + str(player.salary/10**6) + '\n'
            f.write(data)
        f.write('---------------------\n')
        f.write('overall ' + str(ovr_pop[i]) + '\n')
        f.write('team salary ' + str(sal_pop[i]) + ' = ' + str(pct) +\
                "% of available space" + str( '\n'))
        f.write('\n')    
    f.close()
    plt.scatter(sal_pop/10**6, ovr_pop, s=10, c='b')
    
    file2 = 'pareto_' + name + '.txt'
    f = open(file2, 'w')
    _, rows = pareto_sol(pop)
    num = len(rows)
    sal_par, ovr_par = np.zeros(num,), np.zeros(num,) 
    for i in range(num):
        ind = rows[i]
        team = Team(pop[ind, :])
        f.write('team ' + str(ind+1) + '\n')
        sal_par[i] = team.salary()
        ovr_par[i] = team.overall()
        pct = np.round(100 * sal_par[i]/high, decimals = 2)
        for j in range(N_players):
            player = team.members[j]
            data = str(player.ID) + ' ' + str(player.name) + ' ' +  str(player.surname) +\
                ' ' + str(player.ovr) + ' ' + str(player.salary/10**6) + '\n'
            f.write(data)
        f.write('---------------------\n')
        f.write('overall ' + str(ovr_par[i]) + '\n')
        f.write('team salary ' + str(sal_par[i]) + ' = ' + str(pct) +\
                "% of available space" + str( '\n'))
        f.write('\n')    
    f.close()
    par_plot = plt.scatter(sal_par/10**6, ovr_par, s=10, c='r', label='Pareto front')
    
    l1, = plt.plot(low*np.ones(10,)/10**6, np.linspace(74, 90, 10), c='k',\
             linestyle='dotted', label='90% of salary cap')
    l2, = plt.plot(high*np.ones(10,)/10**6, np.linspace(74, 90, 10), c='k',\
             linestyle='dashed', label='Luxury cap threshold')
    plt.xlim(0, 140)
    plt.xlabel("Salary (millions)")
    plt.ylabel("Overall")
    plt.title('Generation ' + name)
    plt.legend(handles=[par_plot, l2, l1])
    plt.grid()
    plt.savefig(name+'.jpg', dpi=300)
    
indices = [1, int(N_gen/3), int(2*N_gen/3), N_gen]
for i in indices:
    write_and_visual(list_pop[i-1], low, sal_max, 'pop_' + str(i))
plt.show()