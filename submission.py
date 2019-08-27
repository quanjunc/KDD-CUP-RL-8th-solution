
import random
import numpy as np
from netsapi.challenge import *

class Particle:
    def __init__(self,x0):
        global alpha
        alpha = 0.6
        self.episode = 0
        self.position_i=[]
        self.pos_best_i=[]
        self.err_best_i=-1
        self.err_i=-1

        for i in range(0,num_dimensions):
            self.position_i.append(x0[i])

    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    def update_position(self,pos_best_g,bounds):
        self.episode += 1
        for i in range(0,num_dimensions):
            self.position_i[i]=pos_best_g[i]+random.uniform(-0.5,0.5)*(alpha**self.episode)

            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]+random.uniform(-0.5,0)*(alpha**self.episode)

            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]+random.uniform(0,0.5)*(alpha**self.episode)

class HRPO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        self.err_best_g=-1
        self.pos_best_g=[]
        self.score = []
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        i=0
        while i < maxiter:
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)
                if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(swarm[j].position_i)
                    self.err_best_g=float(swarm[j].err_i)

            for j in range(0,num_particles):
                swarm[j].update_position(self.pos_best_g,bounds)
            self.score.append(-self.err_best_g)
            i+=1

class CustomAgent:
    def __init__(self, environment):
        self.environment = environment

    def min2max(self,x):
        policy = {}
        for j in range(5):
            policy[str(j+1)]=list(x[j*2:j*2+2])
        rd=-1*self.environment.evaluatePolicy(policy)
        return (rd)

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        bounds = [(0,1) for i in range(10)]
        rwd = []
        p0 = [random.choice([0,1]) for i in range(10)]
        initial_particles = [p0]
        while(len(initial_particles)<4):
            p1 = [random.choice([0,1]) for i in range(10)]
            flag = 0
            for tmp_p in initial_particles:
                if(np.sum(np.abs(np.array(tmp_p)-np.array(p1)))<6):
                    flag+=1
            if(flag==0):
                initial_particles.append(p1)
        for p in initial_particles:
            policy = {}
            for j in range(5):
                policy[str(j+1)]=list(p[j*2:j*2+2])
            rwd.append(self.environment.evaluatePolicy(policy))

        initial = initial_particles[rwd.index(max(rwd))]
        rst = HRPO(self.min2max,initial,bounds,num_particles=4,maxiter=4)

        best_policy = rst.pos_best_g
        best_reward = -rst.err_best_g

        return best_policy,best_reward

if __name__ == "__main__":
    EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, "example.csv")
