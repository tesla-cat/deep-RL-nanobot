# author Ding Ruiqi, version 3, 9 Jul 2019

import numpy as np

class environment:
    def __init__(self,transM,xmax=4,ymax=4,win_reward=100,lose_reward=-1):
        self.observation_space_n = xmax * ymax
        self.xmax, self.ymax, self.goal = xmax, ymax, xmax*ymax-1
        self.win_reward,self.lose_reward = win_reward,lose_reward
        self.actions = {"stay": np.array([0,0]),
                        "right":np.array([1,0]),
                        "left": np.array([-1,0]),
                        "up":   np.array([0,1]),
                        "down": np.array([0,-1])}
        self.transM = self.rand_transM() if transM=="random" else transM
        self.reset()

    def reset(self):
        self.state = np.array([0])
        self.coord = self.index2coord(self.state)
        self.states = {}
        self.step_cnt = 0
        self.done = False
        return self.state

    def rand_transM(self):
        states = self.xmax*self.ymax
        transM = np.random.rand(states,states)
        for s in range(states):
            coord = self.index2coord(np.array([s]))
            possible_new_coords = []
            for key in self.actions:
                new_coord = coord + self.actions[key]
                if (new_coord[0] in range(0,self.xmax) \
                    and new_coord[1] in range(0,self.ymax)):
                    possible_new_coords.append(new_coord)
            possible_new_states=[self.coord2index(c) for c in possible_new_coords]
            for new_s in range(states):
                if new_s not in possible_new_states:
                    transM[s,new_s] = 0
        row_sums = transM.sum(axis=1)
        new_matrix = transM / row_sums[:, np.newaxis] # broadcasting
        # reality check:
        for s in range(states):
            print("state ",s,":")
            print(new_matrix[s].reshape((self.xmax,self.ymax)),"\n")
        return new_matrix

    def step(self,action):
        new_coord = self.coord + self.actions[action]
        if not (new_coord[0] in range(0,self.xmax) \
            and new_coord[1] in range(0,self.ymax)):
            #print("-- You hit the wall! the action is ignored!")
            new_coord = self.coord
            action+="(ignored)"
        new_state = self.coord2index(new_coord)
        probs = self.transM[new_state,:]

        self.state = new_state = np.random.choice(len(probs),1, p=probs)
        self.coord = new_coord2 = self.index2coord(new_state)
        self.step_cnt += 1

        record = {"state":self.state,"coord1":new_coord,
                  "coord2":new_coord2,"action":action}
        self.states["step"+str(self.step_cnt)]=record

        if new_state == self.goal:
            reward = self.win_reward
            self.done = True
            return self.state,reward,self.done
        else:
            reward = self.lose_reward
            return self.state,reward,self.done

    def index2coord(self,index):
        x, y = index//self.ymax, index%self.ymax
        return np.concatenate((x, y), axis=0)
    def coord2index(self,coord):
        return coord[0]*self.ymax + coord[1]

# usage
if __name__== '__main__':
    np.random.seed(0)
    env = environment(transM = "random");
    while True:
        print(env.step("up"))
        if env.done:
            break
        print(env.step("right"))
        if env.done:
            break
    for k in env.states:
        print(k,env.states[k])
    print("\n\nNote: [coord2] corresponds to [state], whereas [coord1] is before random jump")
