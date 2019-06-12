from SwarmPackagePy.intelligence import sw
import numpy as np

class pso(sw):
    def __init__(self, n, function, lb, ub, dimension, iteration, seeds, init=None, w=0.5, c1=1,
                 c2=1):

        super(pso, self).__init__()
        self.scores = []

        if init is None:
            self.__agents = np.random.uniform(lb, ub, (n, dimension))
        else:
            self.__agents = np.array(init)

        velocity = np.zeros((n, dimension))
        self._points(self.__agents)

        Pbest = self.__agents[np.array([function(x, 0)
                                        for x in self.__agents]).argmax()]
        Gbest = Pbest

        for t in range(iteration):

            r1 = np.random.random((n, dimension))
            r2 = np.random.random((n, dimension))
            velocity = w * velocity + c1 * r1 * (
                Pbest - self.__agents) + c2 * r2 * (
                Gbest - self.__agents)
            self.__agents += velocity
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)

            Pbest = self.__agents[
                np.array([function(x, seeds[t]) for x in self.__agents]).argmax()]

            self.scores.append(function(Pbest, seeds[t]))
            if function(Pbest, seeds[t]) > function(Gbest, seeds[t]):
                Gbest = Pbest

        self._set_Gbest(Gbest)

    def get_Score(self):
        return self.scores