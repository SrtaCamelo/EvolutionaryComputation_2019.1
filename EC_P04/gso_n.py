import numpy as np
import fitness as ft

def gso( lb, ub, ite, population,x_train, y_train, x_test, y_test,x_validate,y_validate):
    popsize = np.array(population).shape[0]
    ndim = np.array(population).shape[1]

    initial_angles = np.pi / 4 * np.ones((popsize, (ndim - 1)))
    angle = initial_angles.copy()
    lowerbound = np.zeros((popsize, ndim))
    upperbound = np.zeros((popsize, ndim))

    for x in range(popsize):
        lowerbound[x, :] = lb
        upperbound[x, :] = ub

    vmax = np.ones((popsize, ndim))
    for x in range(ndim):
        vmax[:, x] = upperbound[:, x] - lowerbound[:, x]

    l_max = np.linalg.norm(vmax[1, :])

    distance = l_max * np.tile(np.ones((popsize, 1)), (ndim, 1))
    distance = distance.reshape((popsize), ndim)

    a = np.round((ndim + 1) ** 0.5)

    max_pursuit_angle = (np.pi / (a ** 2))
    max_turning_angle = max_pursuit_angle / 2

    direction = np.zeros((popsize, ndim))

    for x in range(popsize):
        direction[x, 0] = (np.cos(angle[x, 1]))
        for y in range(1, ndim - 1):
            direction[x, y] = np.cos(angle[x, y]) * np.prod(np.sin(angle[x, y:ndim - 1]))
        direction[x, ndim - 1] = np.prod(np.sin(angle[x, :ndim]))

    fvalue = np.zeros(popsize)

    for i in range(popsize):
        evaluated = ft.fitness(population[i],x_train, y_train,x_validate, y_validate)
        fvalue[i] = evaluated[0]

    outflag = np.where((population <= lowerbound) | (population >= upperbound), 1, 0)
    population = population - outflag * distance * direction

    index = np.argmax(fvalue)
    fbestval = fvalue[index]
    bestmember = population[index]
    # print(fbestval)

    oldangle = angle.copy()
    oldindex = index.copy()
    badcounter = 0
    atribu = 0
    best = np.zeros(ite)

    for itera in range(ite):

        for j in range(popsize):
            R1 = np.random.randn(1)
            R2 = np.random.rand(1, ndim - 1)[0]
            R3 = np.random.rand(1, ndim)[0]

            if j == index:

                distance[j, :] = l_max * R1

                sampleposition = []
                sampleangle = []
                samplevalue = []
                sampledirection = []

                if badcounter >= a ** 2:
                    angle[j, :] = oldangle[j, :]

                sampleposition.append(population[j, :])
                sampleangle.append(angle[j, :])
                samplevalue.append(fvalue[j])
                sampledirection.append(direction[j, :])

                ### LOOK STRAIGHT
                direction[j, 0] = np.prod(np.cos(angle[j, :ndim - 1]))
                for i in range(i, ndim - 1):
                    direction[j, i] = np.sin(angle[j, i]) * np.prod(np.cos(angle[j, i:ndim - 1]))
                direction[j, ndim - 1] = np.sin(angle[j, ndim - 2])

                straightposition = population[j, :] + distance[j, :] * direction[j, :]
                outflag = np.where((straightposition > upperbound[j, :]) | (straightposition < lowerbound[j, :]), 1, 0)
                straightposition = straightposition - outflag * distance[j, :] * direction[j, :]
                evaluated = ft.fitness(population[j], x_train, y_train, x_validate, y_validate)
                straightfvalue = evaluated[0]
                sampleposition.append(straightposition)
                sampleangle.append(angle[j, :])
                samplevalue.append(straightfvalue)
                sampledirection.append(direction[j, :])
                ###### END LOOK STRAIGHT

                ###### LOOK LEFT
                leftangle = angle[j, :] + max_pursuit_angle * R2 / 2
                direction[j, 0] = np.prod(np.cos(leftangle[:ndim - 1]))
                for i in range(1, ndim - 1):
                    direction[j, i] = np.sin(leftangle[i]) * np.prod(np.cos(leftangle[i:ndim - 1]))
                direction[j, ndim - 1] = np.sin(leftangle[ndim - 2])

                leftposition = population[j, :] + distance[j, :] * direction[j, :]
                outflag = np.where((leftposition > upperbound[j, :]) | (leftposition < lowerbound[j, :]), 1, 0)
                leftposition = leftposition - outflag * distance[j, :] * direction[j, :]
                evaluated = ft.fitness(population[j], x_train, y_train, x_validate, y_validate)
                leftfvalue = evaluated[0]
                sampleposition.append(leftposition)
                sampleangle.append(leftangle[:])
                samplevalue.append(leftfvalue)
                sampledirection.append(direction[j, :])
                ####END LOOK LEFT

                ##### LOOK RIGHT
                rightangle = angle[j, :] - max_pursuit_angle * R2 / 2  # look right
                direction[j, 1] = np.prod(np.cos(rightangle[:ndim - 1]))
                for i in range(1, ndim - 1):
                    direction[j, i] = np.sin(rightangle[i]) * np.prod(np.cos(rightangle[i:ndim - 1]))
                direction[j, ndim - 1] = np.sin(rightangle[ndim - 2])

                rightposition = population[j, :] + distance[j, :] * direction[j, :]
                outflag = np.where((rightposition > upperbound[j, :]) | (rightposition < lowerbound[j, :]), 1, 0)
                rightposition = rightposition - outflag * distance[j, :] * direction[j, :]
                evaluated = ft.fitness(population[j], x_train, y_train, x_validate, y_validate)
                rightfvalue = evaluated[0]
                sampleposition.append(rightposition)
                sampleangle.append(rightangle[:])
                samplevalue.append(rightfvalue)
                sampledirection.append(direction[j, :])
                #### END LOOK RIGHT

                best_position_id = np.argmax(samplevalue)
                fbestdirectionval = samplevalue[best_position_id]

                population[j, :] = sampleposition[best_position_id][:]

                if best_position_id != 1:
                    angle[j, :] = sampleangle[best_position_id][:]
                    oldangle[j, :] = angle[j, :].copy()
                    badcounter = 0
                else:
                    badcounter += 1
                    angle[j, :] = angle[j, :] + max_turning_angle * R2

            else:
                # print(j)
                angle[j, :ndim - 1] = angle[j, :ndim - 1] + max_turning_angle * R2

                if np.random.rand() > 0.2:

                    distance[j, :] = R3 * (bestmember - population[j, :])
                    population[j, :] = population[j, :] + distance[j, :]

                else:

                    distance[j, :] = l_max * np.tile(a * R1, (1, ndim))
                    direction[j, 1] = np.cos(angle[j, 1])
                    for i in range(1, ndim - 1):
                        direction[j, i] = np.cos(angle[j, i]) * np.prod(np.sin(angle[j, i:ndim - 1]))
                    direction[j, ndim - 1] = np.prod(np.sin(angle[j, 1:ndim - 1]))
                    population[j, :] = population[j, :] + distance[j, :] * direction[j, :]

        outflag = np.where((population <= lowerbound) | (population >= upperbound), 1, 0)
        population = population - outflag * distance * direction

        for i in range(popsize):
            evaluated = ft.fitness(population[i], x_train, y_train, x_validate, y_validate)
            fvalue[i] = evaluated[0]

        index = np.argmax(fvalue)
        fbestval = fvalue[index]
        best[atribu] = fbestval
        atribu += 1

        bestmember = population[index, :]

        # if ite / 10 == np.floor(ite):
        #    print("Best: ", fbestval)
        #    best[int(ite / 10)] = fbestval
        #    if fbestval == 0:
        #        best[(ite / 10):300] = 0
        #        break

    return fbestval, bestmember, best