import numpy as np
from gurobipy import *
import copy

"""
Run Static online packing with known distribution
"""


def static(A, r, C, T, Samples, D, p):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    tasks = range(len(r))
    services = range(len(C))
    horizon = range(1, T + 1)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T, dtype=int)

        # Set up the ALP
        x = create_alp(A, r, C, D, p, T, 1)
        if p.ndim == 1:
            p_accept = [1.0 * x[j] / (T * p[j]) for j in tasks]
        else:
            p_accept = [1.0 * x[j] / sum(p[:, j]) for j in tasks]

        for t in horizon:
            j = Sequence[t - 1]
            for i in services:  # replenish services
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):
                continue  # no enough services

            alpha = np.random.uniform(0, 1)  # Draw algo randomization
            if p_accept[j] >= alpha:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


"""
Run fluid policy with known distribution (benchmark)
"""


def fluid(A, r, C, T, Samples, D, p):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    tasks = range(len(r))
    services = range(len(C))
    horizon = range(1, T + 1)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T, dtype=int)

        for t in horizon:
            # Set up the ALP
            x = create_alp(A, r, C, D, p, T, t)

            j = Sequence[t - 1]
            for i in services:  # replenish services
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):
                continue  # no enough services

            alpha = np.random.uniform(0, 1)  # Draw algo randomization
            if p.ndim == 1:
                p_accept = 1.0 * x[j] / ((T - t + 1) * p[j])
            else:
                p_accept = 1.0 * x[j] / sum(p[t - 1, j] for t in range(t, T + 1))

            if p_accept >= alpha:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


"""
Run greedy policy
"""


def greedy(A, r, C, T, Samples, D, p):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    services = range(len(C))
    horizon = range(1, T + 1)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T)

        for t in horizon:
            j = Sequence[t - 1]
            for i in services:  # replenish services
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if all(C_t[i] >= A[i, j] for i in services):
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


'''
Run Dual-based with known distribution
'''


def dual_k(A, r, C, T, Samples, D, p):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    services = range(len(C))
    horizon = range(1, T + 1)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T, dtype=int)

        # Set up the ALP
        u = create_dual_k(A, r, C, D, p, T)

        for t in horizon:
            j = Sequence[t - 1]
            price = 0
            for i in services:  # replenish services
                price = price + A[i, j] * u[i]
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):
                continue  # no enough services

            if r[j] > price:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


'''
Run Dual-based without distribution
'''


def dual_u(A, r, C, T, Samples, D, K):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    services = range(len(C))
    horizon = range(1, T + 1)
    tau = np.linspace(1, T, K, dtype=int)
    tau = np.delete(tau, 0)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T, dtype=int)
        u_temp = np.zeros(len(C))
        for t in horizon:
            if t in tau:
                u_temp_new = create_dual_u(A, r, C, D, Sequence[0:t], t)
                if str(u_temp_new) != "Unbounded":
                    u_temp = u_temp_new

            j = Sequence[t - 1]
            price = 0
            for i in services:  # replenish services
                price = price + A[i, j] * u_temp[i]
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):
                continue  # no enough services

            if r[j] > price:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


'''
Run PAC in Jasin (2015)
'''


def pac(A, r, C, T, Samples, D, K):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    services = range(len(C))
    horizon = range(1, T + 1)
    tau = np.linspace(1, T, K, dtype=int)
    tau = np.delete(tau, 0)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T, dtype=int)
        p_accept = np.ones(len(C))
        for t in horizon:
            if t in tau:
                y = create_pac(A, r, C, D, T, Sequence[0:t])
                if str(y) != "Unbounded":
                    p_accept = copy.deepcopy(y)

            j = Sequence[t - 1]
            for i in services:  # replenish services
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):
                continue  # no enough services

            alpha = np.random.uniform(0, 1)  # Draw algo randomization
            if p_accept[j] >= alpha:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


'''
Run opportunity-cost based algorithm
'''


def oppo(A, r, C, T, Samples, D, p):
    num_sample = Samples.shape[0]  # amount of samples
    reward = np.zeros(num_sample)

    tasks = range(len(r))
    services = range(len(C))
    horizon = range(1, T + 1)

    r_hat = new_r(A, C, D, p, T, r)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)
        decision = np.zeros(T, dtype=int)

        # Get y_star
        y_star = create_tlp(A, r, C, D, p, T)

        # Calculate Q_hat
        ex = max(D)
        Q_hat = np.zeros((len(tasks), (T + ex)))
        for t in range(T, 0, -1):
            for j in tasks:
                idx_i = generate_Indices(np.argwhere(A[:, j] == 1))
                temp_Q = 0
                for i in idx_i:
                    idx_j_ = generate_Indices(np.argwhere(A[i, :] == 1))
                    temp_Q_i = 0
                    for j_ in idx_j_:
                        for t_ in range(t + 1, t + D[i]):
                            if t_ <= T:
                                temp_Q_i = temp_Q_i + y_star[j_, t_ - 1] * max(r_hat[j_] - Q_hat[j_, t_], 0)
                    temp_Q_i = temp_Q_i / C[i]
                    temp_Q = temp_Q + temp_Q_i
                Q_hat[j, t - 1] = temp_Q

        for t in horizon:
            j = Sequence[t - 1]
            for i in services:  # replenish services
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):
                continue  # no enough services

            if r_hat[j] >= Q_hat[j, t - 1]:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


"""
Run the bayes selector on the given sample paths J
"""


def bayes(A, r, C, T, Samples, D, p):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    services = range(len(C))

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        C_t = copy.deepcopy(C)  # Budget over time
        decision = np.zeros(T, dtype=int)

        for t in range(1, T+1):
            # Set up the LP
            if p.ndim == 1:
                p_t = p
            else:
                p_t = p[t - 1]

            j = Sequence[t - 1]
            for i in services:  # replenish services
                if t > D[i]:
                    t_ = t - 1 - D[i]
                    j_ = Sequence[t_]
                    if A[i, j_] * decision[t_] == 1:
                        C_t[i] = C_t[i] + 1

            if not all(C_t[i] >= A[i, j] for i in services):  # Not enough resources
                continue

            # Update LP and optimize
            flag = create_packing(A, r, C_t, T-t+1, p_t, j)

            if flag:  # Accept
                decision[t - 1] = 1
                reward[s] = reward[s] + r[j]
                C_t = C_t - A[:, j]

    return reward


'''
Run Offline algorithm
'''


def offline(A, r, C, T, Samples, D):
    num_sample = Samples.shape[0]
    reward = np.zeros(num_sample)
    horizon = range(T)

    for s in range(0, num_sample):
        Sequence = Samples[s, :]
        decision = create_bp(A, r, C, D, Sequence, T)
        for t in horizon:
            reward[s] = reward[s] + r[Sequence[t]] * decision[t]

    return reward


"""
-----------------------------------Auxiliary functions---------------------------------------
"""


def new_r(A, C, D, p, T, r):
    Delta = np.zeros(len(C))
    tasks = range(len(r))
    for i in range(len(C)):
        idx = generate_Indices(np.argwhere(A[i, :] == 1))  # E_i
        temp = 0
        if p.ndim == 1:
            for j in idx:
                temp = temp + p[j]
            Delta[i] = min((C[i] / D[i]) / temp, 1)
        else:
            for t in range(T):
                for j in idx:
                    temp = temp + p[t, j]
            Delta[i] = min((C[i] / D[i]) / (temp / T), 1)
    Gamma = np.ones(len(r))
    r_hat = copy.deepcopy(r)
    for j in tasks:
        idx = generate_Indices(np.argwhere(A[:, j] == 1))  # E_j
        for i in idx:
            Gamma[j] = Gamma[j] * Delta[i]
        r_hat[j] = Gamma[j] * r_hat[j]

    return r_hat


"""
---------------------------------Create gurobi models--------------------------------------
"""


def create_bp(A, r, C, D, Sequence, T):
    model = Model("BP")
    services = range(len(C))
    horizon = range(1, T + 1)
    x = model.addVars(horizon, vtype=GRB.BINARY)
    model.update()

    R = np.zeros(T)
    for t in horizon:
        R[t - 1] = r[Sequence[t - 1]]

    model.setObjective(quicksum(R[t - 1] * x[t] for t in horizon), GRB.MAXIMIZE)
    for t in horizon:
        for i in services:
            model.addConstr(quicksum(A[i, Sequence[t_-1]] * x[t_] for t_ in range(max(t-D[i]+1, 1), t+1)) <= C[i])

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    y = np.zeros(T)
    for t in horizon:
        y[t - 1] = x[t].X

    return y


def create_lp(A, r, C, D, Sequence, T):
    model = Model("LP")
    horizon = range(1, T + 1)
    x = model.addVars(horizon, lb=0, ub=1)
    model.update()

    R = np.zeros(T)
    services = range(len(C))
    for t in horizon:
        R[t - 1] = r[Sequence[t - 1]]

    model.setObjective(quicksum(R[t - 1] * x[t] for t in horizon), GRB.MAXIMIZE)
    for i in services:
        model.addConstr(quicksum(A[i, Sequence[t - 1]] * x[t] for t in horizon) <= (T * C[i] / D[i]))

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    y = np.zeros(T)
    for t in horizon:
        y[t - 1] = x[t].X

    return y


def create_tlp(A, r, C, D, p, T):
    model = Model("TLP")
    horizon = range(1, T + 1)
    x = model.addVars(len(r), range(1, T + 1))
    model.update()
    tasks = range(len(r))
    services = range(len(C))

    model.setObjective(quicksum(r[j] * x[j, t] for j in tasks for t in horizon), GRB.MAXIMIZE)
    for t in horizon:
        for i in services:
            model.addConstr(sum(A[i, j] * x[j, t_] for j in tasks for t_ in range(max(t - D[i] + 1, 1), t + 1)) <= C[i])

    model.update()

    if p.ndim == 1:  # i.i.d.
        model.addConstrs(x[j, t] <= p[j] for t in horizon for j in tasks)
    else:  # non-i.i.d.   p_iid = horizon*tasks
        model.addConstrs(x[j, t] <= p[t - 1, j] for t in horizon for j in tasks)

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    y = np.zeros((len(r), T))
    for j in tasks:
        for t in horizon:
            y[j, t - 1] = x[j, t].X

    return y


def create_pac(A, r, C, D, T, Sequence):
    model = Model("ALP")
    x = model.addVars(len(r))
    model.update()

    tasks = range(len(r))
    services = range(len(C))
    time = len(Sequence)
    p = np.zeros(len(r), dtype=int)
    for j in Sequence:
        p[j] = p[j] + 1
    p = p / time

    model.setObjective(sum(r[j] * x[j] for j in tasks), GRB.MAXIMIZE)
    model.addConstrs((sum(A[i, j] * x[j] for j in tasks)) <= ((T - time + 1) * C[i] / D[i]) for i in services)
    model.addConstrs(x[j] <= ((T - time + 1) * p[j]) for j in tasks)

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    if model.Status == 2:
        y = np.zeros(len(r))
        for j in tasks:
            if p[j] != 0:
                y[j] = x[j].X / (T - time + 1) * p[j]
        return y
    elif model.Status == 4:
        model.Params.DualReductions = 0
        model.optimize()
        if model.Status == 5:
            return "Unbounded"


def create_alp(A, r, C, D, p, T, time):
    model = Model("ALP")
    x = model.addVars(len(r))
    model.update()

    tasks = range(len(r))
    services = range(len(C))

    model.setObjective(sum(r[j] * x[j] for j in tasks), GRB.MAXIMIZE)
    model.addConstrs((sum(A[i, j] * x[j] for j in tasks)) <= ((T - time + 1) * C[i] / D[i]) for i in services)
    model.update()

    if p.ndim == 1:  # i.i.d.
        model.addConstrs(x[j] <= ((T - time + 1) * p[j]) for j in tasks)
    else:  # non-i.i.d.  p_iid = T*tasks
        model.addConstrs(x[j] <= sum(p[t - 1, j] for t in range(time, T + 1)) for j in tasks)

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    y = np.zeros(len(r))
    for j in tasks:
        y[j] = x[j].X

    return y


def create_dual_k(A, r, C, D, p, T):
    model = Model("Dual_k")
    u = model.addVars(len(C), name="u")
    z = model.addVars(len(r), name="z")
    v = model.addVars(len(r), name="v")
    model.update()

    tasks = range(len(r))
    services = range(len(C))

    if p.ndim == 1:  # i.i.d.
        model.setObjective(sum(C[i] * u[i] / D[i] for i in services) + sum(p[j]*z[j] for j in tasks), GRB.MINIMIZE)
        for j in tasks:
            model.addConstr(v[j] == r[j] - sum(A[i, j] * u[i] for i in services))
            model.addGenConstrMax(z[j], [v[j], 0])
    else:  # non-i.i.d.  p_niid = T*tasks
        horizon = range(T)
        model.setObjective(sum(C[i] * u[i] / D[i] for i in services) + sum((p[t, j] / T) * z[j] for j in tasks for t in horizon), GRB.MINIMIZE)
        for j in tasks:
            model.addConstr(v[j] == r[j] - sum(A[i, j] * u[i] for i in services))
            model.addGenConstrMax(z[j], [v[j], 0])

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    y = np.zeros(len(C))
    for i in services:
        y[i] = u[i].X

    return y


def create_dual_u(A, r, C, D, horizon, t):
    model = Model("Dual_u")
    u = model.addVars(len(C), name="u")
    z = model.addVars(len(r), name="z")
    v = model.addVars(len(r), name="v")
    model.update()
    services = range(len(C))

    model.setObjective(sum(C[i] * u[i] / D[i] for i in services) + sum(z[j] for j in horizon) / t, GRB.MINIMIZE)
    for j in horizon:
        model.addConstr(v[j] == r[j] - sum(A[i, j] * u[i] for i in services))
        model.addGenConstrMax(z[j], [v[j], 0])

    model.update()
    model.setParam('OutputFlag', 0)
    model.optimize()

    if model.Status == 2:
        y = np.zeros(len(C))
        for i in services:
            y[i] = u[i].X
        return y
    elif model.Status == 4:
        model.Params.DualReductions = 0
        model.optimize()
        if model.Status == 5:
            return "Unbounded"


def create_packing(A, r, C, T, p, j):
    model = Model("Packing")  # Create new models
    model.setParam('OutputFlag', 0)
    tasks = range(len(r))
    services = range(len(C))
    x = model.addVars(len(r))
    model.update()

    model.setObjective(sum(r[j] * x[j] for j in tasks), GRB.MAXIMIZE)
    for i in services:
        model.addConstr(sum(A[i, j] * x[j] for j in tasks) <= C[i])

    model.addConstrs(x[j] <= T * p[j] for j in tasks)

    model.update()
    model.optimize()

    flag = False
    if x[j].X >= T * p[j] / 2:
        flag = True

    return flag


def generate_Samples(T, n, num_samples):
    mat = np.random.rand(T, n - 1)
    p_niid = np.zeros((T, n))  # T * tasks
    Samples = np.zeros((num_samples, T), dtype=int)
    for t in range(T):
        temp = sorted(mat[t])
        temp.insert(0, 0)
        temp.append(1)
        for j in range(n):
            p_niid[t, j] = temp[j + 1] - temp[j]

    for s in range(num_samples):
        for t in range(T):
            Samples[s, t] = np.random.choice(range(n), 1, p=p_niid[t])

    return Samples, p_niid


def generate_Indices(idx_temp):
    idx = idx_temp.transpose()
    idx = idx[0]

    return idx


def xw_toexcel(data, worksheet, counter):
    worksheet.activate()
    row = 'A' + str(counter)
    worksheet.write_row(row, data)
