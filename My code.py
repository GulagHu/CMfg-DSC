import numpy as np
from My_algos import *
from gurobipy import *
import xlsxwriter as xw
import time

if __name__ == '__main__':
    T = 10  # Input data
    A = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                  [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
                  [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                  [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                  [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                  [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
                  [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])

    r1 = [7, 5, 16, 1, 1, 20, 10, 18, 7, 14, 17, 19, 14, 1, 2]
    r2 = [371, 55, 100, 11, 32, 5, 44, 86, 1, 144, 28, 39, 149, 444, 250]  # Reward(Extreme)
    C1 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    C3 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]  # Capacity(Partial abundant)
    D = [6, 8, 10, 7, 7, 9, 16, 15, 8, 9, 13, 16, 9, 7, 14, 8, 12, 15, 11, 14]  # Duration
    num_samples = 100
    n = len(r1)
    tasks = range(n)
    file_path = 'C:/Users/???/'
    algo_names = ['bayes', 'greedy', 'static', 'fluid', 'dual_k', 'oppo']
    p_iid = np.array([0.075, 0.075, 0.125, 0.025, 0.05, 0.062, 0.062, 0.1, 0.1, 0.05, 0.125, 0.012, 0.075, 0.062, 0.002])

    '''-------------------------Test 1: i.i.d. case---------------------------'''
    workbook1 = xw.Workbook(file_path + 'test 1.xlsx')

    worksheet_offline = workbook1.add_worksheet('offline')
    for name in algo_names:
        exec('worksheet_' + name + ' = workbook1.add_worksheet(\'' + name + '\')')

    for counter in range(1, 101):  # counter is scale
        Samples_iid = np.random.choice(tasks, (num_samples, T * counter), p=p_iid)
        print('Now we start!')

        start = time.time()
        r_offline_iid = offline(A, r1, C1, T * counter, Samples_iid, D)
        stop = time.time()
        r_offline_iid = np.append(r_offline_iid, stop - start)
        xw_toexcel(r_offline_iid, worksheet_offline, counter)
        print('Workbook 1, offline,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        # offline first
        for i in range(len(algo_names)):
            start = time.time()
            exec('r_' + algo_names[i] + '_iid = ' + algo_names[i] + '(A, r1, C1, T * counter, Samples_iid, D, p_iid)')
            stop = time.time()
            exec('r_' + algo_names[i] + '_iid = np.append(r_' + algo_names[i] + '_iid, stop - start)')
            exec('xw_toexcel(r_' + algo_names[i] + '_iid, worksheet_' + algo_names[i] + ', counter)')
            print('Workbook 1,', algo_names[i], ', counter =', str(counter), ', take', str(stop - start), 'seconds')

    workbook1.close()

    '''----------------------Test 1: non-i.i.d. case------------------------'''
    workbook2 = xw.Workbook(file_path + 'test 2.xlsx')

    worksheet_offline = workbook2.add_worksheet('offline')
    for name in algo_names:
        exec('worksheet_' + name + ' = workbook2.add_worksheet(\'' + name + '\')')

    for counter in range(1, 101):
        Samples_niid, p_niid = generate_Samples(T * counter, n, num_samples)
        print('A new round starts!')

        start = time.time()
        r_offline_niid = offline(A, r1, C1, T * counter, Samples_niid, D)
        stop = time.time()
        r_offline_niid = np.append(r_offline_niid, stop - start)
        xw_toexcel(r_offline_niid, worksheet_offline, counter)
        print('Workbook 2, offline,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        for i in range(len(algo_names)):
            start = time.time()
            exec('r_' + algo_names[i] + '_niid = ' + algo_names[i] + '(A, r1, C1, T*counter, Samples_niid, D, p_niid)')
            stop = time.time()
            exec('r_' + algo_names[i] + '_niid = np.append(r_' + algo_names[i] + '_niid, stop - start)')
            exec('xw_toexcel(r_' + algo_names[i] + '_niid, worksheet_' + algo_names[i] + ', counter)')
            print('Workbook 2,', algo_names[i], ', counter =', str(counter), ', take', str(stop - start), 'seconds')

    workbook2.close()

    '''-----------------Test 2: history information----------------'''

    workbook3 = xw.Workbook(file_path + 'test 3.xlsx')

    worksheet_offline = workbook3.add_worksheet('offline')
    worksheet_dual_u_2 = workbook3.add_worksheet('dual_u_2')
    worksheet_dual_u_5 = workbook3.add_worksheet('dual_u_5')
    worksheet_dual_u_10 = workbook3.add_worksheet('dual_u_10')
    worksheet_pac_2 = workbook3.add_worksheet('pac_2')
    worksheet_pac_5 = workbook3.add_worksheet('pac_5')
    worksheet_pac_10 = workbook3.add_worksheet('pac_10')

    for counter in range(20, 220, 20):
        Samples, p_niid = generate_Samples(T * counter, n, num_samples)
        print('A new round starts!')

        start = time.time()
        r_offline = offline(A, r1, C1, T * counter, Samples, D)
        stop = time.time()
        r_offline = np.append(r_offline, stop - start)
        xw_toexcel(r_offline, worksheet_offline, counter)
        print('Workbook 3, offline,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        start = time.time()
        r_dual_u_2 = dual_u(A, r1, C1, T * counter, Samples, D, 2)
        stop = time.time()
        r_dual_u_2 = np.append(r_dual_u_2, stop - start)
        xw_toexcel(r_dual_u_2, worksheet_dual_u_2, counter)
        print('Workbook 3, dual_u_2,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        start = time.time()
        r_dual_u_5 = dual_u(A, r1, C1, T * counter, Samples, D, 5)
        stop = time.time()
        r_dual_u_5 = np.append(r_dual_u_5, stop - start)
        xw_toexcel(r_dual_u_5, worksheet_dual_u_5, counter)
        print('Workbook 3, dual_u_5,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        start = time.time()
        r_dual_u_10 = dual_u(A, r1, C1, T * counter, Samples, D, 10)
        stop = time.time()
        r_dual_u_10 = np.append(r_dual_u_10, stop - start)
        xw_toexcel(r_dual_u_10, worksheet_dual_u_10, counter)
        print('Workbook 3, dual_u_10,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        start = time.time()
        r_pac_2 = pac(A, r1, C1, T * counter, Samples, D, 2)
        stop = time.time()
        r_pac_2 = np.append(r_pac_2, stop - start)
        xw_toexcel(r_pac_2, worksheet_pac_2, counter)
        print('Workbook 3, pac_2,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        start = time.time()
        r_pac_5 = pac(A, r1, C1, T * counter, Samples, D, 5)
        stop = time.time()
        r_pac_5 = np.append(r_pac_5, stop - start)
        xw_toexcel(r_pac_5, worksheet_pac_5, counter)
        print('Workbook 3, pac_5,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

        start = time.time()
        r_pac_10 = pac(A, r1, C1, T * counter, Samples, D, 10)
        stop = time.time()
        r_pac_10 = np.append(r_pac_10, stop - start)
        xw_toexcel(r_pac_10, worksheet_pac_10, counter)
        print('Workbook 3, pac_10,', 'counter =', str(counter), ', take', str(stop - start), 'seconds')

    workbook3.close()

    '''----------------Test 3: Extreme reward (iid & non-iid)----------------'''
    workbook4 = xw.Workbook(file_path + 'test 4.xlsx')
    Horizon = [50, 100, 200, 500, 1000]
    algo_names2 = ['bayes_', 'greedy_', 'static_', 'fluid_', 'dual_k_', 'oppo_']

    worksheet_offline = workbook4.add_worksheet('offline')
    for name in algo_names:
        exec('worksheet_' + name + ' = workbook4.add_worksheet(\'' + name + '\')')

    for rounds in Horizon:
        Samples_iid = np.random.choice(tasks, (num_samples, rounds), p=p_iid)
        print('iid round ' + str(rounds) + ' starts!')

        start = time.time()
        r_offline_iid = offline(A, r2, C1, rounds, Samples_iid, D)
        stop = time.time()
        r_offline_iid = np.append(r_offline_iid, stop - start)
        xw_toexcel(r_offline_iid, worksheet_offline, rounds)
        print('Workbook 4, offline iid, rounds =', str(rounds), ', take', str(stop - start), 'seconds')

        # offline first
        for i in range(len(algo_names)):
            start = time.time()
            exec('r_' + algo_names[i] + '_iid = ' + algo_names[i] + '(A, r2, C1, rounds, Samples_iid, D, p_iid)')
            stop = time.time()
            exec('r_' + algo_names[i] + '_iid = np.append(r_' + algo_names[i] + '_iid, stop - start)')
            exec('xw_toexcel(r_' + algo_names[i] + '_iid, worksheet_' + algo_names[i] + ', rounds)')
            print('Workbook 4,', algo_names[i], 'iid, rounds =', str(rounds), ', take', str(stop - start), 'seconds')

    worksheet_offline_ = workbook4.add_worksheet('offline_')
    for name in algo_names2:
        exec('worksheet_' + name + ' = workbook4.add_worksheet(\'' + name + '\')')

    for rounds in Horizon:
        Samples_niid, p_niid = generate_Samples(rounds, n, num_samples)
        print('non-iid round ' + str(rounds) + ' starts!')

        start = time.time()
        r_offline_niid = offline(A, r2, C1, rounds, Samples_niid, D)
        stop = time.time()
        r_offline_niid = np.append(r_offline_niid, stop - start)
        xw_toexcel(r_offline_niid, worksheet_offline_, rounds)
        print('Workbook 4, offline niid,', 'rounds =', str(rounds), ', take', str(stop - start), 'seconds')

        for i in range(len(algo_names2)):
            start = time.time()
            exec('r_' + algo_names2[i] + '_niid = ' + algo_names[i] + '(A, r2, C1, rounds, Samples_niid, D, p_niid)')
            stop = time.time()
            exec('r_' + algo_names2[i] + '_niid = np.append(r_' + algo_names2[i] + '_niid, stop - start)')
            exec('xw_toexcel(r_' + algo_names2[i] + '_niid, worksheet_' + algo_names2[i] + ', rounds)')
            print('Workbook 4,', algo_names2[i], 'niid, rounds =', str(rounds), ', take', str(stop - start), 'seconds')

    workbook4.close()

    '''----------------Test 4: Service supply (iid or non-iid)----------------'''
    workbook5 = xw.Workbook(file_path + 'test 5.xlsx')
    Horizon = [50, 100, 200, 500, 1000]
    algo_names2 = ['bayes_', 'greedy_', 'static_', 'fluid_', 'dual_k_', 'oppo_']

    worksheet_offline = workbook5.add_worksheet('offline')
    for name in algo_names:
        exec('worksheet_' + name + ' = workbook5.add_worksheet(\'' + name + '\')')

    for rounds in Horizon:
        Samples_iid = np.random.choice(tasks, (num_samples, rounds), p=p_iid)
        print('iid round ' + str(rounds) + ' starts!')

        start = time.time()
        r_offline_iid = offline(A, r1, C3, rounds, Samples_iid, D)
        stop = time.time()
        r_offline_iid = np.append(r_offline_iid, stop - start)
        xw_toexcel(r_offline_iid, worksheet_offline, rounds)
        print('Workbook 5, offline iid, rounds =', str(rounds), ', take', str(stop - start), 'seconds')

        # offline first
        for i in range(len(algo_names)):
            start = time.time()
            exec('r_' + algo_names[i] + '_iid = ' + algo_names[i] + '(A, r1, C3, rounds, Samples_iid, D, p_iid)')
            stop = time.time()
            exec('r_' + algo_names[i] + '_iid = np.append(r_' + algo_names[i] + '_iid, stop - start)')
            exec('xw_toexcel(r_' + algo_names[i] + '_iid, worksheet_' + algo_names[i] + ', rounds)')
            print('Workbook 5,', algo_names[i], 'iid, rounds =', str(rounds), ', take', str(stop - start), 'seconds')

    worksheet_offline_ = workbook5.add_worksheet('offline_')
    for name in algo_names2:
        exec('worksheet_' + name + ' = workbook5.add_worksheet(\'' + name + '\')')

    for rounds in Horizon:
        Samples_niid, p_niid = generate_Samples(rounds, n, num_samples)
        print('non-iid round ' + str(rounds) + ' starts!')

        start = time.time()
        r_offline_niid = offline(A, r1, C3, rounds, Samples_niid, D)
        stop = time.time()
        r_offline_niid = np.append(r_offline_niid, stop - start)
        xw_toexcel(r_offline_niid, worksheet_offline_, rounds)
        print('Workbook 5, offline niid,', 'rounds =', str(rounds), ', take', str(stop - start), 'seconds')

        for i in range(len(algo_names2)):
            start = time.time()
            exec('r_' + algo_names2[i] + '_niid = ' + algo_names[i] + '(A, r1, C3, rounds, Samples_niid, D, p_niid)')
            stop = time.time()
            exec('r_' + algo_names2[i] + '_niid = np.append(r_' + algo_names2[i] + '_niid, stop - start)')
            exec('xw_toexcel(r_' + algo_names2[i] + '_niid, worksheet_' + algo_names2[i] + ', rounds)')
            print('Workbook 5,', algo_names2[i], 'niid, rounds =', str(rounds), ', take', str(stop - start), 'seconds')

    workbook5.close()

