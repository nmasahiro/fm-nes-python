import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime, os
import pandas as pd
import sys
import time
import numpy as np
import cv2



params = {}
dim = 40
params['graph_end'] = 240000
# params['graph_end'] = 220000

def plot_fval(df, path, save=True):
    plt.figure()
    plt.title('%s/eval' % path)
    df.plot(kind='line', x='evals', y='fval')
    plt.xlim([0, params['graph_end']])

    plt.ylim([1e-12, 1e+6])
    plt.yscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel(r'$f(\mathbf{x}_\mathrm{best})$')
    plt.grid()
    plt.minorticks_on()

    if save:
        plt.savefig('%s/eval.png' % path)
        # plt.savefig('%s/eval.pdf' % path)
        plt.close()
    else:
        plt.show()

def plot_mean(df, path, n, color=cm.hsv, save=True):
    plt.figure()
    plt.title('%s/meanvector' % path)
    test_arr = []
    for i in range(n):
        test, = plt.plot(df['evals'], df['m%d' % i], color=color(float(i)/n), \
                         label='m%d' % i)
        # test, = df.plot(kind='line', x='evals', y='m%d' % i, color=color(float(i)/n), \
        #                  label='m%d' % i)
        test_arr.append(test)

    # plt.legend(handles=test_arr)
    plt.xlabel('# evaluations')
    plt.ylabel('position of mean vector')
    plt.xlim([0, params['graph_end']])

    plt.ylim(-2, 2)
    # plt.ylim(-2, 50)
    plt.grid()
    plt.minorticks_on()
    if save:
        plt.savefig('%s/mean.png' % path)
        # plt.savefig('%s/mean.pdf' % path)
        plt.close()
    else:
        plt.show()

def plot_eig(df, path, n, color=cm.hsv, save=True):
    plt.figure()
    plt.title('%s/d' % path)
    for i in range(n):
        plt.plot(df['evals'], df['eig%d' % i], color=color(float(i)/n), \
                 label='eig%d' % i)
    plt.xlabel('# evaluations')
    plt.ylabel('eigenvalues')
    plt.xlim([0, params['graph_end']])

    plt.ylim([1e-6, 1e+2])
    plt.yscale('log')
    plt.grid()
    plt.minorticks_on()
    if save:
        plt.savefig('%s/eig.png' % path)
        # plt.savefig('%s/eig.pdf' % path)
        plt.close()
    else:
        plt.show()


def plot_stepsize(df, path, save=True):
    plt.figure()
    plt.title('%s/stepsize' % path)
    plt.plot(df['evals'], df['stepsize'])
    plt.xlim([0, params['graph_end']])

    plt.ylim([1e-6, 1e+2])
    plt.yscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel(r'stepsize')

    plt.grid()
    plt.minorticks_on()
    if save:
        plt.savefig('%s/stepsize.png' % path)
        # plt.savefig('%s/stepsize.pdf' % path)
        plt.close()
    else:
        plt.show()

# for discussion
def plot_fval_and_eig(df, path, n, save=True):
    # plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_xlim([0, params['graph_end']])
    # ax1.set_xticks([0,100000, 200000, 300000, 400000])
    # ax1.set_xticklabels(['0', '1', '2', '3', '4'])

    ax1.plot(df['evals'], df['fval'], color='blue', label=r'$f(\mathbf{x}_\mathrm{best})$')
    # ax1.plot(df['evals'], df['fval'], color='red', label=r'eigenvalues')
    # ax1.plot(df['evals'], df['fval'], color='blue')


    print(df['evals'])
    for i in range(n):
        ax2.plot(df['evals'], df['eig%d' % i], color='red')

    ax1.set_xlabel(r'# number of evaluations ($\times 10^5$)')
    ax1.set_ylabel(r'$f(\mathbf{x}_\mathrm{best})$')
    ax1.set_yscale('log')
    ax1.set_ylim([1e-12, 1e+6])
    ax2.set_yscale('log')
    ax2.set_ylabel(r'sqrt of eigenvalue')
    ax2.set_ylim([1e-6, 1e+2])

    plt.grid()
    plt.minorticks_on()

    # ax1.legend()
    # plt.legend()

    if save:
        plt.savefig('%s/eval_and_eig.pdf' % path)
        # plt.savefig('%s/eval_and_eig.png' % path)
        plt.close()
    else:
        plt.show()

def plot_fval_and_sigma(df, path, n, save=True):
    # plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_xlim([0, params['graph_end']])
    # ax1.set_xticks([0,50000, 100000, 120000])
    # ax1.set_xticklabels(['0', '0.5', '1', '1.2'])

    ax1.plot(df['evals'], df['fval'], color='blue', label=r'$f(\mathbf{x}_\mathrm{best})$')
    ax1.plot(df['evals'], df['fval'], color='green', label=r'stepsize')
    ax1.plot(df['evals'], df['fval'], color='blue')

    ax2.plot(df['evals'], df['stepsize'], color='green')

    ax1.set_xlabel(r'# number of evaluations ($\times 10^5$)')
    ax1.set_ylabel(r'$f(\mathbf{x}_\mathrm{best})$')
    ax1.set_yscale('log')
    ax1.set_ylim([1e-12, 1e+6])
    ax2.set_yscale('log')
    ax2.set_ylabel(r'stepsize')
    ax2.set_ylim([1e-8, 1e+1])

    plt.grid()
    plt.minorticks_on()

    # ax1.legend()
    # plt.legend()

    if save:
        # plt.savefig('%s/eval_and_sigma.pdf' % path)
        plt.savefig('%s/eval_and_sigma.png' % path)
        plt.close()
    else:
        plt.show()

# TODO: log(m)
def plot_log_mean(df, path, n, color=cm.hsv, save=True):
    plt.figure()
    plt.title('%s/meanvector' % path)
    test_arr = []
    for i in range(n):
        test, = plt.plot(df['evals'], df['m%d' % i], color=color(float(i)/n), \
                         label='m%d' % i)
        # test, = df.plot(kind='line', x='evals', y='m%d' % i, color=color(float(i)/n), \
        #                  label='m%d' % i)
        test_arr.append(test)

    # plt.legend(handles=test_arr)
    plt.xlabel('# evaluations')
    plt.ylabel('(log) position of mean vector')
    plt.yscale('log')
    plt.xlim([0, params['graph_end']])

    plt.ylim(-2, 20)
    # plt.ylim(-2, 50)
    plt.grid()
    plt.minorticks_on()
    if save:
        plt.savefig('%s/log_mean.png' % path)
        # plt.savefig('%s/mean.pdf' % path)
        plt.close()
    else:
        plt.show()



# TODO: |p_sigma|
def plot_norm_ps(df, path, dim, save=True):
    plt.figure()
    plt.title('%s/norm_ps' % path)
    df.plot(kind='line', x='evals', y='norm_ps')
    plt.xlim([0, params['graph_end']])

    # plt.ylim([1e-12, 1e+6])
    # plt.yscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel(r'$\| p_{\sigma} \|$')
    plt.grid()
    plt.minorticks_on()

    # plot chiN
    chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
    plt.plot(df['evals'], np.array([chiN for i in range(len(df['evals']))]),
             label=r'$\mathbb{E} [\| \mathcal{N}({\bf 0}, {\bf I}) \| ]$')
    plt.legend()

    # plot axvline
    search_phase = {'move': 0, 'stag': 1, 'conv': 2}
    def judge_search_phase(norm):
        if norm >= chiN:
            return search_phase['move']
        elif norm >= 0.1 * chiN:
            return search_phase['stag']
        else:
            return search_phase['conv']

    df['search_phase'] = df['norm_ps'].map(lambda norm_ps: judge_search_phase(norm_ps))
    c_list = ["blue", "red", "green"]
    for i in range(len(df['search_phase'])):
        plt.axvline(x=df['evals'][i],
                    color= c_list[df['search_phase'][i]],
                    alpha=1.0, ymin=10 / 12, ymax=12 / 12, linewidth=2.3)

    if save:
        plt.savefig('%s/norm_ps.png' % path)
        # plt.savefig('%s/eval.pdf' % path)
        plt.close()
    else:
        plt.show()


# TODO: ratio of feasible solutions
def plot_ratio_of_feasible(df, path, dim, save=True):
    plt.figure()
    plt.title('%s/ratio of feasible' % path)
    plt.plot(df['evals'], df['ratio_feasible'])
    plt.xlim([0, params['graph_end']])
    plt.ylim([0.0, 1.0])

    # plt.ylim([1e-12, 1e+6])
    # plt.yscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel(r'ratio of feasible solutions')
    plt.grid()
    plt.minorticks_on()

    if save:
        plt.savefig('%s/ratio_feasible.png' % path)
        # plt.savefig('%s/eval.pdf' % path)
        plt.close()
    else:
        plt.show()

# TODO: normalized eigenvalues
def plot_eig_normalized(df, path, n, color=cm.hsv, save=True):
    plt.figure()
    plt.title('%s/d' % path)
    maximum_eigs = np.maximum.reduce([df['eig%d' % i] for i in range(n)])
    for i in range(n):
        df['eig_normalized'] = df['eig%d' % i] / maximum_eigs
        plt.plot(df['evals'], df['eig_normalized'], color=color(float(i)/n), \
                 label='eig_normalized%d' % i)
    plt.xlabel('# evaluations')
    plt.ylabel('eigenvalues')
    plt.xlim([0, params['graph_end']])

    plt.ylim([1e-4, 5])
    plt.yscale('log')
    plt.grid()
    plt.minorticks_on()
    if save:
        plt.savefig('%s/eig_normalized.png' % path)
        plt.close()
    else:
        plt.show()

# TODO: gamma
def plot_gamma(df, path, save=True):
    plt.figure()
    plt.title('%s/gamma' % path)
    plt.plot(df['evals'], df['gamma'])
    plt.xlim([0, params['graph_end']])
    # plt.ylim([0.0, 1.0])

    # plt.ylim([1e-12, 1e+6])
    # plt.yscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel(r'gamma')
    plt.grid()
    plt.minorticks_on()

    if save:
        plt.savefig('%s/gamma.png' % path)
        # plt.savefig('%s/eval.pdf' % path)
        plt.close()
    else:
        plt.show()


# https://karaage.hatenadiary.jp/entry/2016/01/29/073000
# def concat_plot(path):
#     # fval_and_eig
#     fval_and_eig = cv2.imread('{}/')
# img2 = cv2.imread('image-2.jpg')
# img3 = cv2.imread('image-3.jpg')
# img4 = cv2.imread('image-4.jpg')
#
# img5 = cv2.vconcat([img1, img2])
# img6 = cv2.vconcat([img3, img4])
# img7 = cv2.hconcat([img5, img6])



    # log(m)

    # eig_normalized

    # stepsize

    # norm(ps)

    # ratio of feasible



df = pd.read_csv('log.csv')
path = 'log/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
if not os.path.isdir(path):
    os.makedirs(path)

print(df.head(3))

df=df.astype(float)
plot_fval(df, path)
plot_mean(df, path, dim)
plot_log_mean(df, path, dim)
plot_eig(df, path, dim)
plot_stepsize(df, path)
plot_fval_and_eig(df, path, dim)
plot_fval_and_sigma(df, path, dim)
plot_norm_ps(df, path, dim)
plot_ratio_of_feasible(df, path, dim)
plot_eig_normalized(df, path, dim)
plot_gamma(df, path)

print("finish!")
