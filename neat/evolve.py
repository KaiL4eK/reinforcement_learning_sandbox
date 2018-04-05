import os
import argparse

import pickle
import time

import neat
import visualize

import reporters as r

parser = argparse.ArgumentParser(description="Description")
parser.add_argument("--chk",    help="Path to checkpoint file to restore",  default=None, action="store")
parser.add_argument("--pops",   help="Count of population",                 default=None, action="store")
parser.add_argument("--cores",  help="Count of processing cores",           default=2,      action="store")
parser.add_argument("--ctrnn",  help="Network with backward connections",   default=False, action="store_true")

ns = parser.parse_args()

populationCount     = int(ns.pops) if ns.pops is not None else None
restoreCheckpoint   = ns.chk
numCores            = int(ns.cores)
print('Processing {} cores'.format(numCores))

ctrnn_flag = ns.ctrnn

if not ctrnn_flag:
    from evaluate_ff import *

    checkpointDir       = 'checkpoints_ff'
    winnerFname         = 'winner_ff'
    pictureDir          = 'pictures_ff'
    configFname         = 'config-feedforward'
    logFname            = 'log_ff.txt'

    print('Processing NEAT with FF structure')
else:
    from evaluate_ctrnn import *

    checkpointDir       = 'checkpoints_ctrnn'
    winnerFname         = 'winner_ctrnn'
    pictureDir          = 'pictures_ctrnn'
    configFname         = 'config-ctrnn'
    logFname            = 'log_ctrnn.txt'

    print('Processing NEAT with CTRNN structure')

checkpointPrefix    = checkpointDir + '/chk_'

if not os.path.exists(checkpointDir):
    os.makedirs(checkpointDir)

if not os.path.exists(pictureDir):
    os.makedirs(pictureDir)

def run():
    if restoreCheckpoint is None:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, configFname)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        pop = neat.Population(config)
    else:
        pop = neat.Checkpointer.restore_checkpoint(restoreCheckpoint)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(r.FileReporter(logFname, True))
    pop.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=checkpointPrefix))

    # if render_flag:
    # winner = pop.run(eval_genomes)
    # else:
    pe = neat.ParallelEvaluator(numCores, eval_genome)
    winner = pop.run(pe.evaluate, n=populationCount)

    # Save the winner.
    with open(winnerFname, 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, view=False, ylog=True, filename=pictureDir+'/fitness.svg')
    visualize.plot_species(stats, view=False, filename=pictureDir+'/speciation.svg')

    # node_names = {-1: 'ext', -2: 'eyt', -3: 'sf', -4: 'sl', -5: 'sr', -6: 'sb', 0: 'ux', 1: 'uy'}
    # visualize.draw_net(config, winner, False, node_names=node_names,
                       # filename='pictures_ff/Digraph.gv')
    # visualize.draw_net(config, winner, view=False, node_names=node_names,
                       # filename="pictures_ff/winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=False, node_names=node_names,
                       # filename="pictures_ff/winner-feedforward-enabled.gv", show_disabled=False)
    #visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                   filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    run()

