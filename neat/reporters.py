from __future__ import division, print_function

import time

import neat
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

class FileReporter(neat.reporting.BaseReporter):
    def __init__(self, filename, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

        self.fd = open(filename, 'w')

    def __del__(self):
        self.fd.close()

    def start_generation(self, generation):
        self.generation = generation
        self.fd.write('\n ****** Running generation {0} ****** \n'.format(generation))
        self.fd.write('\n')
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            self.fd.write('Population of {0:d} members in {1:d} species:\n'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            self.fd.write("   ID   age  size  fitness  adj fit  stag\n")
            self.fd.write("  ====  ===  ====  =======  =======  ====\n")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                self.fd.write(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}\n".format(sid, a, n, f, af, st))
        else:
            self.fd.write('Population of {0:d} members in {1:d} species\n'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        self.fd.write('Total extinctions: {0:d}\n'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            self.fd.write("Generation time: {0:.3f} sec ({1:.3f} average)\n".format(elapsed, average))
        else:
            self.fd.write("Generation time: {0:.3f} sec\n".format(elapsed))


    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        self.fd.write('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}\n'.format(fit_mean, fit_std))
        self.fd.write(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}\n'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        self.fd.write('All species extinct.\n')

    def found_solution(self, config, generation, best):
        self.fd.write('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}\n'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            self.fd.write("\nSpecies {0} with {1} members is stagnated: removing it\n".format(sid, len(species.members)))

    def info(self, msg):
        self.fd.write(msg)
        self.fd.write('\n')