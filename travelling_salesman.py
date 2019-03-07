import matplotlib.pyplot as plt
import numpy as np
import random
import math
import operator

'''
    Uses Genetic Algorithm to find the best route possible to visit 54 post offices in Northeast Tennessee
'''

plt.ion()
plt.figure(figsize=(10, 5))


def plotTSP(generation, path, points, path_distance, save, num_iters=1):
    """
    generation: The generation number to display
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    path_distance: the distance to display in the figure
    save: True if saving to final_route.png, False otherwise
    num_iters: number of paths that are in the path list

    SOURCE: https://gist.github.com/payoung/6087046

    """
    ### MOD: Brian Bennett

    plt.clf()
    plt.suptitle("Tennessee Traveling Postal Worker - Generation " + str(generation) + \
                 "\nPath Length: " + str(path_distance))
    ### END MOD

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = [];
    y = []
    for i in path:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'ko')

    # Set a scale for the arrow heads (there should be a reasonable default for this)
    a_scale = float(max(x)) / float(2500)  # MOD: Brian Bennett
    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = [];
            yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                      head_width=a_scale, color='r',
                      length_includes_head=True, ls='dashed',
                      width=0.001 / float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i + 1] - xi[i]), (yi[i + 1] - yi[i]),
                          head_width=a_scale, color='r', length_includes_head=True,
                          ls='dashed', width=0.001 / float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
              color='b', length_includes_head=True)
    for i in range(0, len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]), head_width=a_scale,
                  color='b', length_includes_head=True)

    if save:
        plt.savefig("final_route.png")

    plt.pause(1)


class GeneticSearch:
    """
        Class: GeneticSearch
    """

    def __init__(self, origin, generations, points, cities, population_size, mutation_rate):
        self.population = None
        self.points = points
        self.cities = cities
        self.chromosome_size = len(self.points)
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.origin = origin
        self.origin_index = self.points.index(self.origin)
        self.values = []

    def print_population(self, generation, chromosomes):
        index = 0
        print("===== GENERATION %d" % generation)
        for chromosome in self.population:
            print("Index %5d , Fitness %0.4f : %s" % (index, chromosome[1], ''.join(str(chromosome[0]))))
            index = index + 1
            if index > chromosomes:
                break

    def initialize_population(self):
        '''
        Initialize population
        :return:
        '''
        self.population = []
        for i in range(self.population_size):

            individual = [x for x in range(self.chromosome_size)]
            random.shuffle(individual)

            # Move the origin_index to the front of the path
            individual.remove(self.origin_index)
            individual = [self.origin_index] + individual

            fitness = self.fitnessfcn(individual)

            # Prevent duplicate individuals in the initial population
            while [individual, fitness] in self.population:
                individual = [x for x in range(self.chromosome_size)]
                random.shuffle(individual)

                individual.remove(self.origin_index)
                individual = [self.origin_index] + individual

                fitness = self.fitnessfcn(individual)

            # POPULATION NODES are in the form [chromosome, fitness]
            self.population.append([individual, fitness])

        # Sort the population in descending order
        # -- "Maximize the objective function"
        self.population.sort(key=operator.itemgetter(1), reverse=True)

    def straight_line_distance(self, p1, p2):
        '''
            Return the Euclidian Distance between p1 and p2
        '''
        sld = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        return sld

    def route_distance(self, individual):
        '''
            Determine the distance for the entire route
        '''
        distance = 0
        value = 0

        tour = individual + [self.origin_index]

        index = 0
        p1 = p2 = None
        while p2 != self.origin:
            p1 = self.points[tour[index]]
            p2 = self.points[tour[index + 1]]
            distance += self.straight_line_distance(p1, p2)
            index += 1

        return distance

    def fitnessfcn(self, individual):
        '''
            Return the negative route distance so it can be maximized.
        '''

        return -self.route_distance(individual)

    def select_parents(self, population):
        '''
            Selects the parent that will be used for reproduction
        '''

        return self.perform_fitness_proportionate_strategy(population)

    def perform_fitness_proportionate_strategy(self, population):
        '''
            Get parent for reproduction using fitness proportionate strategy
            Parent with high fitness value(selection probability) will have higher chance of selection
        :return: individual to be used as a parent for reproduction : eg.[1,3,5,8,2,1,4,9]
        '''

        # calculate total fitness value
        total_fitness_value = sum([individual[1] for individual in population])
        # create selection probabilities for eac individual in population
        individual_selection_probabilities = [individual[1]/total_fitness_value for individual in population]

        # use random.choice with probabilities to select parent index,
        # parent with high fitness value(selection probability) will have higher chance of selection

        parent_index = np.random.choice(len(population), p=individual_selection_probabilities)
        parent = population[parent_index]
        return parent[0]

    def reproduce(self, parent1, parent2):
        '''
            Create two children using parent1,parent2 and ordered crossover strategy.
        '''

        random_crossover_point = random.randint(1, self.chromosome_size - 1)
        first_child = parent1[random_crossover_point:]
        second_child = parent2[:random_crossover_point]

        add_to_first_child = [x for x in parent2 if x not in first_child]
        add_to_second_child = [x for x in parent1 if x not in second_child]

        return first_child + add_to_first_child, second_child + add_to_second_child

    def mutate(self, child):
        '''
            Mutation Strategy : Swapped mutation, change the position of two random cities
        '''

        individual = child

        # randomly get points (locations) to swap
        gene1_index = int(random.random() * len(individual))
        gene2_index = int(random.random() * len(individual))

        # ensure gene2_index and gene1_index are different
        while gene2_index == gene1_index:
            gene2_index = int(random.random() * len(individual))

        # get points (locatons) to swap
        first_location = individual[gene1_index]
        second_location = individual[gene2_index]

        # swap points
        individual[gene1_index] = second_location
        individual[gene2_index] = first_location

        return individual

    def move_start_location_to_begining(self, individual):
        '''
            Move start location (origin) to beginning of route (individual)
        :param individual:
        :return: individual with origin location at the beginning
        '''
        individual.remove(self.origin_index)
        individual = [self.origin_index] + individual

        return individual

    def print_result(self):
        '''
            Displays the resulting route in the console.
        '''
        individual = self.population[0][0]
        fitness = self.population[0][1]

        print(" Final Route in %d Generations" % self.generations)
        print(" Final Distance : %5.3f\n" % -fitness)

        counter = 1

        for index in individual:
            print("%2d. %s" % (counter, self.cities[index]))
            counter += 1

        print("%2d. %s" % (counter, self.cities[self.origin_index]))

    def run(self):
        '''
            Run the genetic algorithm. Note that this method initializes the
             first population.
        '''
        generations = 0

        # create initial population
        self.initialize_population()

        while generations <= self.generations:

            # pass the top 9% individuals from the current population to the next generation
            elite_individuals_upper_bound = int(ELITE_PERCENTAGE * self.population_size)

            # get elite population from population
            new_population = self.population[:elite_individuals_upper_bound]

            while len(new_population) < self.population_size:

                # always select parents from the elite population (new_population)
                parent1 = self.select_parents(new_population)
                parent2 = self.select_parents(new_population)

                # produce children from parents
                child1, child2 = self.reproduce(parent1, parent2)

                # Generate a random number, and only mutate if the number
                #  is below the mutation rate.
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)

                child1 = self.move_start_location_to_begining(child1)
                child2 = self.move_start_location_to_begining(child2)

                # calculate each child's fitness
                fitness1 = self.fitnessfcn(child1)
                fitness2 = self.fitnessfcn(child2)

                new_population.append([child1, fitness1])
                new_population.append([child2, fitness2])

                new_population.sort(key=operator.itemgetter(1), reverse=True)

            generations = generations + 1

            # Sort the new population in descending order
            new_population.sort(key=operator.itemgetter(1), reverse=True)

            # update current population
            self.population = new_population

            # plot travelling salesman route
            if generations % PLOT_DISPLAY_RATE == 0 or generations >= self.generations:
                print("Generation: %d" % generations, "Fitness: %f" % self.population[0][1])
                if generations == self.generations:
                    plotTSP(generations, self.population[0][0], self.points, self.population[0][1], True)
                else:
                    plotTSP(generations, self.population[0][0], self.points, self.population[0][1], False)

            self.values.append(self.population[0][1])

        self.print_result()


if __name__ == '__main__':

    ELITE_PERCENTAGE = 0.09  # Always pass top 9% individuals from current population to next generation
    PLOT_DISPLAY_RATE = 100  # Update route plot every 100 generation
    po_coordinates = "coordinates.txt"
    post_office_names = "post_offices.txt"
    start_office = "Johnson City Post Office, TN"
    locations = list(np.loadtxt(po_coordinates, delimiter=','))
    cities = [line.rstrip('\n') for line in open(post_office_names)]
    points = []
    paths = []
    start_office_index = [i for i in range(len(cities)) if cities[i] == start_office][0]

    loc_x = [x for x, y in locations]
    loc_y = [y for x, y in locations]
    loc_c = ["black" for _ in range(len(locations))]

    for i in range(0, len(loc_x)):
        points.append((loc_x[i], loc_y[i]))

    # origin, generations, points, population_size, mutation_rate
    origin = (locations[start_office_index][0], locations[start_office_index][1])

    # Parameters: 1. origin location,
    #             2. number of generations,
    #             3. locations as a list of tuples,
    #             4. list of city names,
    #             5. number of individuals in each generation,
    #             6. mutation rate

    gs = GeneticSearch(origin, 5500, points, cities, 120, 0.04)
    gs.run()

    x = input("Press Enter to Exit...")
    plt.close()
