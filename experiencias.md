# Experiencias

Todas as experiencias devem ser corridas 5 vezes com seeds diferentes!!! Guardar as seeds!

250 gerações sempre

## Random search

1.

    population_size = 50

## Genetic algorithm

1.

    population_size = 50
    mutation_strenght = 0.02
    mutation_rate = 0.05
    k_points = 2
    tournament_size = 5
    elite_size = 2

2.

    population_size = 50
    mutation_strenght = 0.05
    mutation_rate = 0.1
    k_points = 2
    tournament_size = 4
    elite_size = 3

## CMA-ES

1.

    population_size = 50
    sigma = 1.3

2.

    population_size = 50
    sigma = 0.5

## Diff evolution

1.

    population_size = 50
    base_vector_type = 'random'
    nr_diff_vectors = 1
    scale_factor = 'random'
    scale_max = 1
    scale_min = 0.5

2.

    population_size = 50
    base_vector_type = 'best'
    nr_diff_vectors = 2
    scale_factor = 'linear'
    scale_max = 1
    scale_min = 0.5
