import cv2
import numpy as np
import functools
import operator
from PIL import Image
import glob
from video_api import saving_address as address

'''
	Creating 1D vector. These are CHROMOSOMS.
'''
def create_chromosom(resize_img):

	chromosoms = []


	for i in range(len(resize_img)):

		for j in range(len(resize_img[i])):

			for k in range(len(resize_img[i][j])) :

				chromosoms.append(resize_img[i][j][k])


	#creating the 1D vector
	chromosoms_vector = np.array(chromosoms)

	#print(chromosoms_vector)


	#print(chromosoms_vector.shape)


	return chromosoms_vector


'''
	Recovering the image from chromosoms.
'''
def convert_chromotoframe(chromo_vector, image_shape):

	image = np.reshape(a = chromo_vector, newshape = image_shape)

	return image

'''
	Defining the initial population.
'''
def initial_Population(img_shape, n_individuals=10):

	init_population = np.empty(shape=(n_individuals,functools.reduce(operator.mul, img_shape)),dtype=np.uint8)

	for indv_num in range(n_individuals):

		# Randomly generating initial population chromosomes genes values.
		init_population[indv_num, :] = np.random.random(functools.reduce(operator.mul, img_shape))*256

	return init_population


'''
	Defining the fitness function.
'''
def fitness_function(target, source):

	gene_quality = np.mean(np.abs(target - source))

	gene_quality = np.sum(target) - gene_quality

	return gene_quality


'''
	calculating fitness of each individual from the population.
'''
def population_fitness(target_chromosom, initial_population):

	gene = np.zeros(initial_population.shape[0])

	for i in range(initial_population.shape[0]):

		gene[i] = fitness_function(target_chromosom, initial_population[i, :])


	return gene
	

'''
	Selecting the best fit parents.
'''
def parent_selection(population, fitness_scores, number_of_parents):

	parents = np.empty((number_of_parents, population.shape[1]), dtype=np.uint8)

	for parent in range(number_of_parents):

		parent_index = np.where(fitness_scores == np.max(fitness_scores))

		best_parent_index = parent_index[0][0]

		parents[parent, :] = population[best_parent_index, :]

		fitness_scores[best_parent_index] = -1

	return parents

	
'''
	Producing offspring(crossover).
	offspring will take 1st half of the gene from parent1,
	2nd half from next parent.
'''
def offSpring(parents, offspring_size):

	offspring = np.empty(offspring_size)
	
	crossover_point = np.uint8(offspring_size[1] / 2)


	for parent_num in range(2):

		p1_index = parent_num % parents.shape[0]

		p2_index = (parent_num + 1) % parents.shape[0]

		offspring[parent_num, 0:crossover_point] = parents[p1_index, 0:crossover_point]

		offspring[parent_num, crossover_point] = parents[p2_index, crossover_point]

	return offspring

'''
	Mutation to handle bad offspring.
'''
def mutation(offspring_crossover, num_mutations=1):

    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)

    for idx in range(offspring_crossover.shape[0]):

        gene_idx = mutations_counter - 1

        for mutation_num in range(num_mutations):

            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)

            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value

            gene_idx = gene_idx + mutations_counter

    return offspring_crossover



def model():

	play_ = cv2.VideoCapture(address)

	i = 0

	while play_.isOpened():

		ret, frame = play_.read()

		if ret == False:

			break

		cv2.imwrite(r'E:\prog\canada\frames\horse'+ str(i)+ '.jpg', frame)


		img = cv2.imread(r'E:\prog\canada\frames\horse'+ str(i)+ '.jpg')
		print(img)
		resize_img = cv2.resize(img, (120,120))
		
		print(resize_img)

		source_dimentions = resize_img.shape
		print("source image size:", source_dimentions)


		'''
			Resizing the target image.
		'''
		target_img = cv2.imread(r'E:\prog\canada\zebra.jpg')
		target_resizeimg = cv2.resize(target_img, (120,120))

		target_dimentions = target_resizeimg.shape
		print("target image size:",target_dimentions)


		'''
			Source and target chromosoms.
		'''
		source_chromosom = create_chromosom(resize_img)
		print(source_chromosom)

		target_chromosom = create_chromosom(target_resizeimg)
		print(target_chromosom)


		'''
			getting back the image from chromosoms.
		'''
		image = convert_chromotoframe(source_chromosom, source_dimentions)

		print(image)


		'''
			Creating initial population.
		'''
		population = initial_Population(source_dimentions)

		print(population)


		'''
			getting the fitness of parent.
		'''
		print("Fitness value of genes: ",'\n', fitness_function(target_chromosom, source_chromosom))


		'''
			calculate population fitness.
		'''
		fitness_scores = population_fitness(target_chromosom, population)
		print("Population individual fitness score:", '\n', fitness_scores)


		'''
			selecting parent.
		'''
		parents = parent_selection(population, fitness_scores, 2)
		print("Parents are:", '\n', parents)


		'''
			new offspring.
		'''
		off_spring = offSpring(parents, parents.shape)
		print("offspring:", '\n', off_spring)


		'''
			Mutation result.
		'''
		muted_offspring = mutation(off_spring)

		'''
			saving the genetic algo generated frames.
		'''
		genetic_frames = 'E:\prog\canada\genetic_algo_frames\gn'+ str(i) + '.jpg'
		new_img = Image.fromarray(muted_offspring, 'RGB')
		new_img.save(genetic_frames)
		#new_img.show()


		i = i + 1

	play_.release()

	cv2.destroyAllWindows()

if __name__ == '__main__':

	model()
