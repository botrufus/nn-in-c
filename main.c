#include <stdlib.h>
#include <stdio.h>
#include "network.h"
#include <time.h>

int main(void) {
	srand(time(NULL));	
	
	size_t* sizes = malloc(3 * sizeof(size_t));
	sizes[0] = 1;
	sizes[1] = 3;
	sizes[2] = 1;
	network* network = init_network(2, sizes);

	size_t* n_inputs = malloc(100 * sizeof(size_t));
	size_t* n_expected = malloc(100 * sizeof(size_t));

	double** inputs = malloc(100 * sizeof(double));
	double** expected = malloc(100 * sizeof(double));
 
	for (size_t i = 0; i < 100; ++i) {
		n_inputs[i] = 1;
		n_expected[i] = 1;
		inputs[i] = malloc(sizeof(double));
		expected[i] = malloc(sizeof(double));

		inputs[i][0] = i + 1;
		expected[i][0] = (i+1) * 3;
	}

	data* data = init_data(100, n_inputs, inputs, n_expected, expected); 	
	
	double* res;

	for (size_t i = 0; i < 10000; ++i) {
		train(network, data, 0.0000001);
	}
	
	double* test = malloc(10 * sizeof(double));
	for (size_t i = 0; i < 10; ++i) {
		test[i] = i + 101;
	}
	
	for (size_t i = 0; i < 10; ++i) {
	 	res = calc_out(network, 1, test + i);
		printf("Squared of %f is: %f\n", test[i], *res);
		free(res);	
	}
	
	free(test);
	delete_network(network);
	free(sizes);
	free(inputs);
	free(expected);

	return EXIT_SUCCESS;
}
