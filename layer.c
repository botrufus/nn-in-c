#include "layer.h"
#include <math.h>

double rndm_double(int lower, int upper) {
	return (double) rand() / RAND_MAX * (upper + 1) + lower;
}

void init_rndm(size_t n, double* arr) {
	for (size_t i = 0; i < n; ++i) {
		arr[i] = rndm_double(0, 2);
	}
} 

layer* init_layer(size_t num_in, size_t num_out) {
	layer* layer = malloc(sizeof(*layer));
	if (!layer) {
		fprintf(stderr, "Memory allocation failed for layer.\n");
		return 0;
	}
	layer->num_in = num_in;
	layer->num_out = num_out;	

	double* biases = calloc(num_out, sizeof(double));
	if (!biases) {
		fprintf(stderr, "Memory allocation failed for biases.\n");
		return 0;
	}
	layer->biases = biases;
	
	double* cgb = malloc(num_out * sizeof(double*));
	if (!cgb) {
		fprintf(stderr, "Memory allocation failed for cgb.\n");
		return 0;
	}
	layer->cgb = cgb;

	double** cgw = malloc(num_in * sizeof(double*));
	if (!cgw) {
		fprintf(stderr, "Memory allocation failed for cgw.\n");
		return 0;
	}	

	double** weights = malloc(num_in * sizeof(double*));
	if (!weights) {
		fprintf(stderr, "Memory allocation failed for weights.\n");
		return 0;
	}
	
	double* warr;
	double* cg;
	for (size_t i = 0; i < num_in; ++i) {
		warr = malloc(num_out * sizeof(double));
		if (!warr) {
			fprintf(stderr, 
				"Memory allocation failed for weight of node.\n");
			return 0;
		}
		init_rndm(num_out, warr);
		weights[i] = warr;
	
		cg = malloc(num_out * sizeof(double));
		if (!cg) {
			fprintf(stderr, "Memory allocation failed for cg of cgw.\n");
			return 0;
		}
		cgw[i] = cg;
	}
	layer->cgw = cgw;
	layer->weights = weights;
	
	return layer;
}

void delete_layer(layer* layer) {
	if (!layer) {
		return;
	}
	
	free(layer->biases);
	free(layer->cgb);
	
	if (layer->weights) {
		for (size_t i = 0; i < layer->num_in; ++i) {
			free(layer->weights[i]);
		}
		free(layer->weights);
	}
	
	if (layer->cgw) {
		for (size_t i = 0; i < layer->num_in; ++i) {
			free(layer->cgw[i]);
		}
		free(layer->cgw);
	}

	free(layer);
}

double* calc_layer_out(layer* layer, size_t num_inputs, double inputs[num_inputs]) {
	if (layer->num_in != num_inputs) {
		fprintf(stderr, "Number of inputs doesnt conform to Layer.\n");
		return 0;
	}
	
	double* out = malloc(layer->num_out * sizeof(double));
	if (!out) {
		fprintf(stderr, "Memory allocation failed for output.\n");
		return 0; 	
	}

	double weightedInput;
	for (size_t i = 0; i < layer->num_out; ++i) {
		weightedInput = layer->biases[i];
		for (size_t z = 0; z < layer->num_in; ++z) {
			weightedInput += inputs[z] * layer->weights[z][i];
		}
		out[i] = fmax(0, weightedInput);
	}
	return out;
}

void apply_grads(layer* layer, double lr) {
	for (size_t i = 0; i < layer->num_out; ++i) {
		for (size_t z = 0; z < layer->num_in; ++z) {
			layer->weights[z][i] -= layer->cgw[z][i] * lr;
		}
		layer->biases[i] -= layer->cgb[i] * lr;
	}
}

/*double* calc_layer_out_vals(layer* layer, size_t n, double expected[n]) {
	double* vals = malloc(n * sizeof(double));
	
	for (size_t i = 0; i < n; ++i) {
		double cd = mse_deriv();
		double actd = act_sigmoid_deriv(layer->weights)
		vals[i] = actd * cd;
	}

	return vals;
}*/
