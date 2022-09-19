#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "network.h"

network* init_network(size_t num_layers, size_t layer_sizes[num_layers + 1]) {
	if (num_layers == 0) {
		fprintf(stderr, "Network can not have 0 layers. \n");
		return 0;
	}

	network* network = malloc(sizeof(*network));
	if (!network) {
		fprintf(stderr, "Memory allocation failed for network.\n");
		return 0;
	}

	layer** layers = malloc(num_layers * sizeof(layer*));
	if (!layers) {
		fprintf(stderr, "Memory allocation failed for layers.\n");
		return 0;
	}
	
	layer* layer;
	for (size_t i = 0; i < num_layers; ++i) {
		layer = init_layer(layer_sizes[i], layer_sizes[i+1]);
		if (!layer) {
			fprintf(stderr, "Memory allocation failed for layers.\n");
			return 0;
		}
		layers[i] = layer;
	}
	network->layers = layers;	
	network->num_layers = num_layers;
	return network;	
}

void delete_network(network* network) {
	if (!network) {
		return;
	}
	for (size_t i = 0; i < network->num_layers; i++) {
		delete_layer(network->layers[i]); 		
	}
	free(network->layers);
	free(network);			
}

double* calc_out(network* network, size_t num_inputs, double inputs[num_inputs]) {
	if (network->layers[0]->num_in != num_inputs) {
		fprintf(stderr, "Number of inputs does not conform to first hidden layer.\n");
		return 0;
	}

	double* in = malloc(num_inputs * sizeof(double));
	if (!in) {
		fprintf(stderr, "Memory allocation failed for output copy array.\n");
		return 0;
	}
	memcpy(in, inputs, num_inputs * sizeof(double));

	double* out;
	layer* layer;
	for (size_t i = 0; i < network->num_layers; ++i) {
		layer = network->layers[i];
		out = calc_layer_out(layer, layer->num_in, in);
		free(in);
		in = out;
	}
	return out;
}

double loss(network* network, size_t num_inputs, double inputs[num_inputs], size_t num_expected, double expected[num_expected]) {
	double* out = calc_out(network, num_inputs, inputs);
	double num_out = network->layers[network->num_layers - 1]->num_out;

	if (num_out != num_expected) {
		fprintf(stderr, "Actual and expected number of outputs is different.\n");
		return 0;
	}
	
	double loss = 0;
	for (size_t i = 0; i < num_out; ++i) {
		loss += mse(out[i], expected[i]); 
	}
	
	free(out);
	
	return loss;
}

double avg_loss(network* network, data* data) {
	double total_loss = 0;

	for (size_t i = 0; i < data->n_data; ++i) {
		total_loss += loss(network, data->n_inputs[i], data->inputs[i], data->n_expected[i], data->expected[i]);	
	}
	return total_loss / data->n_data;
}

/*void update_grads(network* network, data* data) {
	calc_out(network, data->n_inputs, data->inputs);
	
	layer* out_layer = network->layers[num_layers - 1];
	double* vals = calc_layer_out_vals(out_layer, data->n_expected, data->expected);
	update_layer_grads(layer, vals);
	
	layer* layer;
	for (size_t i = network->num_layers - 2; i >= 0; --i) {
		layer = network->layers[i];
		vals = layer.calc_layer_hidden_vals(network->layers[i + 1], vals);
		update_layer_grads(layer, vals);
	}		
}*/

void apply_all_grads(network* network, double lr) {
	for (size_t i = 0; i < network->num_layers; ++i) {
		apply_grads(network->layers[i], lr);
	}
}

void train(network* network, data* data, double lr) {
	double h = 0.0001;
	double originalCost = avg_loss(network, data);

	layer* layer;
	double deltaCost;
	for (size_t i = 0; i < network->num_layers; ++i) {
		layer = network->layers[i];
		for (size_t z = 0; z < layer->num_in; ++z) {
			for (size_t y = 0; y < layer->num_out; ++y) {
				layer->weights[z][y] += h;
				deltaCost = avg_loss(network, data) - originalCost;
				layer->weights[z][y] -= h;
				layer->cgw[z][y] = deltaCost / h;
			}
		}
		
		for (size_t z = 0; z < layer->num_out; ++z) {
			layer->biases[z] += h;
			deltaCost = avg_loss(network, data) - originalCost;
			layer->biases[z] -= h;
			layer->cgb[z] = deltaCost / h;
		}
	}
	apply_all_grads(network, lr);	
} 
