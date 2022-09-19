#include <stdlib.h>
#include "layer.h"
#include "data.h"

typedef struct {
	layer** layers;
	size_t num_layers;
} network;

network* init_network(size_t num_layers, size_t layer_sizes[num_layers + 1]);

void delete_network(network* network);

double* calc_out(network* network, size_t num_inputs, double inputs[num_inputs]);

double loss(network* network, size_t num_inputs, double inputs[num_inputs], size_t num_expected, double expected[num_expected]); 

double avg_loss(network* network, data* data);

void apply_all_grads(network* network, double lr);

void train(network* network, data* data, double lr);

//void update_grads(network* network, data* data);
