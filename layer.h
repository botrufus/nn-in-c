#include <stdlib.h>
#include <stdio.h>
#include "lossfunc.h"
#include "actfunc.h"

typedef struct {
	size_t num_in;
	size_t num_out;
	double** cgw;
	double* cgb;
	double** weights;
	double* biases;		
} layer;

layer* init_layer(size_t num_in, size_t num_out);

void delete_layer(layer* layer);

double* calc_layer_out(layer* layer, size_t num_inputs, double inputs[num_inputs]);

void apply_grads(layer* layer, double lr);
//void set_gradients(layer* layer, size_t n, double nodeValues[n]);
