#include "data.h"

data* init_data(size_t n_data, size_t* n_inputs, double** inputs, size_t* n_expected, double** expected) {
	data* data = malloc(sizeof(*data));
	
	data->n_data = n_data;
	data->n_inputs = n_inputs;
	data->inputs = inputs;
	data->n_expected = n_expected;
	data->expected = expected;

	return data;
}

void delete_data(data* data) {
	for (size_t i = 0; i < data->n_data; ++i) {
		free(data->inputs[i]);
		free(data->expected[i]);
	}
	free(data->n_inputs);
	free(data->n_expected);
	free(data);
}
