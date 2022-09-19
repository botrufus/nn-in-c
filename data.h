#include <stdlib.h>

typedef struct {
	size_t n_data;
	size_t* n_inputs;
	double** inputs;
	size_t* n_expected;
	double** expected;
} data;

data* init_data(size_t n_data, size_t* n_inputs, double** inputs, size_t* n_expected, double** expected);

void delete_data(data* data);	
