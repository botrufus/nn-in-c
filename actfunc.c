#include "actfunc.h"
#include "math.h"

double act_step(double weighted_input) {
	return weighted_input < 0 ? 0 : 1;
}

double act_sigmoid(double weighted_input) {
	return 1 / (1 + exp(-weighted_input));
}

double act_sigmoid_deriv(double weighted_input) {
	double x = act_sigmoid(weighted_input);
	return x * (1 - x);
}

// TODO: Add additional activation functions
