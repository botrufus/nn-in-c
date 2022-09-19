#include "lossfunc.h"

double mse(double predicted, double expected) {
	double error = predicted - expected;
	return error * error;
}

double mse_deriv(double predicted, double expected) {
	return predicted - expected;
}
