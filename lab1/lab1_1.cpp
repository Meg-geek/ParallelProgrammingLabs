#include <iostream>
#include <cmath>

double norm_function(double *vector, int N) {
	double sum = 0;
	for (int i = 0; i < N; i++)
		sum += vector[i] * vector[i];
	return sqrt(sum);
}

void make_model_task(double *matrixA, double *b, double *x, int N) {
	for (int i = 0; i < N; i++) {
		x[i] = 0;
		b[i] = N + 1;
		for (int j = 0; j < N; j++)
			matrixA[i*N + j] = (i == j) ? 2 : 1 ;
	}
}

double *mul_matrix_vector (double *matrix, double *vector, int N) {
	double *result = new double[N];
	for (int i = 0; i < N; i++) result[i] = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			result[i] += matrix[i * N + j] * vector[j];
		}
	}
	return result;
}

double *vectors_sub(double *vector1, double *vector2, int N) {
	double *result = new double[N];
	for (int i = 0; i < N; i++) {
		result[i] = vector1[i] - vector2[i];
	}
	return result;
}

void mul_vector_scal(double scal, double *vector, int N) {
	for (int i = 0; i < N; i++) {
		vector[i] = vector[i] * scal;
	}
}

int main(int argc, char **argv)
{
	int N = 500;
	double E = 10e-5;
	double *matrixA = new double[N*N];
	double *beginX = new double[N];
	double *b = new double[N];
	//задали начальное условие
	make_model_task(matrixA, b, beginX, N);
	double T = 10e-4;
	double norm_b = norm_function(b, N);
	while (1) {
		double *mul = mul_matrix_vector(matrixA, beginX, N);
		double *temp_sub = vectors_sub(mul, b, N);

		delete[] mul;
		double norm_A = norm_function(temp_sub, N);
		
		if (( norm_A / norm_b ) < E) {
			delete[] temp_sub;
			break;
		}
		mul_vector_scal(T, temp_sub, N);

		double *newX = vectors_sub(beginX, temp_sub, N);

		delete[] beginX;
		beginX = newX;
		delete[] temp_sub;
	}
	
	for (int i = 0; i < N; i++)
		std::cout << beginX[i] << std::endl;
	
	delete[] beginX;
	delete[] b;
	delete[] matrixA;
	return 0;
}

