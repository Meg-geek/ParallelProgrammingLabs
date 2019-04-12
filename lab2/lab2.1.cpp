 #include <iostream>
#include <cmath>
#include <omp.h>

double norm_function(double *vector, int N) {
	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < N; i++)
		sum += vector[i] * vector[i];
	return sqrt(sum);
}

void make_model_task(double *matrixA, double *b, double *x, int N) {
#pragma omp parallel for  collapse(2)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			matrixA[i*N + j] = (i == j) ? 2 : 1;
	}

#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		x[i] = i;
		b[i] = N + 1;
	}	
}

void  mul_matrix_vector (double *matrix, double *vector, int N, double *result) {
#pragma omp parallel for 
	for (int i = 0; i < N; i++) 
		result[i] = 0;
	#pragma omp parallel for shared(result) collapse(2)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			result[i]+= matrix[i * N + j] * vector[j];
		}
	}
}

void  vectors_sub(double *vector1, double *vector2, int N, double *result) {
#pragma omp parallel for 
	for (int i = 0; i < N; i++) {
		result[i] = vector1[i] - vector2[i];
	}
}

void mul_vector_scal(double scal, double *vector, int N) {
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		vector[i] = vector[i] * scal;
	}
}

bool checkE(double norm_A, double norm_b) {
	double E = 10e-12;
	if (norm_b == 0) {
		std::cout << "Norma b null";
		return false;
	}
	return ((norm_A / norm_b) >= E);
}

int main(int argc, char **argv)
{
	int N = 800;
	double E = 10e-12;
	double T = 10e-4;
	double *matrixA = new double[N*N];
	double *temp = new double[N];
	double *X = new double[N];
	double *b = new double[N];

	double beginTime = omp_get_wtime();
	make_model_task(matrixA, b, X, N);
	
	double norm_b = norm_function(b, N);
	double norm_A = 0;
	int iter = 0;
	do {
		iter++;
		mul_matrix_vector(matrixA, X, N, temp);
		vectors_sub(temp, b, N, temp); //Ax-b
		norm_A = norm_function(temp, N);
		mul_vector_scal(T, temp, N);
		vectors_sub(X, temp, N, X);
	} while (checkE(norm_A, norm_b));
	
	double endTime = omp_get_wtime();
	std::cout << "Time is: " << endTime - beginTime <<" iterations: " <<iter<<std::endl;
	bool correctAnswer = true;
	for (int i = 0; i < N; i++) {
		if (fabs(X[i] - 1) >= 1) {
			std::cout << "i = " << i << " X[i] = " << X[i] << "X[i] - 1 = "<<X[i] - 1<< " E = " << E << " X[i]-1= " << fabs(fabs(X[i]) - 1) << "\nerror" << std::endl;
			correctAnswer = false;
			break;
		}
	}
	std::cout << "Answer correct: " << (correctAnswer == true ? "True " : "False ") << std::endl;

#pragma omp parallel
    std::cout<<"Thread "<<std::endl;

	delete[] X;
	delete[] b;
	delete[] matrixA;
	delete[] temp;
	return 0;
}

