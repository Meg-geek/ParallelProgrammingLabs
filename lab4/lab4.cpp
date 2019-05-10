#include <iostream>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<ctime>
#include<sys/time.h>
#include <mpi.h>

#define Ni 100
#define Nj 100
#define Nk 100
#define a 100

double X = 2.0;
double Y = 2.0;
double Z = 2.0;

double constant; //основной множитель для конечной формулы
double initApproxInner = 0;
double E = 0.000000001;

int I;
int J;
int K;

/* F[0] и F[1] данные текущей и предыдущей итерации*/
double *(F[2]);
double *(buffer[2]);
double hx, hy, hz;

MPI_Request sendRequest[2] = {};
MPI_Request recRequest[2] = {};

double startBorderFunc(double x, double y, double z) {
	double res;
	res = x + y + z;
	return res;
}

/* Функция задания правой части уравнения */
double Ro(double x, double y, double z) {
	double d;
	d = 6 - a * startBorderFunc(x, y, z);
	return d;
}

void initialize(int *perProcess, int *offsets, int processRank) {
	for (int i = 0, startLine = offsets[processRank]; i <= perProcess[processRank] - 1; i++, startLine++) {
		for (int j = 0; j <= Nj; j++) {
			for (int k = 0; k <= Nk; k++) {
				if ((startLine != 0) && (j != 0) && (k != 0) && (startLine != Ni) && (j != Nj) && (k != Nk)) {
					F[0][i*J*K + j * K + k] = initApproxInner;
					F[1][i*J*K + j * K + k] = initApproxInner;
				}
				else {
					F[0][i*J*K + j * K + k] = startBorderFunc(startLine * hx, j * hy, k * hz);
					F[1][i*J*K + j * K + k] = startBorderFunc(startLine * hx, j * hy, k * hz);
				}
			}
		}
	}

}

void calcEdges(int processRank, int processesCount, int *perProcesses, double hxPow, double hyPow, double hzPow, int I0, int I1, int *offsets, int &flag) {
	for (int j = 1; j < Nj; ++j) {
		for (int k = 1; k < Nk; ++k) {
			if (processRank != 0) {
				int i = 0;
				double Fi, Fj, Fk;
				Fi = (F[I0][(i + 1)*J*K + j * K + k] + buffer[0][j*K + k]) / hxPow;
				Fj = (F[I0][i * J * K + (j + 1) * K + k] + F[I0][i * J * K + (j - 1) * K + k]) / hyPow;
				Fk = (F[I0][i * J * K + j * K + (k + 1)] + F[I0][i * J * K + j * K + (k - 1)]) / hzPow;
				F[I1][i*J*K + j * K + k] = (Fi + Fj + Fk - Ro((i + offsets[processRank]) * hx, j * hy, k * hz)) / constant;
				if (fabs(F[I1][i*J*K + j * K + k] - F[I0][i*J*K + j * K + k]) < E) {
					flag = 1;
				}
			}

			if (processRank != processesCount - 1) {
				int i = perProcesses[processRank] - 1;
				double Fi, Fj, Fk;
				Fi = (buffer[1][j*K + k] + F[I0][(i - 1)*J*K + j * K + k]) / hxPow;
				Fj = (F[I0][i * J * K + (j + 1) * K + k] + F[I0][i * J * K + (j - 1) * K + k]) / hyPow;
				Fk = (F[I0][i * J * K + j * K + (k + 1)] + F[I0][i * J * K + j * K + (k - 1)]) / hzPow;
				F[I1][i*J*K + j * K + k] = (Fi + Fj + Fk - Ro((i + offsets[processRank]) * hx, j * hy, k * hz)) / constant;
				if (fabs(F[I1][i*J*K + j * K + k] - F[I0][i*J*K + j * K + k]) < E) {
					flag = 1;
				}
			}

		}
	}
}

void sendData(int processRank, int processesCount, int I0, int *perProcesses) {
	if (processRank != 0) {
		MPI_Isend(&(F[I0][0]), K*J, MPI_DOUBLE, processRank - 1, 0, MPI_COMM_WORLD, &sendRequest[0]); //низ
		MPI_Irecv(buffer[0], K*J, MPI_DOUBLE, processRank - 1, 1, MPI_COMM_WORLD, &recRequest[1]);
	}
	if (processRank != processesCount - 1) { 
		MPI_Isend(&(F[I0][(perProcesses[processRank] - 1)*J*K]), K*J, MPI_DOUBLE, processRank + 1, 1, MPI_COMM_WORLD, &sendRequest[1]); 
		MPI_Irecv(buffer[1], K*J, MPI_DOUBLE, processRank + 1, 0, MPI_COMM_WORLD, &recRequest[0]);
	}
}

void calcCenter(int processRank, int *perProcesses, int *offsets,double hxPow, double hyPow, double hzPow, int I0, int I1, int &flag) {
	for (int i = 1; i < perProcesses[processRank] - 1; ++i) {
		for (int j = 1; j < Nj; ++j) {
			for (int k = 1; k < Nk; ++k) {
				double Fi, Fj, Fk;
				Fi = (F[I0][(i + 1)*J*K + j * K + k] + F[I0][(i - 1)*J*K + j * K + k]) / hxPow;
				Fj = (F[I0][i * J * K + (j + 1) * K + k] + F[I0][i * J * K + (j - 1) * K + k]) / hyPow;
				Fk = (F[I0][i * J * K + j * K + (k + 1)] + F[I0][i * J * K + j * K + (k - 1)]) / hzPow;

				F[I1][i*J*K + j * K + k] = (Fi + Fj + Fk - Ro((i + offsets[processRank]) * hx, j * hy, k * hz)) / constant;

				if (fabs(F[I1][i*J*K + j * K + k] - F[I0][i*J*K + j * K + k]) < E) {
					flag = 1;
				}
			}
		}
	}
}

void waitForData(int processRank, int processesCount) {
	if (processRank != 0) {
		MPI_Wait(&recRequest[1], MPI_STATUS_IGNORE);
		MPI_Wait(&sendRequest[0], MPI_STATUS_IGNORE);
	}
	if (processRank != processesCount - 1) {
		MPI_Wait(&recRequest[0], MPI_STATUS_IGNORE);
		MPI_Wait(&sendRequest[1], MPI_STATUS_IGNORE);
	}
}

void findMaxDiff(int processRank, int *perProcesses, int I0, int I1) {
	double max = 0.0;
	double F1;
	for (int i = 1; i < perProcesses[processRank] - 2; i++) {
		for (int j = 1; j < Nj; j++) {
			for (int k = 1; k < Nk; k++) {
				F1 = fabs(F[I1][i*J*K + j * K + k] - F[I0][i*J*K + j * K + k]);
				if (F1 > max) {
					max = F1;
				}
			}
		}
	}

	double tmpMax = 0;
	MPI_Allreduce(&max, &tmpMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (processRank == 0) {
		max = tmpMax;
		std::cout << "Max difference = " << max << std::endl;
	}
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int processRank, processesCount;
	MPI_Comm_size(MPI_COMM_WORLD, &processesCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

	if (processRank == 0) {
		std::cout << "Processes count: " << processesCount << std::endl;
	}
	int *perProcesses, *offsets;
	perProcesses = new int[processesCount];
	offsets = new int[processesCount];
	int height = Nk + 1;
	int tmp = processesCount - (height % processesCount);
	int currentLine = 0;
	for (int i = 0; i < processesCount; i++) {
		offsets[i] = currentLine;
		perProcesses[i] = i < tmp ? (height / processesCount) : (height / processesCount + 1);
		currentLine += perProcesses[i];
	}

	I = perProcesses[processRank];
	J = (Nj + 1);
	K = (Nk + 1);

	F[0] = new double[I * J * K];
	F[1] = new double[I * J * K];

	buffer[0] = new double[K*J];
	buffer[1] = new double[K*J];

	hx = X / Ni;
	hy = Y / Nj;
	hz = Z / Nk;
	
	double hxPow = hx * hx;
	double hyPow = hy * hy;
	double hzPow = hz * hz;

	constant = 2 / hxPow + 2 / hyPow + 2 / hzPow + a;

	initialize(perProcesses, offsets, processRank);

	double startTime = MPI_Wtime();
	
	int flag = 0, I0 = 1, I1 = 0;
	int iterations = 0;
	do {
		I0 = 1 - I0;
		I1 = 1 - I1;
		int tmpFlag;
		
		sendData(processRank, processesCount, I0, perProcesses);
		if (processRank == 0) {
			iterations++;
		}
		
		calcCenter(processRank, perProcesses, offsets, hxPow, hyPow, hzPow, I0, I1, flag);
		
		waitForData(processRank, processesCount);
		
		calcEdges(processRank, processesCount, perProcesses, hxPow, hyPow, hzPow, I0, I1, offsets, flag);
		
		MPI_Allreduce(&flag, &tmpFlag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		flag = tmpFlag;
	} while (flag == 0);

	double finishTime = MPI_Wtime();

	if (processRank == 0) {
		std::cout<<"Time: "<<finishTime - startTime<<std::endl<<"iterations: "<<iterations<<std::endl;
	}

	findMaxDiff(processRank, perProcesses, I0, I1);

	delete[] buffer[0];
	delete[] buffer[1];
	delete[] F[0];
	delete[] F[1];
	delete[] offsets;
	delete[] perProcesses;

	MPI_Finalize();
	return 0;
}