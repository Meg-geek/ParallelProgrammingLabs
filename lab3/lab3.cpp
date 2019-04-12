#include<cstdio>
#include <iostream>
#include<cstdlib>
#include<mpi.h>
#include<ctime>

#define NNODES 2
#define N1 2000
#define N2 2000
#define N3 2000
#define A(i, j) A[N2*i+j]
#define B(i, j) B[N3*i+j]
#define C(i, j) C[N3*i+j]
#define AA(i, j) AA[matrixSizes[1]*i+j]
#define BB(i, j) BB[nn[1]*i+j]
#define CC(i, j) CC[nn[1]*i+j]


void createTypes(int *matrixSizes, int *nn, MPI_Datatype *typeb, MPI_Datatype *typec) {
	MPI_Datatype types;

	MPI_Type_vector(matrixSizes[1], nn[1], matrixSizes[2], MPI_DOUBLE, &types); //��������� B


	MPI_Aint sizeofdouble; //MPI_Aint is a portable C data type that can hold memory addresses and it could be larger than the usual int
	MPI_Type_extent(MPI_DOUBLE, &sizeofdouble); //������ ������ � ������
	MPI_Type_create_resized(types, 0, sizeofdouble * nn[1], typeb); //������������ ��� ����� �������� ������� � ���������
	MPI_Type_commit(typeb); //������������ ����� ����������� ��� 
	//������ ����� ����������� ����� ����������� ��� ����� ������������ � ���������������� ������������� � ��� ��������������� ������ �����


	MPI_Type_vector(nn[0], nn[1], matrixSizes[2], MPI_DOUBLE, &types); //��������� ��� �
	MPI_Type_create_resized(types, 0, sizeofdouble * nn[1], typec);
	MPI_Type_commit(typec);
}


void calculate(int *matrixSizes, double *A, double *B, double *C, int *dims, MPI_Comm comm) {
	int *recvcountc, *dispc, *sendcountsB, *dispb;
	MPI_Datatype typeb, typec;
	MPI_Comm newComm;


	MPI_Comm_dup(comm, &newComm); //����������� � ����� ������������ 
	MPI_Bcast(matrixSizes, 3, MPI_INT, 0, newComm); //�������� ���� ��������� ������
	MPI_Bcast(dims, 2, MPI_INT, 0, newComm);


	int periods[2] = { 0 }; //���������� ������ ������� ndims ��� ������� ��������� ������� (true - �������������, false - ���������������)
	MPI_Comm comm2D;
	MPI_Cart_create(newComm, NNODES, dims, periods, 0, &comm2D); //������� ������������ � ���������� ����������


	int processCoords[2];
	int processRank;
	MPI_Comm_rank(comm2D, &processRank);
	MPI_Cart_coords(comm2D, processRank, NNODES, processCoords); //������ ���� ����������


	MPI_Comm comm_1D[2];
	int remain_dims[2]; //���������� ������ ������� ndims, �����������, ������ �� i-e ��������� � ����� ����������

	for (int i = 0; i < 2; i++) { //������ ������� � ������� ���������� ���������
		for (int j = 0; j < 2; j++) {
			remain_dims[j] = (i == j); // TF FT
		}
		MPI_Cart_sub(comm2D, remain_dims, &comm_1D[i]); //��������� ������������ �� ��� ���������� ������������� � ����������� ���������
	}


	int nn[2];
	nn[0] = matrixSizes[0] / dims[0]; //���������� ����� � �������  N1/dims[0]
	nn[1] = matrixSizes[2] / dims[1]; //������� � ������� N3 / dims[1]


	double *AA, *BB, *CC;
	AA = new double[nn[0] * matrixSizes[1]]; //����� * N2 �������������� ������ �
	BB = new double[matrixSizes[1] * nn[1]];	//������������ ������ �
	CC = new double[nn[0] * nn[1]]; //���������� �

	//��� �������� ��������
	if (processRank == 0) {
		createTypes(matrixSizes, nn, &typeb, &typec); //������� ���� � B � �

		dispb = new int[dims[1]]; //���������� �������
		sendcountsB = new int[dims[1]];
		dispc = new int[dims[0] * dims[1]];
		recvcountc = new int[dims[0] * dims[1]];
		for (int j = 0; j < dims[1]; j++) {
			dispb[j] = j;
			sendcountsB[j] = 1;
		}

		for (int i = 0; i < dims[0]; i++) {
			for (int j = 0; j < dims[1]; j++) {
				dispc[i * dims[1] + j] = (i * dims[1] * nn[0] + j); 
				recvcountc[i * dims[1] + j] = 1;
			}
		}
	}

	if (processCoords[1] == 0) { //����� �� �y ����� 0
		MPI_Scatter(A, nn[0] * matrixSizes[1], MPI_DOUBLE, AA, nn[0] * matrixSizes[1], MPI_DOUBLE, 0, comm_1D[0]); //������� � � ��������� �� �������, 1 ���
	}


	if (processCoords[0] == 0) {
		MPI_Scatterv(B, sendcountsB, dispb, typeb, BB, matrixSizes[1] * nn[1], MPI_DOUBLE, 0, comm_1D[1]); //������� � � ��������� �� �������, 2 ���
	}


	MPI_Bcast(AA, nn[0] * matrixSizes[1], MPI_DOUBLE, 0, comm_1D[1]); //�� ������ ���, ����� ��������� ����� ������, 3 ���


	MPI_Bcast(BB, matrixSizes[1] * nn[1], MPI_DOUBLE, 0, comm_1D[0]);//4 ���

	for (int i = 0; i < nn[0]; i++) {
		for (int j = 0; j < nn[1]; j++) {
			for (int k = 0; k < matrixSizes[1]; k++) {
				CC(i, j) = CC(i, j) + AA(i, k) * BB(k, j); //5 ��� (������ ������� ����������� ���� ����������)
			}
		}
	}

	MPI_Gatherv(CC, nn[0] * nn[1], MPI_DOUBLE, C, recvcountc, dispc, typec, 0, comm2D);

	delete[] AA;
	delete[] BB;
	delete[] CC;
	MPI_Comm_free(&newComm);
	MPI_Comm_free(&comm2D);
	for (int i = 0; i < 2; i++) {
		MPI_Comm_free(&comm_1D[i]);
	}


	if (processRank == 0) {
		delete[] recvcountc;
		delete[] dispc;
		delete[] sendcountsB;
		delete[] dispb;
		MPI_Type_free(&typeb);
		MPI_Type_free(&typec);
	}
}


bool checkResult(double *C) {
	for (int i = 0; i < N1; ++i) {
		for (int j = 0; j < N3; ++j) {
			if (C(i, j) != N2) {
				return false;
			}
		}
	}
	return true;
}


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int processesAmount, processRank;
	MPI_Comm_size(MPI_COMM_WORLD, &processesAmount);
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

	int dims[NNODES] = { 0 }; //����� ��������� ����� ������� ���������
	//������������ �������, ������ dims, �������� NNODES  �� ������ ������������ dims[i]
	//����������� ������ �� ���������� dims[i], ��� ������� ����� ���������� � ��������� ���� ��������� �������� 0. 
	//������� ��������� ������� ����������� ����������� ������������� ��������� ����� �����������, ���������� �� �� ��������, 
	//�.�. ��� 12-�� ��������� ��� �������� ���������� ����� 4 � 3 � 1
	MPI_Dims_create(processesAmount, NNODES, dims); 

	int matrixSizes[3];
	double *A;
	double *B;
	double *C;
	//����� ������ ���� � ������� ��������
	if (processRank == 0) {
		for (int i = 0; i < NNODES; i++) {
			std::cout << dims[i] << " ";
		}
		std::cout << std::endl;
		matrixSizes[0] = N1;
		matrixSizes[1] = N2;
		matrixSizes[2] = N3;

		A = new double[N1 * N2];
		B = new double[N2 * N3];
		C = new double[N1 * N3];

		for (int i = 0; i < N1; ++i) {
			for (int j = 0; j < N2; ++j) {
				A(i, j) = 1;
			}
		}


		for (int i = 0; i < N2; ++i) {
			for (int j = 0; j < N3; ++j) {
				B(i, j) = 1;
			}
		}
	}

	double startTime = MPI_Wtime();
	calculate(matrixSizes, A, B, C, dims, MPI_COMM_WORLD);
	double finishTime = MPI_Wtime();

	if (processRank == 0) {
		std::cout << "Time :" << finishTime - startTime << std::endl;
		std::cout << "Result correct: " << (checkResult(C) ? "yes" : "no") << std::endl;
		delete[] A;
		delete[] B;
		delete[] C;
	}

	MPI_Finalize();
	return 0;
}
