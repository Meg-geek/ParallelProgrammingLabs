#include <iostream>
#include <cmath>
#include <mpi.h>

int main(int argc, char **argv)
{
	int N = 500;
	double E = 10e-5;
	double T = 10e-4;

	MPI_Init(&argc, &argv);
	int numbStreams = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &numbStreams);//number of streams
	int curStream = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &curStream);//current stream

	int *linesPerStream = new int[numbStreams];//how many lines in one stream
	int borderLines = numbStreams - (N % numbStreams);
	int firstGroup = N / numbStreams;
	int secondGroup = firstGroup + 1;

	int beginLine = 0; //where to start

	for (int i = 0; i < numbStreams; i++) {
		if (i < borderLines) {
			linesPerStream[i] = firstGroup;
		}
		else
			linesPerStream[i] = secondGroup;
		if (i < curStream)
			beginLine += linesPerStream[i];
	}
	//разбили матрицу ј по строкам по потокам

	double *matrixA = new double[linesPerStream[curStream] * N];
	for (int i = 0; i < linesPerStream[curStream]; i++) {
		for (int j = 0; j < N; j++) {
			if (beginLine + i == j) //if diag
			{
				matrixA[i*N + j] = 2;
			}
			else
				matrixA[i*N + j] = 1;
		}
	}
	//заполнили матрицу ј

	double *b = new double[linesPerStream[curStream]];
	for (int i = 0; i < linesPerStream[curStream]; i++) {
		b[i] = N + 1;
	}
	//заполнили b

	double *X = new double[N];
	double *partsOfX = new double[numbStreams];
	int *displs = new int[numbStreams];
	displs[0] = 0;
	for (int i = 1; i < numbStreams; i++) {
		displs[i] = displs[i - 1] + linesPerStream[i - 1];
	}

	double norm_b = 0;
	double start = 0;
	if (curStream == 0) {
		start = MPI_Wtime(); //замерили врем€ начала
	}

	for (int i = 0; i < linesPerStream[curStream]; i++) {
		norm_b += b[i] * b[i];
	} //посчитали часть нормы, котора€ используетс€ в потоке
	double normB = 0;
	MPI_Allreduce(&norm_b, &normB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //сложили со всех частей
	//int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
	/*¬ исходных текстах примеров дл€ MPI часто используетс€ идентификатор\
	MPI_COMM_WORLD. Ёто название коммуникатора, создаваемого библиотекой автоматически.\
	ќн описывает стартовую область св€зи, объедин€ющую все процессы приложени€.*/
	norm_b = sqrt(normB);
	//посчитали норму b

	double norm_A = 0;
	double *streamTempRes = new double[linesPerStream[curStream]];
	while (1) {
		double temp; //i координата Ax - b
		for (int i = 0; i < linesPerStream[curStream]; i++) {
			temp = 0;
			for (int j = 0; j < N; j++) {
				temp += matrixA[i * N + j] * X[j];//Ax
			}
			temp = temp - b[i];//Ax-b
			streamTempRes[i] = X[i + beginLine] - temp * T;//x-t(Ax-b)
			norm_A += temp * temp;
		}
		double normA = 0;
		MPI_Allreduce(&norm_A, &normA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //считаем норму ј
		MPI_Allgatherv(streamTempRes, linesPerStream[curStream], MPI_DOUBLE, X, linesPerStream, displs,\
			MPI_DOUBLE, MPI_COMM_WORLD);
		//из streamTempRes число строк дл€ данного потока формата double
		//записываем в ’, с каждого процесса по linesPerStream[i] строк, смещение отн-но ’
		//какой индекс икс какому потоку
		/*
		int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
				   void *recvbuf, const int *recvcounts, const int *displs,
				   MPI_Datatype recvtype, MPI_Comm comm)
				   sendbuf
						starting address of send buffer (choice)
					sendcount
						number of elements in send buffer (integer)
					sendtype
						data type of send buffer elements (handle)
					recvcounts
							integer array (of length group size) containing the number of elements that are to be received from each process
					displs
							integer array (of length group size). Entry i specifies the displacement (relative to recvbuf) at which to place the incoming data from process i
					recvtype
							data type of receive buffer elements (handle)
					comm
						communicator (handle)
		*/
		norm_A = sqrt(normA);
		if (norm_A / norm_b < E) {
			break;
		}
		norm_A = 0;
	}

	if (curStream == 0) {
		double finish = MPI_Wtime();
		std::cout << "time: " << (finish - start) << std::endl;
		bool correctAnswer = true;
		for (int i = 0; i < N; i++) {
			if (fabs(fabs(X[i]) - 1) >= E) {
				std::cout << i << " " << X[i] << " " << E << " " << fabs(fabs(X[i]) - 1) << " error"<<std::endl;
				correctAnswer = false;
				break;
			}
		}
		std::cout << "Answer correct: " << (correctAnswer == true ? "True " : "False ") << std::endl;
	}
	delete[] streamTempRes;
	delete[] X;
	delete[] b;
	delete[] matrixA;
	delete[] linesPerStream;
	delete[] partsOfX;
	delete[] displs;
	MPI_Finalize();
	return 0;
}
