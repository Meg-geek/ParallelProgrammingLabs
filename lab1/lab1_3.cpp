#include <iostream>
#include <cmath>
#include <mpi.h>

int main(int argc, char **argv)
{
	int N = 5000;
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

	double *streamX = new double[linesPerStream[curStream]];
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
	normB = sqrt(normB);
	//посчитали норму b


 double *localSum = new double[linesPerStream[curStream]];
  while (1) {
      for (int i = 0; i < linesPerStream[curStream]; i++) {
          localSum[i] = 0;
      }
      int currentLine = beginLine;
      for (int i = 0; i < numbStreams; i++) {
          int endLine = currentLine + linesPerStream[i];
          for (int j = 0; j < linesPerStream[curStream]; j++) {
              for (int k = currentLine; k < endLine; k++) {
                  localSum[j] += matrixA [j * N + k] * streamX[k - currentLine];
              }
          }
          MPI_Sendrecv_replace(streamX, linesPerStream[curStream], MPI_DOUBLE, (curStream - 1 + numbStreams) % numbStreams, 0,
              (curStream + 1) % numbStreams, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  /*
		  int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, 
                       int dest, int sendtag, int source, int recvtag,
                       MPI_Comm comm, MPI_Status *status)

					   count
						number of elements in send and receive buffer (integer)
					datatype
							type of elements in send and receive buffer (handle)
						dest
						rank of destination (integer)
					sendtag
						send message tag (integer)
					source
						rank of source (integer)
					recvtag
						receive message tag (integer)
					comm
						communicator (handle)
		  */
          currentLine = (currentLine + linesPerStream[i]) % N;
      }
      double threadAnswer = 0;
      for (int i = 0; i < linesPerStream[curStream]; i++) {
          localSum[i] -= b[i]; //Ax - b
          streamX[i] = streamX[i] - localSum[i] * T;
          threadAnswer += localSum[i] * localSum[i];
      }

      double norm_A = 0;
      MPI_Allreduce(&threadAnswer, &norm_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      norm_A = sqrt(norm_A);
      if (norm_A / normB < E) {
          break;
      }
  }
  
  double *fullX = new double[N]();
  MPI_Allgatherv(streamX, linesPerStream[curStream], MPI_DOUBLE, fullX, linesPerStream, displs, MPI_DOUBLE, MPI_COMM_WORLD);

	if (curStream == 0) {
		double finish = MPI_Wtime();
		std::cout << "Threads: " << numbStreams << ", time: " << (finish - start) << std::endl;
		bool correctAnswer = true;
		for (int i = 0; i < N; i++) {
			if (fabs(fabs(fullX[i]) - 1) > E) {
				std::cout << i << " " <<fullX[i] << " " << E << " " << fabs(fabs(fullX[i]) - 1) << " error" << std::endl;
				correctAnswer = false;
				break;
			}
		}
		std::cout << "Answer correct: " << (correctAnswer == true ? "True " : "False ") << std::endl;
	}
	delete[] streamX;
	delete[] b;
	delete[] matrixA;
	delete[] linesPerStream;
	delete[] partsOfX;
	delete[] displs;
	delete[] localSum;
	delete[] fullX;
	MPI_Finalize();
	return 0;
}
