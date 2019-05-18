#include <iostream>
#include <mpi.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <cstdarg>
#include <unistd.h>
#include <cstddef>
#include <vector>

using std::cout;
using std:: endl;
class Task;

double doTask(Task *task);
double fastCalcTask(Task *task);
int rand(int from, int to);
double rand(double from, double to);
void genTasks(std::vector<Task> &tasks, int countTasks, int currentIter);
void print(const char *msg, ...);
void print(int threadRank, const char *msg, ...);

#define PTHREAD_INIT_SUCCESS 0
#define SETDETACHSTATE_SUCCESS 0
#define PTHREAD_JOIN_SUCCESS 0

#define DATA_THREAD 0
#define CALC_THREAD 1
#define DEFAULT_WEIGHT 100
#define EXTRA_WEIGHT 5
#define COUNT_ITER 4
#define COUNT_TASKS 20
#define REQUEST_TAG 1
#define ANSWER_TAG 2
#define DATA_TAG 3
#define CHIP_TAG 5

double EPS = 0.000001;

pthread_mutex_t taskMutex;
pthread_mutex_t iterMutex;
//объекты типа "описатель потока"
pthread_t threadsDescr[2];
std::vector<Task> tasks;
MPI_Datatype TASK_DATATYPE;
int currentRank;
int ranksAmount;
int currentIter;
int currentTask;
long long int countOperations;



class Task {
public:
	Task() {}
	double start;
	double step;
	int count;
};

volatile bool threadTasksFlag;
bool chip;
volatile bool iterationContinue;

void *calcThread(void *args) {
	for (currentIter = 0; currentIter < COUNT_ITER; currentIter++) {
		print(CALC_THREAD, "Start iteration #%d", currentIter);
		print(CALC_THREAD, "Generate tasks");
		pthread_mutex_lock(&taskMutex);
		int count = COUNT_TASKS / ranksAmount;
		if (currentRank >= ranksAmount - (COUNT_TASKS % ranksAmount)) {
			++count;
		}
		genTasks(tasks, count, currentIter);
		pthread_mutex_unlock(&taskMutex);
		threadTasksFlag = iterationContinue = true;
		if (currentRank == 0) {
			chip = true;
		}
		currentTask = 0;
		while (iterationContinue) {
			pthread_mutex_lock(&taskMutex);
			if (currentTask < tasks.size()) {
				threadTasksFlag = true;
				double result = doTask(&tasks[currentTask]);
				double checkResult = fastCalcTask(&tasks[currentTask]);
				++currentTask;
				print(CALC_THREAD, "Finish task #%d, result: %lf is %s", currentTask - 1, result,
					(result - checkResult) < EPS ? "correct" : "incorrect");
			}
			else {
				threadTasksFlag = false;
			}
			pthread_mutex_unlock(&taskMutex);
		}

		print(CALC_THREAD, "End iteration #%d, wait all", currentIter);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	print(CALC_THREAD, "End work");
	return NULL;
}

void *dataThread(void *args) {
	while (currentIter < COUNT_ITER) {
		MPI_Request sendRequest;
		MPI_Request recRequest;
		while (iterationContinue) {
			//ждем, когда наш процесс сделает все задания
			for (; threadTasksFlag;);
            print(DATA_THREAD, "Process did all the tasks");
			//если у нас есть фишка и она до этого была у всех
			if (currentRank == ranksAmount - 1 && chip) {
				iterationContinue = false;
			}
			//если у нас есть фишка - передаем по кругу, для нас итерация закончена
			if (chip) {
				int dest = currentRank + 1;
				if (dest >= ranksAmount) {
					dest = 0;
				}
				MPI_Isend(&chip, 1, MPI_C_BOOL, dest, CHIP_TAG, MPI_COMM_WORLD, &sendRequest);
				chip = false;
                iterationContinue = false;
                continue;
			}
			int source = (currentRank != 0) ? currentRank - 1 : ranksAmount - 1;
			MPI_Irecv(&chip, 1, MPI_C_BOOL, source, CHIP_TAG, MPI_COMM_WORLD, &recRequest);
			MPI_Wait(&recRequest, MPI_STATUS_IGNORE);
			//MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
		}
	}
}

void createType() {
    const int nBlocks = 3;
    int blockLengthArray[] = {1, 1, 1};
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(Task, start);
    offsets[1] = offsetof(Task, step);
    offsets[2] = offsetof(Task, count);

    MPI_Type_create_struct(nBlocks, blockLengthArray, offsets, types, &TASK_DATATYPE);
    MPI_Type_commit(&TASK_DATATYPE);
}

int makeThreads() {
	//атрибуты потока
    pthread_attr_t attrs;

    if (pthread_attr_init(&attrs) != PTHREAD_INIT_SUCCESS) {
        perror("Cannot initialize attributes");
        return EXIT_FAILURE;
    };

    if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != SETDETACHSTATE_SUCCESS) {
        perror("Error in setting attributes");
        return EXIT_FAILURE;
    }
	
    if (pthread_create(&threadsDescr[CALC_THREAD], &attrs, calcThread, NULL) != EXIT_SUCCESS) {
        perror("Cannot create th calc thread");
        return EXIT_FAILURE;
    }
	
    if (pthread_create(&threadsDescr[DATA_THREAD], &attrs, dataThread, NULL) != EXIT_SUCCESS) {
        perror("Cannot create data thread");
        return EXIT_FAILURE;
    }
	//освобождение ресурсов, занимаемых описателем атрибутов
    pthread_attr_destroy(&attrs);
	
    for (int i = 0; i < 2; ++i) {
        if (pthread_join(threadsDescr[i], NULL) != EXIT_SUCCESS) {
            perror("Cannot join a thread");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

double doTask(Task *task) {
    countOperations += task->count;

    double sum = 0;
    double tmp = task->start;
    for (int i = 0; i < task->count; ++i) {
        tmp = pow(tmp, 2.);
        tmp = sqrt(tmp);
        sum += tmp;
        tmp *= task->step;
    }

    return sum;
}

double fastCalcTask(Task *task) {
    return task->start * (1 - pow(task->step, task->count)) / (1 - task->step);
}

int rand(int from, int to) {
    return rand() % (to - from) + from;
}

double rand(double from, double to) {
    double res = (double) rand(0, INT_MAX) / INT_MAX;
    return res * (to - from) + from;
}

void genTasks(std::vector<Task> &tasks, int countTasks, int currentIter) {
    int weight = abs(currentRank - (currentIter % ranksAmount)) + 1;

    print(CALC_THREAD, "Generate %d tasks with weight %d", countTasks, weight);
    tasks.clear();

    for (int i = 0; i < countTasks; ++i) {
        Task task;
        task.start = rand(1000., 10000.);
        task.count = DEFAULT_WEIGHT + EXTRA_WEIGHT * weight;
        task.step = 1 / rand(2., 100.);
        tasks.push_back(task);
    }
}

void print(const char *msg, ...) {
    va_list args;
    va_start (args, msg);

	cout << "Process Rank: " << currentRank << " ";
    vprintf(msg, args);
	cout << endl;
    fflush(stdout);

    va_end (args);
}

void print(int threadRank, const char *msg, ...) {
    va_list args;
    va_start (args, msg);

    printf("Process #%d (thread #%d): ", currentRank, threadRank);
    vprintf(msg, args);
    printf("\n");
    fflush(stdout);

    va_end (args);
}

int main(int argc, char **argv) {
	int provided = MPI::Init_thread(argc, argv, MPI::THREAD_FUNNELED);
	if (provided < MPI::THREAD_FUNNELED)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI::COMM_WORLD.Abort(1);
	}


	MPI_Comm_size(MPI_COMM_WORLD, &ranksAmount);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    if(currentRank == 0){
        cout<<"Processes amount: "<<ranksAmount<<endl;
    }
	
	createType();

	srand(time(NULL));
	pthread_mutex_init(&taskMutex, NULL);

	double start = MPI_Wtime();

	if (makeThreads() != EXIT_SUCCESS) {
		pthread_mutex_destroy(&taskMutex);
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	double end = MPI_Wtime();
    if(currentRank == 0){
        cout << "Time: " << end - start << endl;
    }

	pthread_mutex_destroy(&taskMutex);
    
    print("Operations amount: %lld", countOperations);
	//cout << "Operations amount: " << countOperations << endl;

	MPI_Finalize();
	return EXIT_SUCCESS;
}
