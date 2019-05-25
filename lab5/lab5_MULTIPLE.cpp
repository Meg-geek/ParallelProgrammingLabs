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
#include <pthread.h>

using std::cout;
using std:: endl;
class Task;

bool getNewTask(int procRank);
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
#define DEFAULT_WEIGHT 10000000
#define EXTRA_WEIGHT 500000
#define COUNT_ITER 10
#define COUNT_TASKS 40
#define REQUEST_TAG 1
#define ANSWER_TAG 0
#define DATA_TAG 3

double EPS = 0.000001;

pthread_mutex_t taskMutex;
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

void initFreeTasks(bool *hasFreeTasksArray) {
	for (int i = 0; i < ranksAmount; i++) {
		hasFreeTasksArray[i] = !(currentRank == i);
	}
}

void *calcThread(void *args) {
	bool *hasFreeTasksArray = new bool[ranksAmount];
	initFreeTasks(hasFreeTasksArray);

    for (currentIter = 0; currentIter < COUNT_ITER; ++currentIter) {
        print(CALC_THREAD, "Start iteration #%d", currentIter);
		
        pthread_mutex_lock(&taskMutex);
		print(CALC_THREAD, "Locked the mutex");
        int count = COUNT_TASKS / ranksAmount;
        if(currentRank >= ranksAmount - (COUNT_TASKS % ranksAmount)) {
            ++count;
        }
		print(CALC_THREAD, "Start generating tasks");
        genTasks(tasks, count, currentIter);
        pthread_mutex_unlock(&taskMutex);
		print(CALC_THREAD, "Calc thread start working");
        currentTask = 0;
        for(int i = 0; i < ranksAmount;) {
            //Считаем
			pthread_mutex_lock(&taskMutex);
            while (currentTask < tasks.size()) {
                ++currentTask;
                pthread_mutex_unlock(&taskMutex);

                double result = doTask(&tasks[currentTask - 1]);
                double checkResult = fastCalcTask(&tasks[currentTask - 1]);

                print(CALC_THREAD, "Finish task #%d, result: %lf is %s", currentTask - 1, result,
                      (result - checkResult) < EPS ? "correct" : "incorrect");

                pthread_mutex_lock(&taskMutex);
            }
            pthread_mutex_unlock(&taskMutex);

			int procToAsk = (currentRank + i) % ranksAmount;
            if(!hasFreeTasksArray[procToAsk]) {
				i++;
                continue;
            }
			hasFreeTasksArray[procToAsk] = getNewTask(procToAsk);
            if(!hasFreeTasksArray[procToAsk]) {
                ++i;
            }
        }
        pthread_mutex_unlock(&taskMutex);
        print(CALC_THREAD, "End iteration #%d, wait all", currentIter);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int req = 0;
    MPI_Send(&req, 1, MPI_INT, currentRank, REQUEST_TAG, MPI_COMM_WORLD);
    print(CALC_THREAD, "End work");
    delete[] hasFreeTasksArray;
    return NULL;
}

//функция потока
void *dataThread(void *args) {
    while (currentIter < COUNT_ITER) {
        //Если прислали запрос
        MPI_Status status;
        int res = 0;
        MPI_Recv(&res, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        if(!res) {
            break;
        }
        print(DATA_THREAD, "Get request from #%d", status.MPI_SOURCE);

        pthread_mutex_lock(&taskMutex);
		print(DATA_THREAD, "Locked the mutex");
        if (currentTask >= tasks.size()) { //Если нет лишних заданий countTasks
            int answer = 0;
            MPI_Send(&answer, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
           pthread_mutex_unlock(&taskMutex);
            print(DATA_THREAD, "I don't have data for #%d", status.MPI_SOURCE);
			print(DATA_THREAD, "Unlocked the mutex");
            continue;
        }

        //Если есть свободные задания
        print(DATA_THREAD, "Send data to #%d", status.MPI_SOURCE);
        int answer = 1;
		MPI_Send(&answer, 1, MPI_INT, status.MPI_SOURCE, ANSWER_TAG, MPI_COMM_WORLD);
        MPI_Send(&tasks.back(), 1, TASK_DATATYPE, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD);
        tasks.pop_back();
        pthread_mutex_unlock(&taskMutex);
		print(DATA_THREAD, "Unlocked the mutex");
    }
    print(DATA_THREAD, "End work");
    return NULL;
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


bool getNewTask(int procToAsk) {
    print(CALC_THREAD, "Try get data from #%d", procToAsk);

    int req = 1;
    MPI_Send(&req, 1, MPI_INT, procToAsk, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&req, 1, MPI_INT, MPI_ANY_SOURCE, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(req == 0) {
        print(CALC_THREAD, "#%d doesn't has tasks", procToAsk);
        return false;
    }

    Task task;
    MPI_Recv(&task, 1, TASK_DATATYPE, procToAsk, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pthread_mutex_lock(&taskMutex);
    tasks.push_back(task);
    pthread_mutex_unlock(&taskMutex);

    print(CALC_THREAD, "New task loaded from #%d", procToAsk);
    return true;
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
        perror("Cannot create calc thread");
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
    countOperations += task->count / 1000000;

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
    int weight = abs(currentRank - (currentIter % ranksAmount));
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

	cout << "Process Rank: " << currentRank << endl;
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
	int provided = MPI::Init_thread(argc, argv, MPI::THREAD_MULTIPLE);
	if (provided < MPI::THREAD_MULTIPLE)
	{
		printf("ERROR: The MPI library does not have full thread support\n");
		MPI::COMM_WORLD.Abort(1);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &ranksAmount);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

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
	

	pthread_mutex_destroy(&taskMutex);
	if (currentRank == 0) {
		cout << "Time: " << end - start << endl;
		cout << "Operations amount: " << countOperations << " milloins" << endl;
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
