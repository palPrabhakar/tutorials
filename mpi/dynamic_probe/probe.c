#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number_count;

    if (world_rank == 0) {
        const int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];
        
        srand(time(NULL));
        number_count = (rand() / (float)RAND_MAX) * MAX_NUMBERS;

         MPI_Send(numbers, number_count, MPI_INT, 1, 0, MPI_COMM_WORLD);
         printf("0 sent %d numbers to 1\n", number_count);
    }
    else if (world_rank == 1) {
        MPI_Status status;

        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

        MPI_Get_count(&status, MPI_INT, &number_count);

        int* number_buf = (int*)malloc(sizeof(int)*number_count);

        MPI_Recv(number_buf, number_count, MPI_INT, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("1 dynamically received %d numbers from 0.\n",
                number_count);
        free(number_buf);
    }

    MPI_Finalize();

    return 0;
}



    
    