#include <mpi.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <stdlib.h>

void decompose_domain(int domain_size, int world_rank,
                      int world_size, int* subdomain_start,
                      int* subdomain_size) {
    if (world_size > domain_size) {
        // Don't worry about this special case. Assume the domain
        // size is greater than the world size.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *subdomain_start = domain_size / world_size * world_rank;
    *subdomain_size = domain_size / world_size;
    if (world_rank == world_size - 1) {
        // Give remainder to last process
        *subdomain_size += domain_size % world_size;
    }
  }

typedef struct {
    int location;
    int num_steps_left_in_walk;
} Walker;

void initialize_walkers(int num_walkers_per_proc, int max_walk_size,
                        int subdomain_start, int subdomain_size,
                        std::vector<Walker>* incoming_walkers) {
    Walker walker;
    for (int i = 0; i < num_walkers_per_proc; i++) {
        // Initialize walkers in the middle of the subdomain
        walker.location = subdomain_start;
        walker.num_steps_left_in_walk =
            (rand() / (float)RAND_MAX) * max_walk_size;
        incoming_walkers->push_back(walker);
    }
}

void walk(Walker* walker, int subdomain_start, int subdomain_size,
          int domain_size, std::vector<Walker>* outgoing_walkers) {
    while (walker->num_steps_left_in_walk > 0) {
        if (walker->location == subdomain_start + subdomain_size) {
            // Take care of the case when the walker is at the end
            // of the domain by wrapping it around to the beginning
            if (walker->location == domain_size) {
                walker->location = 0;
            }
            outgoing_walkers->push_back(*walker);
            break;
        } else {
            walker->num_steps_left_in_walk--;
            walker->location++;
        }
    }
}

void send_outgoing_walkers(std::vector<Walker>* outgoing_walkers, int world_rank, int world_size) 
{

    MPI_Send((void*)outgoing_walkers->data(), outgoing_walkers->size()*sizeof(Walker), MPI_BYTE, (world_rank + 1)%world_size, 0, MPI_COMM_WORLD);
    outgoing_walkers->clear();
}

void receive_incoming_walkers(std::vector<Walker>* incoming_walkers, int world_rank, int world_size)
{
    MPI_Status status;
    int incoming_rank = (world_rank == 0) ? world_size - 1 : world_rank - 1;
    MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);
    int walker_size;
    MPI_Get_count(&status, MPI_BYTE, &walker_size);
    incoming_walkers->resize(
        walker_size / sizeof(Walker));
    MPI_Recv((void*)incoming_walkers->data(), walker_size,
             MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
}

int main() 
{
    const int domain_size = 10;
    const int max_walk_size = 1;
    const int num_walkers_per_proc = 5;
    std::vector<Walker> incoming_walkers;
    std::vector<Walker> outgoing_walkers;
    int subdomain_start, subdomain_size;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Find your part of the domain
    decompose_domain(domain_size, world_rank, world_size,
            &subdomain_start, &subdomain_size);

    // Initialize walkers in your subdomain
    initialize_walkers(num_walkers_per_proc, max_walk_size,
            subdomain_start, subdomain_size,
            &incoming_walkers);

    int maximum_sends_recvs =
        max_walk_size / (domain_size / world_size) + 1;
    for (int m = 0; m < maximum_sends_recvs; m++) {
        // Process all incoming walkers
        for (int i = 0; i < incoming_walkers.size(); i++) {
            walk(&incoming_walkers[i], subdomain_start, subdomain_size, domain_size, &outgoing_walkers); 
        }

        // Send and receive if you are even and vice versa for odd
        if (world_rank % 2 == 0) {
            send_outgoing_walkers(&outgoing_walkers, world_rank, world_size);
            receive_incoming_walkers(&incoming_walkers, world_rank, world_size);
        } else {
            receive_incoming_walkers(&incoming_walkers, world_rank, world_size);
            send_outgoing_walkers(&outgoing_walkers, world_rank, world_size);
        }
    }

    MPI_Finalize();

    return 0;
}

