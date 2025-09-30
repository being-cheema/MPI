#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <chrono>  // For timing

// Function to allocate a matrix with random values from 0-9
int allocMatrix(int*** mat, int rows, int cols) {
    int* p = (int*)malloc(sizeof(int) * rows * cols);
    if (!p) return -1;

    *mat = (int**)malloc(rows * sizeof(int*));
    if (!(*mat)) {
        free(p);
        return -1;
    }

    for (int i = 0; i < rows; i++) {
        (*mat)[i] = &(p[i * cols]);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*mat)[i][j] = rand() % 10;
        }
    }

    return 0;
}

int freeMatrix(int ***mat) {
    free((*mat)[0]);
    free(*mat);
    return 0;
}

void matrixMultiply(int **a, int **b, int rows, int cols, int ***c) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int val = 0;
            for (int k = 0; k < rows; k++) {
                val += a[i][k] * b[k][j];
            }
            (*c)[i][j] = val;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Comm cartComm;
    int dim[2], period[2], reorder;
    int coord[2], id;
    int **A = NULL, **B = NULL, **C = NULL;
    int **localA = NULL, **localB = NULL, **localC = NULL;
    int rows = 0, columns = 0;
    int worldSize, procDim, blockDim;
    int left, right, up, down;
    int bCastData[4];  // procDim, blockDim, rows, columns

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Command-line arguments
        if (argc >= 3) {
            rows = atoi(argv[1]);
            columns = atoi(argv[2]);
        } else {
            rows = 4;
            columns = 4;
            printf("No matrix size provided. Using default: 4 x 4\n");
        }

        // Check square matrix
        if (rows != columns) {
            printf("[ERROR] Only square matrices supported! Got %d x %d\n", rows, columns);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Check process grid compatibility
        double sqroot = sqrt(worldSize);
        if ((sqroot - floor(sqroot)) != 0) {
            printf("[ERROR] Number of processes (%d) must be a perfect square!\n", worldSize);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        int intRoot = (int)sqroot;
        if (rows % intRoot != 0) {
            printf("[ERROR] Matrix size %d not divisible by process grid dimension %d!\n", rows, intRoot);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        procDim = intRoot;
        blockDim = rows / intRoot;

        // Prepare broadcast data
        bCastData[0] = procDim;
        bCastData[1] = blockDim;
        bCastData[2] = rows;
        bCastData[3] = columns;
    }

    // Broadcast problem dimensions
    MPI_Bcast(&bCastData, 4, MPI_INT, 0, MPI_COMM_WORLD);
    procDim = bCastData[0];
    blockDim = bCastData[1];
    rows = bCastData[2];
    columns = bCastData[3];

    // Allocate global matrices on all ranks
    if (allocMatrix(&A, rows, columns) != 0) {
        printf("[ERROR] Matrix alloc for A failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (allocMatrix(&B, rows, columns) != 0) {
        printf("[ERROR] Matrix alloc for B failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    if (allocMatrix(&C, rows, columns) != 0) {
        printf("[ERROR] Matrix alloc for C failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 6);
    }

    // Create Cartesian grid
    dim[0] = procDim; dim[1] = procDim;
    period[0] = 1; period[1] = 1;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);

    // Allocate local blocks
    allocMatrix(&localA, blockDim, blockDim);
    allocMatrix(&localB, blockDim, blockDim);

    // Subarray datatype
    int globalSize[2] = { rows, columns };
    int localSize[2] = { blockDim, blockDim };
    int starts[2] = { 0, 0 };
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, globalSize, localSize, starts, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, blockDim * sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int *globalptrA = NULL, *globalptrB = NULL, *globalptrC = NULL;
    if (rank == 0) {
        globalptrA = &(A[0][0]);
        globalptrB = &(B[0][0]);
        globalptrC = &(C[0][0]);
    }

    // Scatter to local blocks
    int* sendCounts = (int*)malloc(sizeof(int) * worldSize);
    int* displacements = (int*)malloc(sizeof(int) * worldSize);
    if (rank == 0) {
        for (int i = 0; i < worldSize; i++) sendCounts[i] = 1;
        int disp = 0;
        for (int i = 0; i < procDim; i++) {
            for (int j = 0; j < procDim; j++) {
                displacements[i * procDim + j] = disp;
                disp += 1;
            }
            disp += (blockDim - 1) * procDim;
        }
    }

    MPI_Scatterv(globalptrA, sendCounts, displacements, subarrtype, &(localA[0][0]),
        rows * columns / (worldSize), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalptrB, sendCounts, displacements, subarrtype, &(localB[0][0]),
        rows * columns / (worldSize), MPI_INT, 0, MPI_COMM_WORLD);

    if (allocMatrix(&localC, blockDim, blockDim) != 0) {
        printf("[ERROR] Matrix alloc for localC in rank %d failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    // Initial skew
    MPI_Cart_coords(cartComm, rank, 2, coord);
    MPI_Cart_shift(cartComm, 1, coord[0], &left, &right);
    MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(cartComm, 0, coord[1], &up, &down);
    MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);

    // Multiply with timing
    auto start = std::chrono::high_resolution_clock::now();
    int** multiplyRes = NULL;
    if (allocMatrix(&multiplyRes, blockDim, blockDim) != 0) {
        printf("[ERROR] Matrix alloc for multiplyRes in rank %d failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 8);
    }
    for (int k = 0; k < procDim; k++) {
        matrixMultiply(localA, localB, blockDim, blockDim, &multiplyRes);
        for (int i = 0; i < blockDim; i++) {
            for (int j = 0; j < blockDim; j++) {
                localC[i][j] += multiplyRes[i][j];
            }
        }
        MPI_Cart_shift(cartComm, 1, 1, &left, &right);
        MPI_Cart_shift(cartComm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, MPI_INT, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, MPI_INT, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (rank == 0) {
        printf("Matrix multiplication (%d x %d) with %d processes took %f seconds.\n",
               rows, columns, worldSize, duration.count());
    }

    // Gather results
    MPI_Gatherv(&(localC[0][0]), rows * columns / worldSize, MPI_INT,
        globalptrC, sendCounts, displacements, subarrtype, 0, MPI_COMM_WORLD);

    // Cleanup
    freeMatrix(&localC);
    freeMatrix(&multiplyRes);
    MPI_Finalize();
    return 0;
}
