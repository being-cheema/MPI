#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>  // for strcmp
#include <chrono>    // <--- Added chrono include

int allocMatrix(int*** mat, int rows, int cols) {
    int* p = (int*)malloc(sizeof(int) * rows * cols);
    if (!p) return -1;
    *mat = (int**)malloc(rows * sizeof(int*));
    if (!*mat) {
        free(p);
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        (*mat)[i] = &(p[i * cols]);
    }
    return 0;
}

void freeMatrix(int*** mat) {
    if (mat && *mat) {
        free(&((*mat)[0][0]));
        free(*mat);
        *mat = NULL;
    }
}

void fillRandomMatrix(int** mat, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            mat[i][j] = rand() % 10;  // random 0-9
}

void fillIdentityBlock(int** mat, int blockSize, int coords[2], int procDim) {
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            int globalRow = coords[0] * blockSize + i;
            int globalCol = coords[1] * blockSize + j;
            mat[i][j] = (globalRow == globalCol) ? 1 : 0;
        }
    }
}

void matrixMultiply(int** a, int** b, int blockSize, int** c) {
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            int sum = 0;
            for (int k = 0; k < blockSize; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

void addMatrix(int** a, int** b, int blockSize) {
    for (int i = 0; i < blockSize; i++)
        for (int j = 0; j < blockSize; j++)
            a[i][j] += b[i][j];
}

bool verifyIdentity(int** C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j && C[i][j] != 1) {
                printf("Verification failed at diagonal element (%d,%d): %d != 1\n", i, j, C[i][j]);
                return false;
            }
            else if (i != j && C[i][j] != 0) {
                printf("Verification failed at off-diagonal element (%d,%d): %d != 0\n", i, j, C[i][j]);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int procDim = (int)sqrt((double)worldSize);
    if (procDim * procDim != worldSize) {
        if (rank == 0) printf("Number of processes must be a perfect square\n");
        MPI_Finalize();
        return -1;
    }

    if (argc < 2) {
        if (rank == 0) printf("Usage: mpirun -np <p^2> %s <matrix_size> [identity]\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    int N = atoi(argv[1]);
    if (N % procDim != 0) {
        if (rank == 0) printf("Matrix size must be divisible by sqrt(worldSize)=%d\n", procDim);
        MPI_Finalize();
        return -1;
    }

    int useIdentity = 0;
    if (argc >= 3 && strcmp(argv[2], "identity") == 0) {
        useIdentity = 1;
        if (rank == 0) printf("Running identity matrix multiplication test...\n");
    }

    int blockSize = N / procDim;

    if (rank == 0 && !useIdentity) srand(time(NULL));

    int dims[2] = { procDim, procDim };
    int periods[2] = { 1, 1 };
    int reorder = 1;
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartComm);

    int coords[2];
    MPI_Cart_coords(cartComm, rank, 2, coords);

    int** localA; allocMatrix(&localA, blockSize, blockSize);
    int** localB; allocMatrix(&localB, blockSize, blockSize);
    int** localC; allocMatrix(&localC, blockSize, blockSize);

    for (int i = 0; i < blockSize; i++)
        for (int j = 0; j < blockSize; j++)
            localC[i][j] = 0;

    if (useIdentity) {
        // Fill identity blocks on all processes
        fillIdentityBlock(localA, blockSize, coords, procDim);
        fillIdentityBlock(localB, blockSize, coords, procDim);
    } else {
        if (rank == 0) {
            int** A; allocMatrix(&A, N, N);
            int** B; allocMatrix(&B, N, N);

            fillRandomMatrix(A, N);
            fillRandomMatrix(B, N);

            // Scatter blocks to all processes including self
            for (int r = 0; r < worldSize; r++) {
                int destCoords[2];
                MPI_Cart_coords(cartComm, r, 2, destCoords);

                if (r == 0) {
                    for (int i = 0; i < blockSize; i++)
                        for (int j = 0; j < blockSize; j++) {
                            localA[i][j] = A[i][j];
                            localB[i][j] = B[i][j];
                        }
                } else {
                    int* sendBufA = (int*)malloc(blockSize * blockSize * sizeof(int));
                    int* sendBufB = (int*)malloc(blockSize * blockSize * sizeof(int));

                    int startRow = destCoords[0] * blockSize;
                    int startCol = destCoords[1] * blockSize;

                    for (int i = 0; i < blockSize; i++)
                        for (int j = 0; j < blockSize; j++) {
                            sendBufA[i * blockSize + j] = A[startRow + i][startCol + j];
                            sendBufB[i * blockSize + j] = B[startRow + i][startCol + j];
                        }

                    MPI_Send(sendBufA, blockSize * blockSize, MPI_INT, r, 0, MPI_COMM_WORLD);
                    MPI_Send(sendBufB, blockSize * blockSize, MPI_INT, r, 1, MPI_COMM_WORLD);

                    free(sendBufA);
                    free(sendBufB);
                }
            }

            freeMatrix(&A);
            freeMatrix(&B);
        } else {
            MPI_Recv(&(localA[0][0]), blockSize * blockSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&(localB[0][0]), blockSize * blockSize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Initial skewing:
    int leftRank, rightRank, upRank, downRank;

    MPI_Cart_shift(cartComm, 1, -coords[0], &rightRank, &leftRank);
    MPI_Sendrecv_replace(&(localA[0][0]), blockSize * blockSize, MPI_INT,
                         leftRank, 0, rightRank, 0, cartComm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(cartComm, 0, -coords[1], &downRank, &upRank);
    MPI_Sendrecv_replace(&(localB[0][0]), blockSize * blockSize, MPI_INT,
                         upRank, 1, downRank, 1, cartComm, MPI_STATUS_IGNORE);

    int** tempC; allocMatrix(&tempC, blockSize, blockSize);

    // Start timing here
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < procDim; step++) {
        matrixMultiply(localA, localB, blockSize, tempC);
        addMatrix(localC, tempC, blockSize);

        MPI_Cart_shift(cartComm, 1, 1, &rightRank, &leftRank);
        MPI_Sendrecv_replace(&(localA[0][0]), blockSize * blockSize, MPI_INT,
                             leftRank, 2, rightRank, 2, cartComm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(cartComm, 0, 1, &downRank, &upRank);
        MPI_Sendrecv_replace(&(localB[0][0]), blockSize * blockSize, MPI_INT,
                             upRank, 3, downRank, 3, cartComm, MPI_STATUS_IGNORE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    std::chrono::duration<double> diff = end - start;
    if (rank == 0) {
        printf("Matrix multiplication took %.6f seconds\n", diff.count());
    }

    freeMatrix(&tempC);

    // Gather results at rank 0
    if (rank == 0) {
        int** C; allocMatrix(&C, N, N);

        for (int i = 0; i < blockSize; i++)
            for (int j = 0; j < blockSize; j++)
                C[i][j] = localC[i][j];

        for (int r = 1; r < worldSize; r++) {
            int coordsRecv[2];
            MPI_Cart_coords(cartComm, r, 2, coordsRecv);

            int* recvBuf = (int*)malloc(blockSize * blockSize * sizeof(int));
            MPI_Recv(recvBuf, blockSize * blockSize, MPI_INT, r, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int startRow = coordsRecv[0] * blockSize;
            int startCol = coordsRecv[1] * blockSize;

            for (int i = 0; i < blockSize; i++)
                for (int j = 0; j < blockSize; j++)
                    C[startRow + i][startCol + j] = recvBuf[i * blockSize + j];

            free(recvBuf);
        }

        if (useIdentity) {
            if (verifyIdentity(C, N)) {
                printf("Identity matrix multiplication verification PASSED.\n");
            } else {
                printf("Identity matrix multiplication verification FAILED.\n");
            }
        }

        freeMatrix(&C);
    } else {
        MPI_Send(&(localC[0][0]), blockSize * blockSize, MPI_INT, 0, 4, MPI_COMM_WORLD);
    }

    freeMatrix(&localA);
    freeMatrix(&localB);
    freeMatrix(&localC);

    MPI_Comm_free(&cartComm);
    MPI_Finalize();
    return 0;
}
