#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

using namespace std;
using namespace std::chrono;

#define N 10000 // Use 1000 for demo; adjust for performance tests

MPI_Status status;

// Helper to access 1D vector as 2D matrix
inline double& at(vector<double>& mat, int row, int col, int ncols) {
    return mat[row * ncols + col];
}

// Serial multiplication of identity matrices for correctness test
void multiplyIdentitySerial(int size, vector<double>& C) {
    // C = I * I = I
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            at(C, i, j, size) = (i == j) ? 1.0 : 0.0;
        }
    }
}

// Check if matrix C is identity matrix of given size
bool isIdentityMatrix(const vector<double>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (C[i * size + j] != expected) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    int processCount, processId;
    int slaveTaskCount, source, dest, rows, offset;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    slaveTaskCount = processCount - 1;

    // Define matrices as vectors dynamically
    vector<double> matrix_a;
    vector<double> matrix_b;
    vector<double> matrix_c;

    if (processId == 0) {
        // Root process initializes matrices
        matrix_a.resize(N * N, 0);
        matrix_b.resize(N * N, 0);
        matrix_c.resize(N * N, 0);

        // Initialize matrix_a and matrix_b as identity matrices for correctness test
        for (int i = 0; i < N; i++) {
            at(matrix_a, i, i, N) = 1.0;
            at(matrix_b, i, i, N) = 1.0;
        }

        // Distribute rows to slaves
        int baseRows = N / slaveTaskCount;
        int extraRows = N % slaveTaskCount;
        offset = 0;

        for (dest = 1; dest <= slaveTaskCount; dest++) {
            rows = baseRows + (dest <= extraRows ? 1 : 0);

            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix_a[offset * N], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(matrix_b.data(), N * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            offset += rows;
        }

        // Time parallel multiplication only (receive phase excluded as it’s after multiplication)
        auto start = high_resolution_clock::now();

        // Receive results from slaves
        offset = 0;
        for (int i = 1; i <= slaveTaskCount; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix_c[offset * N], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        auto stop = high_resolution_clock::now();
        duration<double> duration_sec = stop - start;

        cout << "Parallel multiplication took " << duration_sec.count() << " seconds." << endl;

        // Serial multiplication of identity matrices (only correctness)
        vector<double> serial_result(N * N, 0);
        multiplyIdentitySerial(N, serial_result);

        // Check correctness: matrix_c vs serial_result
        bool correct = true;
        for (int i = 0; i < N * N; ++i) {
            if (matrix_c[i] != serial_result[i]) {
                correct = false;
                break;
            }
        }

        cout << "Correctness check on identity matrices: " << (correct ? "PASSED" : "FAILED") << endl;

    } else {
        // Slave processes
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        vector<double> sub_matrix_a(rows * N);
        vector<double> matrix_b(N * N);
        vector<double> sub_matrix_c(rows * N, 0);

        MPI_Recv(sub_matrix_a.data(), rows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(matrix_b.data(), N * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

        // Perform matrix multiplication
        for (int i = 0; i < rows; ++i) {
            for (int k = 0; k < N; ++k) {
                double sum = 0.0;
                for (int j = 0; j < N; ++j) {
                    sum += sub_matrix_a[i * N + j] * matrix_b[j * N + k];
                }
                sub_matrix_c[i * N + k] = sum;
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(sub_matrix_c.data(), rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
