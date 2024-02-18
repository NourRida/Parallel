#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_ITERATIONS 1000
#define SCATTER_CHUNK_SIZE 20000 // Define the static scatter chunk size

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq =  z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITERATIONS) && (lengthsq < 4.0));
    return iter;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute work range for each process
    int work_chunk_size = HEIGHT / size;
    int start_row = rank * work_chunk_size;
    int end_row = start_row + work_chunk_size;
    if (rank == size - 1) {
        end_row = HEIGHT; // Last process handles remaining rows
    }

    // Allocate buffer for result
    int *result_buffer = malloc(WIDTH * (end_row - start_row) * sizeof(int));

    // Compute pixel values for the assigned chunk of rows
    for (int row = start_row; row < end_row; row++) {
        for (int x = 0; x < WIDTH; x++) {
            struct complex c = {
                -2.0 + 3.0 * x / (double)(WIDTH),
                -1.5 + 3.0 * row / (double)(HEIGHT)
            };
            int value = cal_pixel(c);
            result_buffer[(row - start_row) * WIDTH + x] = value;
        }
    }

    // Gather all pixel values at the root process
    int *all_pixels = NULL;
    if (rank == 0) {
        all_pixels = malloc(WIDTH * HEIGHT * sizeof(int));
    }
    MPI_Gather(result_buffer, WIDTH * (end_row - start_row), MPI_INT, all_pixels, WIDTH * (end_row - start_row), MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the gathered pixel values to all processes in larger chunks
    MPI_Scatter(all_pixels, SCATTER_CHUNK_SIZE, MPI_INT, result_buffer, SCATTER_CHUNK_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Save data if rank 0
    if (rank == 0) {
        const char *filename = "mandelbrot_static.pgm";
        FILE *pgmimg = fopen(filename, "wb");
        fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
        fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
        fprintf(pgmimg, "255\n");  // Writing the maximum gray value
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                int temp = all_pixels[i * WIDTH + j];
                fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file
            }
            fprintf(pgmimg, "\n");
        }
        fclose(pgmimg);
        free(all_pixels);
    }

    free(result_buffer);
    MPI_Finalize();
    return 0;
}
