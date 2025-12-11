#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#include "bmp_structs.h"

static inline uint8_t median_uint8_hist(const uint8_t *values, int size) {
    long hist[256] = {0}; // For large masks, it might be better to reuse hist[]

    for (int i = 0; i < size; i++)
        hist[values[i]]++;

    int mid = size / 2;
    int acc = 0;

    for (int v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc > mid)
            return (uint8_t)v;
    }
}

// Function to apply median filter
void apply_median_filter(pixel *local_input_1d, pixel *local_output_1d, int local_height, int width, int N, int rank, int size, MPI_Comm comm) {
    int half = N / 2;
    int size_ = N * N;
    int med_idx = size_ / 2;

    // Allocate temps once
    uint8_t *r_values = (uint8_t *)malloc(size_ * sizeof(uint8_t));
    uint8_t *g_values = (uint8_t *)malloc(size_ * sizeof(uint8_t));
    uint8_t *b_values = (uint8_t *)malloc(size_ * sizeof(uint8_t));
    if (r_values == NULL || g_values == NULL || b_values == NULL) {
        fprintf(stderr, "Allocation failed in rank %d\n", rank);
        MPI_Abort(comm, 1);
    }

    // Halo offset
    for (int y = 0; y < local_height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = 0;
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    int ny = y + dy + half;
                    int nx = x + dx;
                    ny = (ny < 0) ? 0 : (ny >= local_height + 2 * half ? local_height + 2 * half - 1 : ny);
                    nx = (nx < 0) ? 0 : (nx >= width ? width - 1 : nx);
                    r_values[idx] = local_input_1d[ny * width + nx].r;
                    g_values[idx] = local_input_1d[ny * width + nx].g;
                    b_values[idx] = local_input_1d[ny * width + nx].b;
                    idx++;
                }
            }
            local_output_1d[y * width + x].r = median_uint8_hist(r_values, size_);
            local_output_1d[y * width + x].g = median_uint8_hist(g_values, size_);
            local_output_1d[y * width + x].b = median_uint8_hist(b_values, size_);
        }
    }

    free(r_values);
    free(g_values);
    free(b_values);
}

// Function to convert to grayscale (in-place, local)
void convert_to_grayscale_inplace(pixel *matrix, int rows, int cols) {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            double val = 0.299 * matrix[idx].r + 0.587 * matrix[idx].g + 0.114 * matrix[idx].b;
            uint8_t gray_val = (uint8_t)(val + 0.5);
            matrix[idx].r = gray_val;
            matrix[idx].g = gray_val;
            matrix[idx].b = gray_val;
        }
    }
}

// Function to perform histogram equalization (distributed)
void equalize_histogram(pixel *matrix, int rows, int cols, int rank, int size, MPI_Comm comm) {
    long local_hist[256] = {0};
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            local_hist[matrix[y * cols + x].r]++;
        }
    }

    // Sum all processes' 256-bin histograms into 'hist' on rank 0.
    long hist[256];
    MPI_Reduce(local_hist, hist, 256, MPI_LONG, MPI_SUM, 0, comm);

    uint8_t map[256];
    if (rank == 0) {
        long cdf[256] = {0};
        cdf[0] = hist[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + hist[i];
        }
        double total_pixels = (double)cdf[255];
        for (int i = 0; i < 256; i++) {
            map[i] = (uint8_t)((cdf[i] * 255.0 / total_pixels) + 0.5);
        }
    }
    MPI_Bcast(map, 256, MPI_UINT8_T, 0, comm);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            uint8_t old_val = matrix[idx].r;
            uint8_t new_val = map[old_val];
            matrix[idx].r = new_val;
            matrix[idx].g = new_val;
            matrix[idx].b = new_val;
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) fprintf(stderr, "Usage: mpirun -n <procs> %s <input.bmp> <output.bmp> <mask_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    char *input_path = argv[1];
    char *output_path = argv[2];
    int N = atoi(argv[3]);
    if ((N & 1) == 0 || N < 1) {
        if (rank == 0) fprintf(stderr, "Mask size must be a positive odd integer\n");
        MPI_Finalize();
        return 1;
    }
    int half = N / 2;

    // Create MPI datatype for pixel
    MPI_Datatype MPI_PIXEL;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_UINT8_T, MPI_UINT8_T, MPI_UINT8_T};
    MPI_Aint offsets[3] = {offsetof(pixel, b), offsetof(pixel, g), offsetof(pixel, r)};
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);

    bmpHeader header;
    int width = 0, height = 0;
    pixel *full_original = NULL;

    if (rank == 0) {
        FILE *input_fp = fopen(input_path, "rb");
        if (input_fp == NULL) {
            perror("Error opening input image");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fread(&header, sizeof(bmpHeader), 1, input_fp) != 1) {
            fprintf(stderr, "Error reading header\n");
            fclose(input_fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (header.type != 0x4D42 || header.bits != 24 || header.compression != 0 || header.height <= 0) {
            fprintf(stderr, "Unsupported BMP format\n");
            fclose(input_fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        width = header.width;
        height = header.height;
        full_original = (pixel *)malloc(width * height * sizeof(pixel));
        if (full_original == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(input_fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int padding = (4 - (width * 3 % 4)) % 4;
        fseek(input_fp, header.offset, SEEK_SET);
        for (int y = 0; y < height; y++) {
            fread(full_original + y * width, sizeof(pixel), width, input_fp);
            fseek(input_fp, padding, SEEK_CUR);
        }
        fclose(input_fp);
    }

    // Broadcast dimensions and header
    MPI_Bcast(&header, sizeof(bmpHeader), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Contiguous row type
    MPI_Datatype MPI_PIXEL_ROW;
    MPI_Type_contiguous(width, MPI_PIXEL, &MPI_PIXEL_ROW);
    MPI_Type_commit(&MPI_PIXEL_ROW);

    // Compute local heights, sendcounts, displs (in rows)
    int *local_heights = (int *)malloc(size * sizeof(int));
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    if (rank == 0) {
        int base = height / size;
        int rem = height % size;
        int offset = 0;
        for (int i = 0; i < size; i++) {
            local_heights[i] = base + (i < rem ? 1 : 0);
            sendcounts[i] = local_heights[i];
            displs[i] = offset;
            offset += local_heights[i];
        }
    }
    MPI_Bcast(local_heights, size, MPI_INT, 0, MPI_COMM_WORLD);
    int local_height = local_heights[rank];

    // Allocate local 1D
    pixel *local_original = (pixel *)malloc(local_height * width * sizeof(pixel));
    pixel *local_working = (pixel *)malloc(local_height * width * sizeof(pixel));
    if (local_original == NULL || local_working == NULL) {
        if (rank == 0) fprintf(stderr, "Local memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter with Scatterv (using row type)
    MPI_Scatterv(full_original, sendcounts, displs, MPI_PIXEL_ROW, local_original, local_height, MPI_PIXEL_ROW, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(full_original);
        full_original = NULL;
    }

    /* 
        Note: Excludes the initial data distribution cost, but includes the halo-exchange
        communication overhead since it is a dependency of the median_filter operation. 
    */

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Allocate 1D halo
    pixel *local_input_with_halo = (pixel *)malloc((local_height + 2 * half) * width * sizeof(pixel));
    if (local_input_with_halo == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Copy center
    for (int y = 0; y < local_height; y++) {
        memcpy(local_input_with_halo + (y + half) * width, local_original + y * width, width * sizeof(pixel));
    }

    // Non-blocking halo exchange vertical (use row type):
    MPI_Request *reqs = (MPI_Request *)malloc(4 * half * sizeof(MPI_Request));
    if (reqs == NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_reqs = 0;
    if (rank > 0) {
        for (int h = 0; h < half; h++) {
            MPI_Isend(local_input_with_halo + (half + h) * width, 1, MPI_PIXEL_ROW, rank - 1, 0, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(local_input_with_halo + h * width, 1, MPI_PIXEL_ROW, rank - 1, 1, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }
    }
    if (rank < size - 1) {
        for (int h = 0; h < half; h++) {
            MPI_Isend(local_input_with_halo + (local_height + h) * width, 1, MPI_PIXEL_ROW, rank + 1, 1, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(local_input_with_halo + (local_height + half + h) * width, 1, MPI_PIXEL_ROW, rank + 1, 0, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }
    }
    MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
    free(reqs);

    // Fill edge halos with clamping
    if (rank == 0) {
        for (int h = 0; h < half; h++) {
            memcpy(local_input_with_halo + h * width, local_input_with_halo + half * width, width * sizeof(pixel));
        }
    }
    if (rank == size - 1) {
        for (int h = 0; h < half; h++) {
            memcpy(local_input_with_halo + (local_height + half + h) * width, local_input_with_halo + (local_height + half - 1) * width, width * sizeof(pixel));
        }
    }

    /*
        Timing starts before halo exchange to ensure a fair comparison: 
        from this point, all distributed processes operate mostly on local memory.

        // MPI_Barrier(MPI_COMM_WORLD);
        // double start = MPI_Wtime();
    */

    // Apply median (1D version)
    apply_median_filter(local_input_with_halo, local_working, local_height, width, N, rank, size, MPI_COMM_WORLD);

    free(local_input_with_halo);

    free(local_original);
    local_original = NULL;

    // Grayscale (1D)
    convert_to_grayscale_inplace(local_working, local_height, width);

    // Histogram equalization (1D)
    equalize_histogram(local_working, local_height, width, rank, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Execution time: %.6f seconds\n", end - start);
    }

    // Gather with Gatherv
    pixel *full_working = NULL;
    if (rank == 0) {
        full_working = (pixel *)malloc(height * width * sizeof(pixel));
    }
    MPI_Gatherv(local_working, local_height, MPI_PIXEL_ROW, full_working, sendcounts, displs, MPI_PIXEL_ROW, 0, MPI_COMM_WORLD);

    // Rank 0 writes output
    if (rank == 0) {
        FILE *output_fp = fopen(output_path, "wb");
        if (output_fp == NULL) {
            perror("Error opening output image");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fwrite(&header, sizeof(bmpHeader), 1, output_fp) != 1) {
            fprintf(stderr, "Error writing header\n");
            fclose(output_fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int padding = (4 - (width * 3 % 4)) % 4;
        fseek(output_fp, header.offset, SEEK_SET);
        for (int y = 0; y < height; y++) {
            fwrite(full_working + y * width, sizeof(pixel), width, output_fp);
            for (int p = 0; p < padding; p++) fputc(0, output_fp);
        }
        fclose(output_fp);
        free(full_working);
    }

    free(local_working);
    free(local_heights);
    free(sendcounts);
    free(displs);
    MPI_Type_free(&MPI_PIXEL_ROW);
    MPI_Type_free(&MPI_PIXEL);
    MPI_Finalize();
    return 0;
}