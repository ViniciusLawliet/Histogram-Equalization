#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "bmp_structs.h"

// Allocates a linear pixel buffer representing a 2D image (row-major order)
pixel *allocate_matrix(int height, int width) {
    pixel *matrix = (pixel *)malloc(height * width * sizeof(pixel));
    if (matrix == NULL) {
        return NULL;
    }
    return matrix;
}

// Releases the memory allocated for the pixel buffer
void free_matrix(pixel *matrix, int height) {
    if (matrix == NULL) return;
    free(matrix);
}

// Function to read BMP image data into a matrix (bottom-up)
int read_image(FILE *fp, bmpHeader *header, pixel *matrix) {
    int width = header->width;
    int height = header->height;
    int padding = (4 - (width * 3 % 4)) % 4;

    fseek(fp, header->offset, SEEK_SET);
    for (int y = 0; y < height; y++) {
        fread(matrix + y * width, sizeof(pixel), width, fp);
        fseek(fp, padding, SEEK_CUR);
    }
    return 0;
}

// Function to write BMP image data from a matrix
int write_image(FILE *fp, bmpHeader *header, pixel *matrix) {
    int width = header->width;
    int height = header->height;
    int padding = (4 - (width * 3 % 4)) % 4;

    fseek(fp, header->offset, SEEK_SET);
    for (int y = 0; y < height; y++) {
        fwrite(matrix + y * width, sizeof(pixel), width, fp);
        for (int p = 0; p < padding; p++) {
            fputc(0, fp);
        }
    }
    return 0;
}

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

// Function to apply median filter (parallelized)
void apply_median_filter(pixel *input, pixel *output, int height, int width, int N) {
    int half = N / 2;
    int size = N * N;
    int med_idx = size / 2;

#pragma omp parallel
    {
        // Allocate per thread (reused for all pixels of this thread)
        uint8_t *r_values = (uint8_t *)malloc(size * sizeof(uint8_t));
        uint8_t *g_values = (uint8_t *)malloc(size * sizeof(uint8_t));
        uint8_t *b_values = (uint8_t *)malloc(size * sizeof(uint8_t));
        if (r_values == NULL || g_values == NULL || b_values == NULL) {
#pragma omp critical
            {
                fprintf(stderr, "Thread-local allocation failed\n");
            }
            // check later
        }

#pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = 0;
                for (int dy = -half; dy <= half; dy++) {
                    for (int dx = -half; dx <= half; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        ny = (ny < 0) ? 0 : (ny >= height ? height - 1 : ny);
                        nx = (nx < 0) ? 0 : (nx >= width ? width - 1 : nx);
                        r_values[idx] = input[ny * width + nx].r;
                        g_values[idx] = input[ny * width + nx].g;
                        b_values[idx] = input[ny * width + nx].b;
                        idx++;
                    }
                }
                output[y * width + x].r = median_uint8_hist(r_values, size);
                output[y * width + x].g = median_uint8_hist(g_values, size);
                output[y * width + x].b = median_uint8_hist(b_values, size);
            }
        }

        // Free per thread
        free(r_values);
        free(g_values);
        free(b_values);
    }
}

// Function to convert to grayscale (in-place, parallelized)
void convert_to_grayscale_inplace(pixel *matrix, int height, int width) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double val = 0.299 * matrix[y * width + x].r + 0.587 * matrix[y * width + x].g + 0.114 * matrix[y * width + x].b;
            uint8_t gray_val = (uint8_t)(val + 0.5);
            matrix[y * width + x].r = gray_val;
            matrix[y * width + x].g = gray_val;
            matrix[y * width + x].b = gray_val;
        }
    }
}

// Function to perform histogram equalization (in-place, parallelized where possible)
void equalize_histogram(pixel *matrix, int height, int width) {
    long hist[256] = {0};

    // Parallel histogram computation with reduction
#pragma omp parallel for collapse(2) reduction(+:hist[:256]) schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            hist[matrix[y * width + x].r]++;
        }
    }

    // Serial CDF computation (small, no benefit from parallel)
    long cdf[256] = {0};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    double total_pixels = (double)(cdf[255]);

    // Serial mapping (small)
    uint8_t map[256];
    for (int i = 0; i < 256; i++) {
        map[i] = (uint8_t)((cdf[i] * 255.0 / total_pixels) + 0.5);
    }

    // Parallel pixel transformation
#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t old_val = matrix[y * width + x].r;
            uint8_t new_val = map[old_val];
            matrix[y * width + x].r = new_val;
            matrix[y * width + x].g = new_val;
            matrix[y * width + x].b = new_val;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <input.bmp> <output.bmp> <mask_size> [num_threads]\n", argv[0]);
        return 1;
    }
    char *input_path = argv[1];
    char *output_path = argv[2];
    int N = atoi(argv[3]);

    int NThreads;
    if (argc == 5)
        NThreads = atoi(argv[4]);
    else
        NThreads = omp_get_max_threads();

    omp_set_num_threads(NThreads);

    if ((N & 1) == 0 || N < 1) {
        fprintf(stderr, "Mask size must be a positive odd integer\n");
        return 1;
    }

    FILE *input_fp = fopen(input_path, "rb");
    if (input_fp == NULL) {
        perror("Error opening input image");
        return 1;
    }

    bmpHeader header;
    if (fread(&header, sizeof(bmpHeader), 1, input_fp) != 1) {
        fprintf(stderr, "Error reading header\n");
        fclose(input_fp);
        return 1;
    }

    if (header.type != 0x4D42 || header.bits != 24 || header.compression != 0 || header.height <= 0) {
        fprintf(stderr, "Unsupported BMP format (must be 24-bit uncompressed with positive height)\n");
        fclose(input_fp);
        return 1;
    }

    int width = header.width;
    int height = header.height;

    pixel *original = allocate_matrix(height, width);
    pixel *working = allocate_matrix(height, width);
    if (original == NULL || working == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free_matrix(original, height);
        free_matrix(working, height);
        fclose(input_fp);
        return 1;
    }

    if (read_image(input_fp, &header, original)) {
        fprintf(stderr, "Error reading image data\n");
        goto cleanup;
    }
    fclose(input_fp);
    
    double start = omp_get_wtime();

    // Step 1: Apply median filter (parallel inside function)
    apply_median_filter(original, working, height, width, N);

    // Free original now that it's no longer needed
    free_matrix(original, height);
    original = NULL;

    // Step 2: Convert to grayscale (parallel inside function)
    convert_to_grayscale_inplace(working, height, width);

    // Step 3: Equalize histogram (parallel inside function)
    equalize_histogram(working, height, width);
    
    double end = omp_get_wtime();
    
    printf("Execution time: %.6f seconds\n", end - start);

    // Write the final image
    FILE *output_fp = fopen(output_path, "wb");
    if (output_fp == NULL) {
        perror("Error opening output image");
        goto cleanup;
    }

    if (fwrite(&header, sizeof(bmpHeader), 1, output_fp) != 1) {
        fprintf(stderr, "Error writing header\n");
        fclose(output_fp);
        goto cleanup;
    }

    if (write_image(output_fp, &header, working)) {
        fprintf(stderr, "Error writing image data\n");
        fclose(output_fp);
        goto cleanup;
    }
    fclose(output_fp);

// Cleanup section — ensures all allocated memory is freed before exit.
cleanup:
    free_matrix(original, height);
    free_matrix(working, height);
    return 0;
}