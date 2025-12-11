# Histogram Equalization: Sequential, OpenMP, and MPI Implementations

## Project Overview

The histogram equalization process in a noisy image begins with a smoothing step aimed at reducing abrupt intensity fluctuations caused by noise, particularly salt-and-pepper noise. In this stage, smoothing filters — most commonly the **Median Filter** — are applied to attenuate noise while preserving edges and important structural details. The median filter replaces each pixel with the median of the values in its neighborhood, defined by a square mask (e.g., 3×3, 5×5, 7×7). The mask values are sorted, and the central value is assigned to the corresponding pixel.

After smoothing, the image is converted to **Grayscale**, since histogram equalization is defined for single-channel intensity images. This conversion is performed using a weighted combination of the RGB channels, assigning greater weight to green, followed by red and blue.

Next, histogram equalization is performed to enhance contrast. First, the histogram of intensity levels (0–255) is computed. Then, its **Cumulative Distribution Function (CDF)** is obtained and normalized to the 0–255 range, generating a mapping that reassigns each original intensity to a new value. Applying this mapping to all pixels produces the equalized image.

<p style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <img src="resources/img_before.png" alt="Before" style="height:561px;" />
  <img src="resources/img_after.png" alt="After" style="height:561px;" />
</p>

<p align="center">
  Comparison of the image **Before and After** applying the Histogram Equalization.
</p>

---

## Core workflow

1. Apply N×N **Median Filter** (histogram‑based median per pixel).  
2. Convert filtered image to **Grayscale** with weights ***R=0.299, G=0.587, B=0.114***.  
3. Build 256‑bin histogram, compute **CDF**, normalize to [0..255], remap pixels.

---

## BMP 24‑Bit Format

The 24-bit per pixel (24bpp) format supports 16,777,216 distinct colors and stores 1 pixel value per 3 bytes. Each pixel value defines the red, green, and blue samples of the pixel (8.8.8.0.0 in RGBAX notation). Specifically, in the order: blue, green, and red (8 bits per each sample) [Microsoft Help and Support](https://learn.microsoft.com/en-us/previous-versions/ms969901(v=msdn.10)?redirectedfrom=MSDN).

---
## Usage & Examples

### **1. Sequential**
- **Compile:**  
  ```bash
  gcc 1.Sequential.c -o 1.Sequential
  ```
- **Run:**  
  ```bash
  ./1.Sequential <input.bmp> <output.bmp> <mask_size>
  ```
- **Example:**  
  ```bash
  ./1.Sequential img.bmp new_img.bmp 3
  ```

---
### **2. MPI (Distributed)**
- **Compile:**  
  ```bash
  mpicc 2.MPI.c -o 2.MPI
  ```
- **Run:**  
  ```bash
  mpirun -n <procs> 2.MPI <input.bmp> <output.bmp> <mask_size>
  ```
- **Example:**  
  ```bash
  mpirun -n 2 2.MPI img.bmp new_img.bmp 5
  ```
> *Use `--oversubscribe` if needed.*

---
### **3. OpenMP (Shared Memory)**
- **Compile:**  
  ```bash
  gcc 3.OpenMP.c -o 3.OpenMP -fopenmp
  ```
- **Run:**  
  ```bash
  ./3.OpenMP <input.bmp> <output.bmp> <mask_size> [num_threads]
  ```
- **Example:**  
  ```bash
  ./3.OpenMP img.bmp new_img.bmp 3 2
  ```
---
## Documentation for Median Function

A histogram‑based median for `uint8_t` domain used in the project:

```c
static inline uint8_t median_uint8_hist(const uint8_t *values, int size) {
    long hist[256] = {0}; // For large masks, better to reuse hist[]
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
```

**Notes:**  
- This avoids sorting the neighborhood values and leverages the fixed 0–255 range.  
- Complexity per pixel: O(mask_size + 256) (dominant cost: building hist).  
- For large masks reuse of `hist[]` across pixels is recommended to reduce allocation/initialization overhead.

---
## OpenMP vs MPI — concise comparison (observed)

- **OpenMP (shared memory)**: low‑overhead parallelism, simple pragmas, ideal for single‑node multicore use. May suffer from thread contention and false sharing in some reductions/updates. Easier to implement.
- **MPI (message passing)**: designed for clusters but runs efficiently locally (optimized shared‑memory transports). Processes can provide better cache isolation and avoid some thread‑level contention, sometimes yielding better local performance. Communication cost is negligible for single‑node experiments, but rises for multi‑node runs.
---

## Performance metrics

- **Speedup** = time(1 resource) / time(N resources)  
- **Efficiency** = Speedup / N  (0 .. 1)  

The experimental baseline is defined by the **execution times of the sequential implementation** (`1.Sequential`). All performance comparisons for OpenMP and MPI are measured relative to this baseline. Sequential timings (averaged over 5 executions) for different mask sizes are as follows:

| Mask Size | Average Time (s)     | Raw Times (s)                                    |
|-----------|----------------------|--------------------------------------------------|
| 3         | 3.202684             | 3.250705, 3.507560, 3.068295, 3.058426, 3.128436 |
| 5         | 4.801925             | 4.829399, 4.980449, 4.708634, 4.716927, 4.774217 |
| 7         | 7.197326             | 7.315207, 7.082426, 7.150358, 7.374483, 7.064157 |

---
## Environment & Test Setup

- **OS:** Ubuntu 24.04 (WSL)  
- **Compiler & Libraries:** GCC 13.3.0, Open MPI 4.1.6, OpenMP 201511  
- **Machine:** Intel® Core™ i5‑2400 (4 physical cores, no hyperthreading), 8 GB RAM (4 GB allocated to WSL)  
- **Test Image:** 3000 × 2000 BMP, 24‑bit  
- **Mask Sizes Tested:** 3×3, 5×5, 7×7  
- **Measurements:** Each timing is the average of 5 independent runs  

> Note: Local MPI transport reduces communication overhead; results reflect single-node behavior.

---
## Results (visual summary)

![Speedup & Efficiency per Core (for each mask)](/resources/graph_comparison.png)

---

## License & Attribution
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
---