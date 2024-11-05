#ifndef UTIL_H
#define UTIL_H

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
        exit(code);
   }
}

struct event_pair {
  cudaEvent_t start;
  cudaEvent_t end;
};


inline void check_launch(const char *kernel_name) {
  cudaDeviceSynchronize();
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert in kernel launch %s: %s\n", kernel_name, cudaGetErrorString(code));
    exit(1);
  }
}


inline void start_timer(event_pair *p) {
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
  cudaEventSynchronize(p->start);
}


inline void stop_timer(event_pair * p, const char *kernel_name) {
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  printf("%s took %.1f ms\n",kernel_name, elapsed_time);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
}

inline bool AlmostEqual2sComplement(float A, float B, int maxUlps) {
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    // assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    int aInt = *(int*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;
    // Make bInt lexicographically ordered as a twos-complement int
    int bInt = *(int*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;
    int intDiff = abs(aInt - bInt);
    if (intDiff <= maxUlps)
        return true;
    return false;
}

#endif