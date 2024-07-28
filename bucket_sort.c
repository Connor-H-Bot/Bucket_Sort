// Bucket Sort program for HPPC project by Connor Harris.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <limits.h>

// Structures

// Globals
unsigned long long int problemsize;
unsigned int thread_count;
omp_lock_t lock;

// Functions
double get_wall_seconds();
void *emalloc(size_t size);
void *erealloc(unsigned int *array, size_t new_size);
int checkresults(unsigned int *problem_array);
int cpu_cachesize();
void shuffle_array(unsigned int *problem_array, size_t size);
int compare(const void *a, const void *b);
int compare_large(const void *a, const void *b);
void bucket_sort(unsigned int *problem_array);
void bucket_sort_large(unsigned long long int *problem_array);
void basic_load_balance(unsigned int *problem_array, unsigned int **buckets);
void uniform_problem();
void normal_problem();
void exponential_problem();

// Timing function from HPPC labs
double get_wall_seconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
    return seconds;
}

/*
    Malloc and check integer array function from user @Dave at:
    https://stackoverflow.com/questions/7940279/should-we-check-if-memory-allocations-fail
*/
void *emalloc(size_t size)
{
    void *v = (unsigned int *)malloc(size * sizeof(unsigned int));
    if (!v)
    {
        fprintf(stderr, "out of mem\n");
        exit(EXIT_FAILURE);
    }
    return v;
}

// Realloc using parts of emalloc() code
void *erealloc(unsigned int *array, size_t new_size)
{
    //void *v = (unsigned int *)malloc(size * sizeof(unsigned int));
    void *v = realloc(array, (new_size * sizeof(unsigned int)));
    //buckets[j] = realloc(buckets[j], (buckets[j][0] + 1) * sizeof(unsigned int));
    if (!v)
    {
        fprintf(stderr, "ERROR: Reallocation of memory failed\n");
        exit(EXIT_FAILURE);
    }
    return v;
}


// Iterate through and make sure results are ascending
int checkresults(unsigned int *problem_array)
{
    for (unsigned int i = 0; i < (problemsize - 1); i++)
    {
        if (problem_array[i] > problem_array[i + 1])
        {
            printf("Array is not in ascending order. Sorting failed.\n");
            return 0;
        }
    }
    printf("Order of elements are OK.\n");
    return 1;
}

// Returns the size in MiB of the largest cache on the systems processor. Only checks L2 and L3 cache and compares the two.
int cpu_cachesize()
{
    FILE *fp;
    char path[1035];
    int cache_sizes[2] = {0, 0};

    fp = popen("lscpu | grep -E '^L[23]' | awk -F': ' '{print $2}' | awk '{print $1}' | paste -sd ' ' -", "r");
    if (fp == NULL)
    {
        printf("Failed to run command\n");
        exit(1);
    }

    if (fgets(path, sizeof(path) - 1, fp) != NULL)
    {
        sscanf(path, "%d %d", &cache_sizes[0], &cache_sizes[1]);
    }
    pclose(fp);

    return (cache_sizes[0] < cache_sizes[1]) ? cache_sizes[1] : cache_sizes[0];
}

/*
    Fisher-Yates shuffle algorithm, modified version of @Roland Illig's answer on:
    stackoverflow.com/questions/3343797/is-this-c-implementation-of-fisher-yates-shuffle-correct
*/
void shuffle_array(unsigned int *problem_array, size_t size)
{
    srand(time(0));
    for (uint i = size - 1; i > 0; i--)
    {
        unsigned int j = rand() % (i + 1);
        unsigned int temp = problem_array[i];
        problem_array[i] = problem_array[j];
        problem_array[j] = temp;
    }
}

// Integer comparison functions for quicksort, from tutorialspoint:
// https://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm
inline int compare(const void *a, const void *b)
{
    return (*(unsigned int *)a - *(unsigned int *)b);
}

// Integer comparison modified to handle unsigned long long ints, by @2501
// https://stackoverflow.com/questions/36681906/c-qsort-doesnt-seem-to-work-with-unsigned-long
inline int compare_large(const void *a, const void *b)
{ // needs optimisation
    unsigned long long int arg1 = *(const unsigned long long int *)a;
    unsigned long long int arg2 = *(const unsigned long long int *)b;

    if (arg1 < arg2)
        return -1;
    if (arg1 > arg2)
        return 1;
    return 0;
}

// Bucket sorting algorithm for int (32 bit) sized problems
void bucket_sort(unsigned int *problem_array)
{
    unsigned int bin_capacity = problemsize / thread_count;
    omp_init_lock(&lock);
    omp_set_num_threads(thread_count);
    //unsigned int *buckets = emalloc(thread_count);
    unsigned int **buckets = (unsigned int**)malloc(thread_count * sizeof(unsigned int*)); // this should be checked for safety. Make a function for it
    basic_load_balance(problem_array, buckets);
    // create array of pointers, with each pointer pointing to the first element of a new bucket

#pragma omp parallel shared(lock)
    {
        int thread_id = omp_get_thread_num();
        // unsigned int start_index = thread_id * bin_capacity;
        unsigned int end_index = (thread_id == thread_count - 1) ? problemsize : (thread_id + 1) * bin_capacity;
        qsort(problem_array, end_index, sizeof(int), compare);
    }
    // free buckets and all memory associated with it
    omp_destroy_lock(&lock); 
}

// Bucket sort algorithm for ULLONG_MAX (64 bit) sized problems
void bucket_sort_large(unsigned long long int *problem_array)
{
    unsigned long long int bin_capacity = problemsize / thread_count;
    omp_init_lock(&lock);
    omp_set_num_threads(thread_count);

#pragma omp parallel shared(lock)
    {
        int thread_id = omp_get_thread_num();
        // unsigned int start_index = thread_id * bin_capacity;
        unsigned long long int end_index = (thread_id == thread_count - 1) ? problemsize : (thread_id + 1) * bin_capacity;
        // load balance
        qsort(problem_array, end_index, sizeof(unsigned long long int), compare_large);
    }
    // free buckets and all memory associated with it
    omp_destroy_lock(&lock); 
}

/*
    Basic load balance:
        Break down the problem_array into N buckets, and return an array
        containing pointers to each bucket.
        Basic implementation does not know the highest value, so it 
        divides up the buckets assuming even distribution of 0 -> UINT_MAX integers.
*/
void basic_load_balance(unsigned int *problem_array, unsigned int **buckets)
{
    // Initialize each pointer in the buckets array to NULL
    for (unsigned int i = 0; i < thread_count; i++) {
        buckets[i] = emalloc(1); // set the length of each bucket to 1
        buckets[i][0] = 1; // and set its first element to 1 (length of the array)
    }
    // divide buckets into ranges (first element = length of array, so start at 1)
    // move backwards through the problem array and for each element:
        // add the element to the bucket via realloc (do function for this)
        // reduce the memory allocated for problem_array by 1
    unsigned int bucket_size = UINT32_MAX / thread_count;
    unsigned int *bucket_limits = emalloc(thread_count);
    for (unsigned int i = 0; i < thread_count; i ++){
        bucket_limits[i] = ((i == thread_count - 1) ? UINT32_MAX : (i + 1) * bucket_size);
    }

    for (unsigned int i = (problemsize - 1); i > 0; i--){
        //loop through the array from the back and start deallocating
        unsigned int temp = problem_array[i]; 
        for (unsigned int j = 0; j < thread_count; j++){
            if (temp <= bucket_limits[j]){
                buckets[j] = erealloc(buckets[j], (buckets[j][0] + 1)); // extend the length of a bucket
                unsigned int bucket_index = buckets[j][0]; 
                buckets[j][bucket_index] = temp; // add the value to the bucket
                buckets[j][0] += 1; // increase the buckets length (indicated by this variable)
                problem_array = erealloc(problem_array, i); // reduce the size of the array by removing its last index
                // maybe this is possible with memcpy (and faster)
            }
        }
    }

    for (int i = 0; i < thread_count; i++){
        printf("Bucket: %d. Length: %u\n", i, buckets[i][0]);
    }
    
    printf("Closing the program to make sure everything is done with\n");
    free(bucket_limits);
    free(problem_array);
    free(buckets);
    exit(1);
    
}

// Uniform distribution problem set
void uniform_problem()
{
    printf("\nUniform distribution problem\n");
    unsigned int cachesize = cpu_cachesize();
    unsigned int uints_max = cachesize * 26214; // 26,214 = 80% of (1MiB / sizeof(uint_MAX))
    if (problemsize > uints_max)
    {
        printf("WARNING: %llu is a larger problem size than the guarenteed cache limit of %d UINT_MAX's. \nThis may result in poor performance. \n", problemsize, uints_max);
    }

    unsigned int *problem_array = emalloc(problemsize);
    for (uint i = 0; i < problemsize; i++)
    {
        problem_array[i] = i;
    }
    shuffle_array(problem_array, problemsize);

    double timer = get_wall_seconds();
    bucket_sort(problem_array);
    timer = get_wall_seconds() - timer;
    printf("\nElements: %llu\nThreads: %d\nTime taken: %lf\n\n", problemsize, thread_count, timer);

    checkresults(problem_array);
    free(problem_array);
}

// Random distribution problem set
void normal_problem()
{
    printf("\nNormal randomisation problem\n");
    unsigned int cachesize = cpu_cachesize();
    unsigned int uints_max = cachesize * 26214; // 26,214 = 80% of (1MiB / sizeof(uint_MAX))
    if (problemsize > uints_max)
    {
        printf("WARNING: %llu is a larger problem size than the guarenteed cache limit of %d UINT_MAX's. \nThis may result in poor performance. \n", problemsize, uints_max);
    }

    unsigned int *problem_array = emalloc(problemsize);

    // populate the problem set with randomly generated ints
    srand(time(0));
    for (unsigned int i = 0; i < problemsize; i++)
    {
        problem_array[i] = rand();
    }

    double timer = get_wall_seconds();
    bucket_sort(problem_array);
    timer = get_wall_seconds() - timer;
    printf("\nElements: %llu\nThreads: %d\nTime taken: %lf\n\n", problemsize, thread_count, timer);

    checkresults(problem_array);
    free(problem_array);
}

// Exponential numbers problem set
void exponential_problem()
{
    printf("\nExponential distribution problem\n");
    int cachesize = cpu_cachesize();
    int ulongints_max = cachesize * 13100; // 13,100 = 80% of (1MiB / sizeof(ulong_MAX)
    if (problemsize > ulongints_max)
    {
        printf("WARNING: %llu is a larger problem size than the guarenteed cache limit of %d ULLONG_MAX's. \nThis may result in poor performance. \n", problemsize, ulongints_max);
    }

    unsigned long long *exponential_array = (unsigned long long *)malloc(problemsize * sizeof(unsigned long long));
    if (exponential_array == NULL)
    {
        perror("Can't allocate exponent array\n");
        exit(EXIT_FAILURE);
    }

    // populate the array with a randomly selected 2^N. (0 <= N <= 64), as the largest possible int is 2^64
    for (unsigned long long int i = 0; i < problemsize; i++)
    {
        unsigned long long int random_exponent = rand() % 65;
        exponential_array[i] = ((__uintmax_t)1 << random_exponent);
    }

    double timer = get_wall_seconds();
    bucket_sort_large(exponential_array);
    timer = get_wall_seconds() - timer;
    printf("\nElements: %llu\nThreads: %d\nTime taken: %lf\n\n", problemsize, thread_count, timer);

    // Check results
    for (int i = 0; i < (problemsize - 1); i++)
    {
        if (exponential_array[i] > exponential_array[i + 1])
        {
            printf("Exponential array is not in ascending order. Sorting failed.\n");
            free(exponential_array);
            return;
        }
    }
    printf("Order of elements are OK.\n");
    free(exponential_array);
}

int main(int argc, char const *argv[])
{
    if (argc != 4)
    {
        printf("Error: File requires 3 input parameters. Useage: \n");
        printf("./bucket_sort problem_size thread_count problem_options(u, n, e)\n");
        return 1;
    }
    problemsize = atoi(argv[1]);
    thread_count = atoi(argv[2]);
    char problem_type = argv[3][0];
    if ((problemsize < 1) || (thread_count < 1))
    {
        printf("ERROR: bucket_sort requires positive numbers as problemsize and thread count input. Input example: \n");
        printf("\t./bucket_sort 1000 8 u\n");
        return 1;
    }

    switch (problem_type)
    {
    case 'u':
        uniform_problem();
        break;
    case 'n':
        normal_problem();
        break;
    case 'e':
        exponential_problem();
        break;
    default:
        printf("ERROR: '%c' is not a valid problem type. Please enter:\n", problem_type);
        printf("\t u: uniform distribution of integers from 0 - N, where N = problem_size. \n");
        printf("\t n: normal randomised distribution of integers. A problem_size array of random integers between 0 - uint_max (4 billion) is created. \n");
        printf("\t e: exponential distribution. A problem_size array where every element is 2^N, where n ranges from 0-64 (2^64 = ULLONG_MAX). \n");
        break;
    }

    return 0;
}
