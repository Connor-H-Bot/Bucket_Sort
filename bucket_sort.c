// Bucket Sort program for HPPC project by Connor Harris.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <string.h>

// Globals
unsigned long long int problemsize;
unsigned int thread_count;

// Functions
double get_wall_seconds();
void *emalloc(size_t size);
void *emalloc_large(size_t size);
void *erealloc(unsigned int *array, size_t new_size);
void *erealloc_large(unsigned long long int *array, size_t new_size);
int checkresults(unsigned int *problem_array);
int cpu_cachesize();
void shuffle_array(unsigned int *problem_array, size_t size);
int compare(const void *a, const void *b);
int compare_large(const void *a, const void *b);
void *bucket_sort(unsigned int *problem_array);
void *bucket_sort_large(unsigned long long int *problem_array);
void basic_load_balance(unsigned int *problem_array, unsigned int **buckets);
void basic_load_balance_large(unsigned long long int *problem_array, unsigned long long int **buckets);
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

// emalloc for 64 bit integers
void *emalloc_large(size_t size)
{
    void *v = (unsigned long long int *)malloc(size * sizeof(unsigned long long int));
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
    // void *v = (unsigned int *)malloc(size * sizeof(unsigned int));
    void *v = realloc(array, (new_size * sizeof(unsigned int)));
    // buckets[j] = realloc(buckets[j], (buckets[j][0] + 1) * sizeof(unsigned int));
    if (!v)
    {
        fprintf(stderr, "ERROR: Reallocation of memory failed\n");
        exit(EXIT_FAILURE);
    }
    return v;
}

// Realloc using parts of emalloc() code for 64 bit integers
void *erealloc_large(unsigned long long int *array, size_t new_size)
{
    // void *v = (unsigned int *)malloc(size * sizeof(unsigned int));
    void *v = realloc(array, (new_size * sizeof(unsigned long long int)));
    // buckets[j] = realloc(buckets[j], (buckets[j][0] + 1) * sizeof(unsigned int));
    if (!v)
    {
        fprintf(stderr, "ERROR: Reallocation of memory for large ints failed\n");
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
            printf("Array is not in ascending order. Sorting failed at index: %u\n", i);
            for (unsigned int j = i; j < (i + 10); j++)
            {
                printf("Next ints: %u\n", problem_array[j]);
            }
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
    fp = NULL;
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
// And modified with inspiration from the tutorialspoint version
// https://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm
inline int compare_large(const void *a, const void *b)
{
    unsigned long long int arg1 = *(const unsigned long long int *)a;
    unsigned long long int arg2 = *(const unsigned long long int *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

// Bucket sorting algorithm for int (32 bit) sized problems
void *bucket_sort(unsigned int *problem_array)
{
    omp_set_num_threads(thread_count);

    unsigned int **buckets = (unsigned int **)malloc(thread_count * sizeof(unsigned int *));
    if (buckets == NULL)
    {
        fprintf(stderr, "Error: Memory allocation for buckets failed.\n");
        exit(EXIT_FAILURE);
    }
    basic_load_balance(problem_array, buckets);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int end_index = buckets[thread_id][0];
        qsort(buckets[thread_id] + 1, end_index - 1, sizeof(unsigned int), compare); // Sorting starts from index 1
#pragma omp barrier
    }

    // merge all buckets together
    unsigned int *sorted_array = emalloc(1 * sizeof(unsigned int));
    unsigned int array_index = 0;
    unsigned int temp;

    for (unsigned int i = 0; i < thread_count; i++)
    {
        temp = buckets[i][0];
        if (temp > 1)
        {
            sorted_array = erealloc(sorted_array, (array_index + temp - 1) * sizeof(unsigned int));
            memcpy(&sorted_array[array_index], &buckets[i][1], (temp - 1) * sizeof(unsigned int));
            array_index += temp - 1;
        }
        free(buckets[i]);
    }
    free(buckets);

    return sorted_array;
}

// Bucket sort algorithm for 64 bit sized problems
void *bucket_sort_large(unsigned long long int *problem_array)
{
    // unsigned long long int bin_capacity = problemsize / thread_count;
    omp_set_num_threads(thread_count);
    unsigned long long int **buckets = (unsigned long long int **)malloc(thread_count * sizeof(unsigned long long int *));
    if (buckets == NULL)
    {
        fprintf(stderr, "Error: Memory allocation for buckets failed.\n");
        exit(EXIT_FAILURE);
    }
    basic_load_balance_large(problem_array, buckets);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned long long int end_index = buckets[thread_id][0];
        qsort(buckets[thread_id] + 1, end_index - 1, sizeof(unsigned long long int), compare_large);
#pragma omp barrier
    }

    // merge all buckets together
    unsigned long long int *sorted_array = emalloc_large(1 * sizeof(unsigned long long int));
    unsigned long long int array_index = 0;
    unsigned long long int temp;

    for (unsigned int i = 0; i < thread_count; i++)
    {
        temp = buckets[i][0];
        if (temp > 1)
        {
            sorted_array = erealloc_large(sorted_array, (array_index + temp - 1) * sizeof(unsigned long long int));
            memcpy(&sorted_array[array_index], &buckets[i][1], (temp - 1) * sizeof(unsigned long long int));
            array_index += temp - 1;
        }
        free(buckets[i]);
    }
    free(buckets);

    return sorted_array;
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
    for (unsigned int i = 0; i < thread_count; i++)
    {
        buckets[i] = emalloc(2 * sizeof(unsigned int)); // Allocate space for at least one element plus the size
        buckets[i][0] = 1;                              // Set the length of the bucket to 1 initially
    }

    unsigned int highest_num = 0;
    omp_set_num_threads(thread_count);
    unsigned int *highest_numbers = emalloc(thread_count);
    unsigned int problem_size_per_thread = problemsize / thread_count;

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int start_index = thread_id * problem_size_per_thread;
        unsigned int end_index = (thread_id == thread_count - 1) ? problemsize : (thread_id + 1) * problem_size_per_thread;
        unsigned int highest_num_thread = 0;
        for (unsigned int i = start_index; i < end_index; i++)
        {
            if (problem_array[i] > highest_num_thread)
            {
                highest_num_thread = problem_array[i];
            }
        }
        highest_numbers[thread_id] = highest_num_thread;
#pragma omp barrier
    }

    // select the highest number from the list
    for (int i = 0; i < thread_count; i++)
    {
        if (highest_numbers[i] > highest_num)
        {
            highest_num = highest_numbers[i];
        }
    }
    free(highest_numbers);

    unsigned int bucket_size = highest_num / thread_count;
    unsigned int *bucket_limits = emalloc(thread_count * sizeof(unsigned int));
    for (unsigned int i = 0; i < thread_count; i++)
    {
        bucket_limits[i] = ((i == thread_count - 1) ? highest_num : (i + 1) * bucket_size);
    }

    // Move elements from the problem array into the bucket
    unsigned int temp;
    unsigned int bucket_index;
    for (unsigned int i = 0; i < problemsize; i++)
    {
        temp = problem_array[i];
        for (unsigned int j = 0; j < thread_count; j++)
        {
            if (temp <= bucket_limits[j])
            {
                bucket_index = buckets[j][0];
                buckets[j] = erealloc(buckets[j], (bucket_index + 1) * sizeof(unsigned int)); // Extend the bucket by one element
                buckets[j][bucket_index] = temp;                                              // Add the value to the bucket
                buckets[j][0] += 1;                                                           // Increase the bucket's length
                break;
            }
        }
    }

    // second pass to check bucket distribution and shift them if they are unchanged.
    /*
        move through buckets and decide what the inbalance is (buckets should be +- 20% of the bucket size)
        identify buckets that are, and buckets that are not in the range
    */
    for (unsigned int j = 0; j < thread_count; j++)
    {
        printf("Bucket: %d Bucket limits: %d\n", j, buckets[j][0]);
    }

    free(bucket_limits);
    free(problem_array);
}

/*
    Basic load balance large:
        Break down the problem_array into N buckets, and return an array
        containing pointers to each bucket.
        Basic implementation does not know the highest value, so it
        divides up the buckets assuming even distribution of 0 -> UINT64_MAX integers.
        Same as the basic load balancing function, with functionality for 64 bit integers
*/
void basic_load_balance_large(unsigned long long int *problem_array, unsigned long long int **buckets)
{
    for (unsigned long long int i = 0; i < thread_count; i++)
    {
        buckets[i] = emalloc_large(2 * sizeof(unsigned long long int)); // Allocate space for at least one element plus the size
        buckets[i][0] = 1;                                              // Set the length of the bucket to 1 initially
    }

    unsigned long long int highest_num = 0;
    omp_set_num_threads(thread_count);
    // unsigned long long int highest_numbers[thread_count]; // for some reason this is generating a set but not used warning
    unsigned long long int *highest_numbers = emalloc_large(thread_count);
    unsigned long long int problem_size_per_thread = problemsize / thread_count;

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned long long int start_index = thread_id * problem_size_per_thread;
        unsigned long long int end_index = (thread_id == thread_count - 1) ? problemsize : (thread_id + 1) * problem_size_per_thread;
        unsigned long long int highest_num_thread = 0;
        for (unsigned long long int i = start_index; i < end_index; i++)
        {
            if (problem_array[i] > highest_num_thread)
            {
                highest_num_thread = problem_array[i];
            }
        }
        highest_numbers[thread_id] = highest_num_thread;
#pragma omp barrier
    }

    for (unsigned long long int i = 0; i < problemsize; i++)
    {
        if (problem_array[i] > highest_num)
        {
            highest_num = problem_array[i];
        }
    }
    unsigned long long int bucket_size = highest_num / thread_count;
    unsigned long long int *bucket_limits = emalloc_large(thread_count * sizeof(unsigned long long int));
    for (unsigned int i = 0; i < thread_count; i++)
    {
        bucket_limits[i] = ((i == thread_count - 1) ? highest_num : (i + 1) * bucket_size);
    }

    unsigned long long int temp;
    unsigned long long int bucket_index;

    for (unsigned long long int i = 0; i < problemsize; i++)
    {
        temp = problem_array[i];
        for (unsigned int j = 0; j < thread_count; j++)
        {
            if (temp <= bucket_limits[j])
            {
                bucket_index = buckets[j][0];
                buckets[j] = erealloc_large(buckets[j], (bucket_index + 1) * sizeof(unsigned long long int)); // Extend the bucket by one element
                buckets[j][bucket_index] = temp;                                                              // Add the value to the bucket
                buckets[j][0] += 1;                                                                           // Increase the bucket's length
                break;
            }
        }
    }

    for (unsigned int j = 0; j < thread_count; j++)
    {
        printf("Bucket: %d Bucket limits: %llu\n", j, buckets[j][0]);
    }

    // second pass to check bucket distribution and shift them if they are unchanged.
    /*
        move through buckets and decide what the inbalance is (buckets should be +- 20% of the bucket size)
        identify buckets that are, and buckets that are not in the range
        - count buckets which are empty
        - use it as the base for a load distribution (if this occurs, it is likely the problem is exponential)
        -
        count empty buckets
        count fullest bucket
        shift buckets not full to the end
        divide the biggest bucket accross small buckets
        programatically:
        get empty buckets
        shift fuller buckets towards the end
        have biggest bucket (first if exponential) divide capacity accross the X empty buckets (which should be placed next to it)
    */
    printf("\n\n");
    int empty_buckets = 0;
    for (int i = (thread_count - 1); i > 1; i--)
    {
        // move backwards through the list and once an empty bucket is found:
        // swap places with its nearest full bucket (make sure nearest bucket is not the first one)
        if (buckets[i][0] == 1) // empty bucket found
        {
            for (int j = (i - 1); j > 0; j--) // move backwards through the list to find the next full bucket to swap with
            {
                if (buckets[j][0] != 1) // found the next non-empty bucket
                {
                    unsigned long long int *temp = buckets[j]; // swap them
                    buckets[j] = buckets[i];
                    buckets[i] = temp;
                    empty_buckets++;
                    break; // exit the inner loop after swapping
                }
            }
        }
    }
    // now loop through the first bucket which is full?
    unsigned long long int highest_number = 0;
    for (unsigned long long int i = 0; i < (buckets[0][0] - 1); i++)
    {
        if (buckets[0][i] > highest_number)
        {
            highest_number = buckets[0][i];
        }
    }

    unsigned long long int chunk_size = highest_number / (empty_buckets + 1);
    unsigned long long int *bucket_limits_2 = emalloc_large((empty_buckets + 1) * sizeof(unsigned long long int));
    for (unsigned int i = 0; i < (empty_buckets + 1); i++)
    {
        bucket_limits_2[i] = ((i == empty_buckets) ? highest_num : (i + 1) * chunk_size);
    }
    // copy everything from b0 into a new bucket, make b0 0, and move everything back accross like the first implementation
    unsigned long long int *holding_bucket = emalloc_large((buckets[0][0]) * sizeof(unsigned long long int));
    memcpy(&holding_bucket, &buckets[0], (buckets[0][0]) * sizeof(unsigned int));
    buckets[i] = erealloc_large(buckets[i], (2 * sizeof(unsigned long long int))); // Allocate space for at least one element plus the size
    // erealloc_large(buckets[j], (bucket_index + 1) * sizeof(unsigned long long int)); 
    buckets[i][0] = 1;    

    for (unsigned long long int i = 1; i < buckets[0][0]; i++) // start going through the full bucket (1st)
    {
        // reallocate elements from bucket0 accross b0-bempty_buckets 
        temp = holding_bucket[i];
        for (unsigned int j = 0; j < empty_buckets; j++)
        {
            if (temp <= bucket_limits_2[j])
            {
                bucket_index = buckets[j][0];
                buckets[j] = erealloc_large(buckets[j], (bucket_index + 1) * sizeof(unsigned long long int)); // Extend the bucket by one element
                buckets[j][bucket_index] = temp;                                                              // Add the value to the bucket
                buckets[j][0] += 1;                                                                           // Increase the bucket's length
                break;
            }
        }
    }

    for (unsigned int j = 0; j < thread_count; j++)
    {
        printf("Bucket: %d Bucket limits: %llu\n", j, buckets[j][0]);
    }

    free(bucket_limits);
    free(problem_array);
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
    unsigned int *sorted_array = bucket_sort(problem_array);
    timer = get_wall_seconds() - timer;
    printf("\nElements: %llu\nThreads: %d\nTime taken: %lf\n\n", problemsize, thread_count, timer);

    checkresults(sorted_array);
    free(sorted_array);
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
    unsigned int *sorted_array = bucket_sort(problem_array);
    timer = get_wall_seconds() - timer;
    printf("\nElements: %llu\nThreads: %d\nTime taken: %lf\n\n", problemsize, thread_count, timer);

    checkresults(sorted_array);
    free(sorted_array);
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
    unsigned long long int *sorted_array = bucket_sort_large(exponential_array);
    timer = get_wall_seconds() - timer;
    printf("\nElements: %llu\nThreads: %d\nTime taken: %lf\n\n", problemsize, thread_count, timer);

    // Check results
    for (int i = 0; i < (problemsize - 1); i++)
    {
        if (sorted_array[i] > sorted_array[i + 1])
        {
            printf("Exponential array is not in ascending order. Sorting failed.\n");
            free(sorted_array);
            return;
        }
    }
    printf("Order of elements are OK.\n");
    free(sorted_array);
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
