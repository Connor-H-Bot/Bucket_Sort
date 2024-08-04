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
        fprintf(stderr, "ERROR: No memory available for emalloc\n");
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
        fprintf(stderr, "ERROR: No memory available for emalloc_large\n");
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
            for (unsigned int j = i; j < ((i + 10 < (problemsize - 1)) ? i + 10 : i); j++)
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

// Integer comparison modified to handle unsigned long long ints, by @2501 (and modified with original compare() inspiration)
// https://stackoverflow.com/questions/36681906/c-qsort-doesnt-seem-to-work-with-unsigned-long
inline int compare_large(const void *a, const void *b)
{
    unsigned long long int arg1 = *(const unsigned long long int *)a;
    unsigned long long int arg2 = *(const unsigned long long int *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

// Bucket sorting algorithm for int (32 bit) sized problems
//     - Takes a single problem array and breaks it into distributed buckets equal to threads.
//     - Each thread/bucket sorts using quicksort, and merges the arrays back together.
//     - The sorted, merged array pointer is returned and everything else is freed.
void *bucket_sort(unsigned int *problem_array)
{
    // Initialise the empty buckets
    omp_set_num_threads(thread_count);
    unsigned int **buckets = (unsigned int **)malloc(thread_count * sizeof(unsigned int *));
    if (buckets == NULL)
    {
        fprintf(stderr, "Error: Memory allocation for buckets failed.\n");
        exit(EXIT_FAILURE);
    }

    // Break down the problem array into buckets using a basic load balancing implementation
    basic_load_balance(problem_array, buckets);

    // Sort each bucket in parallel
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned int end_index = buckets[thread_id][0];
        qsort((buckets[thread_id] + 1), (end_index - 1), sizeof(unsigned int), compare);
#pragma omp barrier
    }

    // Merge the buckets back together
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

// Bucket sorting algorithm for maxint (64 bit) sized problems
//     - Takes a single problem array and breaks it into distributed buckets equal to threads.
//     - Each thread/bucket sorts using quicksort, and merges the arrays back together.
//     - The sorted, merged array pointer is returned and everything else is freed.
void *bucket_sort_large(unsigned long long int *problem_array)
{
    // Initialise the empty buckets
    omp_set_num_threads(thread_count);
    unsigned long long int **buckets = (unsigned long long int **)malloc(thread_count * sizeof(unsigned long long int *));
    if (buckets == NULL)
    {
        fprintf(stderr, "Error: Memory allocation for buckets failed.\n");
        exit(EXIT_FAILURE);
    }

    // Break down the problem array into buckets using a basic load balancing implementation
    basic_load_balance_large(problem_array, buckets);

    // Sort each bucket in parallel
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned long long int end_index = buckets[thread_id][0];
        qsort(buckets[thread_id] + 1, end_index - 1, sizeof(unsigned long long int), compare_large);
#pragma omp barrier
    }

    // Merge the buckets back together
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
    Basic load balance (32 bit sized ints):
    -    Break down the problem_array into N buckets, and return an array containing pointers to each bucket.
    -    Basic implementation does not know the highest value, so it divides up the buckets
    -        assuming an even distribution of 0 -> UINT_MAX integers.
*/
void basic_load_balance(unsigned int *problem_array, unsigned int **buckets)
{
    // Create enough space in each bucket for two elements. First element is used to represent sizeof() the sub array
    for (unsigned int i = 0; i < thread_count; i++)
    {
        buckets[i] = emalloc(2 * sizeof(unsigned int));
        buckets[i][0] = 1;
    }

    unsigned int highest_num = 0; // Highest number in the problem set
    omp_set_num_threads(thread_count);
    unsigned int *highest_problemset_number_perthread = emalloc(thread_count);
    unsigned int problem_size_per_thread = problemsize / thread_count;

    // each thread finds the highest number in its zone of the unsorted numbers
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
        highest_problemset_number_perthread[thread_id] = highest_num_thread;
#pragma omp barrier
    }

    // select the highest number from the list
    for (int i = 0; i < thread_count; i++)
    {
        if (highest_problemset_number_perthread[i] > highest_num)
        {
            highest_num = highest_problemset_number_perthread[i];
        }
    }
    free(highest_problemset_number_perthread);

    // create buckets == threads and set the number range evenly for all buckets (first pass load balance)
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
                // When a suitable bucket is found, extend its length and incrememnt its counter
                bucket_index = buckets[j][0];
                buckets[j] = erealloc(buckets[j], (bucket_index + 1) * sizeof(unsigned int));
                buckets[j][bucket_index] = temp;
                buckets[j][0] += 1;
                break;
            }
        }
    }

    free(bucket_limits);
    free(problem_array);
}

/*
    Basic load balance for large (64 bit) problems:
    -    Break down the problem_array into N buckets, and return an array containing pointers to each bucket.
    -    Performs load balancing in two passes if there are empty buckets, otherwise it is assumed
    -    that the input is relatively uniformly distributed.
    -    If no balancing is performed, the distribution accross buckets is 0 -> UINT64_MAX.
*/
void basic_load_balance_large(unsigned long long int *problem_array, unsigned long long int **buckets)
{
    // Create enough space in each bucket for two elements. First element is used to represent sizeof() the sub array
    for (unsigned long long int i = 0; i < thread_count; i++)
    {
        buckets[i] = emalloc_large(2 * sizeof(unsigned long long int));
        buckets[i][0] = 1;
    }

    unsigned long long int highest_problemset_number = 0; // Highest number in the problem set
    omp_set_num_threads(thread_count);
    unsigned long long int *highest_problemset_number_perthread = emalloc_large(thread_count);
    unsigned long long int problem_size_per_thread = problemsize / thread_count;

    // each thread finds the highest number in its zone of the unsorted numbers
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        unsigned long long int start_index = (thread_id * problem_size_per_thread);
        unsigned long long int end_index = (thread_id == (thread_count - 1)) ? problemsize : ((thread_id + 1) * problem_size_per_thread);
        unsigned long long int highest_num_thread = 0;
        for (unsigned long long int i = start_index; i < end_index; i++)
        {
            if (problem_array[i] > highest_num_thread)
            {
                highest_num_thread = problem_array[i];
            }
        }
        highest_problemset_number_perthread[thread_id] = highest_num_thread;
#pragma omp barrier
    }

    // select highest number found by any thread
    for (unsigned long long int i = 0; i < problemsize; i++)
    {
        if (problem_array[i] > highest_problemset_number)
        {
            highest_problemset_number = problem_array[i];
        }
    }
    free(highest_problemset_number_perthread);

    // create buckets == threads and set the number range evenly for all buckets (first pass load balance)
    unsigned long long int bucket_size = highest_problemset_number / thread_count;
    unsigned long long int *bucket_limits = emalloc_large(thread_count * sizeof(unsigned long long int));
    for (unsigned int i = 0; i < thread_count; i++)
    {
        bucket_limits[i] = ((i == thread_count - 1) ? highest_problemset_number : (i + 1) * bucket_size);
    }

    // Loop through the entire problem array and add each number to a respective bucket
    unsigned long long int temp, bucket_index;
    for (unsigned long long int i = 0; i < problemsize; i++)
    {
        temp = problem_array[i];
        for (unsigned int j = 0; j < thread_count; j++)
        {
            if (temp <= bucket_limits[j])
            {
                // When a suitable bucket is found, extend its length and incrememnt its counter
                bucket_index = buckets[j][0];
                buckets[j] = erealloc_large(buckets[j], (bucket_index + 1) * sizeof(unsigned long long int));
                buckets[j][bucket_index] = temp;
                buckets[j][0] += 1;
                break;
            }
        }
    }

    // Enumerate through the buckets list and if any are empty put them next to eachother (except for the first bucket)
    int empty_buckets = 0;
    for (int i = (thread_count - 1); i > 1; i--)
    {
        if (buckets[i][0] == 1) // Empty bucket found
        {
            for (int j = (i - 1); j > 0; j--) // Walk backwards to find the next full bucket, and switch places
            {
                if (buckets[j][0] != 1) // Non-empty bucket found
                {
                    unsigned long long int *temp_bucket = buckets[j]; // Switch them
                    buckets[j] = buckets[i];
                    buckets[i] = temp_bucket;
                    break; // Do not continue the inner loop
                }
            }
        }
    }

    // Count the number of empty buckets
    for (unsigned int j = 0; j < thread_count; j++)
    {
        if (buckets[j][0] == 1)
        {
            empty_buckets++;
        }
    }

    // If there is an empty bucket perform a second pass load distribution (problem is assumed to be exponential if this occurs)
    // WARNING: This produces significant performance degradation for 50000000+ sized problems
    if ((empty_buckets > 0) && (problemsize < 6000000))
    {
        // Loop through the first bucket to find its highest number
        unsigned long long int highest_number = 0;
        unsigned long long int bucket_max = buckets[0][0];
        for (unsigned long long int i = 0; i < bucket_max; i++)
        {
            if (buckets[0][i] > highest_number)
            {
                highest_number = buckets[0][i];
            }
        }

        // Calculate an exponential distribution for the buckets instead of uniform
        double factor = pow((double)highest_number, 1.0 / (empty_buckets + 1));
        unsigned long long int bucket_limits_2[empty_buckets + 1];
        for (unsigned int i = 0; i <= empty_buckets; i++)
        {
            bucket_limits_2[i] = (unsigned long long int)(pow(factor, i + 1) - 1); // Exponential distribution
            if (i == empty_buckets)
            {
                bucket_limits_2[i] = highest_number; // Ensure the last bucket limit is set to the highest number
            }
        }

        // Copy data from bucket 0 into a new holding bucket and empty b0 to reinsert elements accross all empty buckets
        unsigned long long int *holding_bucket = emalloc_large(bucket_max * sizeof(unsigned long long int));
        memcpy(holding_bucket, buckets[0], bucket_max * sizeof(unsigned long long int));

        // Make sure all empty buckets are initialised as empty
        for (int i = 0; i <= empty_buckets; i++)
        {
            buckets[i] = erealloc_large(buckets[i], (2 * sizeof(unsigned long long int)));
            buckets[i][0] = 1;
        }

        // Perform the second pass and insert elements into better balanced buckets
        for (unsigned long long int i = 1; i < bucket_max; i++)
        {
            temp = holding_bucket[i];
            for (unsigned int j = 0; j <= empty_buckets; j++)
            { 
                if (temp <= bucket_limits_2[j])
                {
                    // When a suitable bucket is found, extend its length and incrememnt its counter
                    bucket_index = buckets[j][0];
                    buckets[j] = erealloc_large(buckets[j], (bucket_index + 1) * sizeof(unsigned long long int)); 
                    buckets[j][bucket_index] = temp;
                    buckets[j][0] += 1;
                    break;
                }
            }
        }
        free(holding_bucket);
        free(bucket_limits);
    }
    free(problem_array);
}

// Uniform distribution problem set
void uniform_problem()
{
    printf("\nUniform distribution problem\n");
    unsigned int cachesize = cpu_cachesize();
    unsigned int uints_max = cachesize * 209715; // 209,715 = 80% of (1MiB / sizeof(uint_MAX))
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
    unsigned int uints_max = cachesize * 209715; // 209,715 = 80% of (1MiB / sizeof(uint_MAX))
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
    int ulongints_max = cachesize * 104857; // 104,857 = 80% of (1MiB / sizeof(ulong_MAX)
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
    for (unsigned long long int i = 0; i < (problemsize - 1); i++)
    {
        if (sorted_array[i] > sorted_array[i + 1])
        {
            printf("Exponential array is not in ascending order. Sorting failed at index: %llu\n", i);
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
