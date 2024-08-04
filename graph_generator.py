import matplotlib.pyplot as plt
import numpy as np

# Data points
exponential_times = [12.176116, 11.567336, 11.502285, 15.410210, 11.524221, 11.614221, 11.761126, 11.493968, 11.422009, 12.980059, 20.463849, 11.884481, 11.581762, 12.850046, 11.474564, 13.004926, 13.092325, 11.436620, 11.492508, 12.487290]
random_times = [19.320626, 16.130218, 21.061376, 6.878341, 9.587570, 11.197120, 7.599667, 7.733208, 6.042058, 5.585877, 5.540269, 5.287187, 5.495048, 4.926807, 4.966314, 4.252542, 4.509034, 4.637902, 4.484439, 4.326381]
uniform_times = [15.927635, 26.377389, 18.044846, 14.576115, 10.876097, 11.121721, 8.844263, 8.406701, 7.463123, 5.335927, 6.105023, 5.480987, 5.647648, 5.180497, 4.663692, 4.553177, 4.649474, 4.790359, 4.489410, 4.433332]
threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

baseline_time = np.mean([exponential_times[0], random_times[0], uniform_times[0]])
P = 0.84  # Fraction of the code that can be parallelized
ideal_speedup = [baseline_time / ((1 - P) + (P / min(thread, 16))) for thread in threads]

amdahls_speedup = [baseline_time / ideal_speedup[i] for i in range(len(ideal_speedup))]

# Create the plot
plt.figure(figsize=(10, 6))


plt.plot(threads, exponential_times, marker='o', linestyle='-', color='red', label='Exponential problem set')
plt.plot(threads, random_times, marker='o', linestyle='-', color='green', label='Random problem set')
plt.plot(threads, uniform_times, marker='o', linestyle='-', color='blue', label='Uniform problem set')
plt.plot(threads, amdahls_speedup, marker='x', linestyle='--', color='orange', label='Ideal Speedup (Amdahl\'s law)')

# Invert the y-axis
#plt.gca().invert_yaxis()
plt.xticks(threads)

# Add titles and labels
plt.title('Parallel Bucket Sort: 50 Elements')
plt.xlabel('Thread count')
plt.ylabel('Time Taken (seconds)')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
