#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <functional>
#include <omp.h>

const int ARRAY_SIZE = 100000;
const int NUM_THREADS = 4;

void fill_random_vector(std::vector<int>& vec, int size, int min_value, int max_value) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(min_value, max_value);

    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = distribution(gen);
    }
}

void insertion_sort(std::vector<int>& vec, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = vec[i];
        int j = i - 1;
        while (j >= start && vec[j] > key) {
            vec[j + 1] = vec[j];
            j--;
        }
        vec[j + 1] = key;
    }
}

std::vector<int> merge_sorted_vectors(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    std::vector<int> merged_vec;
    merged_vec.reserve(vec1.size() + vec2.size());
    std::merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(merged_vec));
    return merged_vec;
}

std::vector<int> parallel_sort(const std::vector<int>& input_vec) {
    int size = input_vec.size();
    std::vector<std::vector<int>> sorted_segments(NUM_THREADS);

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = size / NUM_THREADS;
        int start = thread_id * chunk_size;
        int end = (thread_id == NUM_THREADS - 1) ? size - 1 : start + chunk_size - 1;

        std::vector<int> sorted_segment(input_vec.begin() + start, input_vec.begin() + end + 1);
        insertion_sort(sorted_segment, 0, end - start);

        sorted_segments[thread_id] = std::move(sorted_segment);

#pragma omp barrier
    }

    std::vector<int> final_sorted_vec = sorted_segments[0];
    for (int i = 1; i < NUM_THREADS; i++) {
        final_sorted_vec = merge_sorted_vectors(final_sorted_vec, sorted_segments[i]);
    }

    return final_sorted_vec;
}

std::vector<int> non_parallel_sort(const std::vector<int>& input_vec) {
    int size = input_vec.size();
    std::vector<std::vector<int>> sorted_segments(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        int thread_id = i;
        int chunk_size = size / NUM_THREADS;
        int start = thread_id * chunk_size;
        int end = (thread_id == NUM_THREADS - 1) ? size - 1 : start + chunk_size - 1;

        std::vector<int> sorted_segment(input_vec.begin() + start, input_vec.begin() + end + 1);
        insertion_sort(sorted_segment, 0, end - start);

        sorted_segments[thread_id] = std::move(sorted_segment);
    }
    std::vector<int> final_sorted_vec = sorted_segments[0];
    for (int i = 1; i < NUM_THREADS; i++) {
        final_sorted_vec = merge_sorted_vectors(final_sorted_vec, sorted_segments[i]);
    }

    return final_sorted_vec;
}

void sort_using(std::function<std::vector<int>(const std::vector<int>&)> sort_func, const std::vector<int>& input_vec)
{
    auto start_time = std::chrono::steady_clock::now();
    std::vector<int> sorted_array = sort_func(input_vec);
    auto end_time = std::chrono::steady_clock::now();

    std::cout << "Sorted array (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << sorted_array[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Execution time: " << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
}

int main() {
    std::vector<int> input_array;
    fill_random_vector(input_array, ARRAY_SIZE, 1, 10000);

    std::cout << "Generated array (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << input_array[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Non-parallel version:" << std::endl;
    sort_using(non_parallel_sort, input_array);

    std::cout << "Parallel version:" << std::endl;
    sort_using(parallel_sort, input_array);
}
