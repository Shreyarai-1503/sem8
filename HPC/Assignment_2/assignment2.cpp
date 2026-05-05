// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use existing algorithms and measure the performance of sequential and parallel algorithms. 

#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>
using namespace std;

// ── Sequential Bubble Sort ───────────────────────────────────
void sequential_bubble_sort(vector<int> arr) {
    int n = arr.size();
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Bubble Sort Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

// ── Parallel Bubble Sort (Odd-Even Transposition) ────────────
// Even phase: compare (0,1),(2,3),(4,5)...
// Odd  phase: compare (1,2),(3,4),(5,6)...
// Each phase is independent → safe to parallelize
void parallel_bubble_sort(vector<int> arr) {
    int n = arr.size();
    auto start = chrono::high_resolution_clock::now();
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            // Even phase
            #pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2)
                if (arr[i] > arr[i + 1])
                    swap(arr[i], arr[i + 1]);
        } else {
            // Odd phase
            #pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2)
                if (arr[i] > arr[i + 1])
                    swap(arr[i], arr[i + 1]);
        }
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Parallel Bubble Sort Time:   " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

// ── Merge helper ─────────────────────────────────────────────
void merge(vector<int>& arr, int low, int mid, int high) {
    vector<int> temp;
    int i = low, j = mid + 1;
    while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) temp.push_back(arr[i++]);
        else                  temp.push_back(arr[j++]);
    }
    while (i <= mid)  temp.push_back(arr[i++]);
    while (j <= high) temp.push_back(arr[j++]);
    for (int k = low; k <= high; k++) arr[k] = temp[k - low];
}

// ── Sequential Merge Sort ────────────────────────────────────
void mergesort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergesort(arr, low, mid);
        mergesort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

void sequential_merge_sort(vector<int> arr) {
    auto start = chrono::high_resolution_clock::now();
    mergesort(arr, 0, arr.size() - 1);
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Merge Sort Time:  " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

// ── Parallel Merge Sort ──────────────────────────────────────
// #pragma omp parallel sections splits the two recursive calls
// into separate threads, so left and right halves sort concurrently
void parallel_mergesort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_mergesort(arr, low, mid);
            #pragma omp section
            parallel_mergesort(arr, mid + 1, high);
        }
        merge(arr, low, mid, high);
    }
}

void parallel_merge_sort(vector<int> arr) {
    auto start = chrono::high_resolution_clock::now();
    parallel_mergesort(arr, 0, arr.size() - 1);
    auto end = chrono::high_resolution_clock::now();
    cout << "Parallel Merge Sort Time:    " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

int main() {
    int SIZE;
    cout << "Enter size of array: ";
    cin >> SIZE;

    vector<int> arr(SIZE);
    for (int i = 0; i < SIZE; i++) arr[i] = rand() % 1000;

    sequential_bubble_sort(arr);
    parallel_bubble_sort(arr);
    sequential_merge_sort(arr);
    parallel_merge_sort(arr);
    return 0;
}