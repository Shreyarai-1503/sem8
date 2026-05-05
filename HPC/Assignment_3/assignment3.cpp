// Implement Min, Max, Sum and Average operations using Parallel Reduction.

#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>
using namespace std;

// ── Min ──────────────────────────────────────────────────────
void findMin(vector<int>& arr) {
    // Sequential
    int min_seq = INT_MAX;
    auto start = chrono::high_resolution_clock::now();
    for (int x : arr) if (x < min_seq) min_seq = x;
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Min: " << min_seq << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    // Parallel Reduction
    int min_par = INT_MAX;
    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(min: min_par)
    for (int i = 0; i < (int)arr.size(); i++)
        if (arr[i] < min_par) min_par = arr[i];
    end = chrono::high_resolution_clock::now();
    cout << "Parallel Min:   " << min_par << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

// ── Max ──────────────────────────────────────────────────────
void findMax(vector<int>& arr) {
    int max_seq = INT_MIN;
    auto start = chrono::high_resolution_clock::now();
    for (int x : arr) if (x > max_seq) max_seq = x;
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Max: " << max_seq << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    int max_par = INT_MIN;
    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(max: max_par)
    for (int i = 0; i < (int)arr.size(); i++)
        if (arr[i] > max_par) max_par = arr[i];
    end = chrono::high_resolution_clock::now();
    cout << "Parallel Max:   " << max_par << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

// ── Sum ──────────────────────────────────────────────────────
void findSum(vector<int>& arr) {
    long long sum_seq = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int x : arr) sum_seq += x;
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Sum: " << sum_seq << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    long long sum_par = 0;
    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+: sum_par)
    for (int i = 0; i < (int)arr.size(); i++)
        sum_par += arr[i];
    end = chrono::high_resolution_clock::now();
    cout << "Parallel Sum:   " << sum_par << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

// ── Average ──────────────────────────────────────────────────
void findAverage(vector<int>& arr) {
    double avg_seq = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int x : arr) avg_seq += x;
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Avg: " << avg_seq / arr.size() << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    double avg_par = 0;
    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+: avg_par)
    for (int i = 0; i < (int)arr.size(); i++)
        avg_par += arr[i];
    end = chrono::high_resolution_clock::now();
    cout << "Parallel Avg:   " << avg_par / arr.size() << " | Time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

int main() {
    int N;
    cout << "Enter number of elements: ";
    cin >> N;

    vector<int> arr(N);
    for (int i = 0; i < N; i++) arr[i] = rand() % 1000;

    findMin(arr);
    findMax(arr);
    findSum(arr);
    findAverage(arr);
    return 0;
}