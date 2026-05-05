// Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS . 

#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>
using namespace std;

class Graph {
public:
    int vertices = 6;
    // Adjacency list: 0-1-2-3-4-5 connected as a small undirected graph
    vector<vector<int>> graph = {{1}, {0,2,3}, {1,4,5}, {1,4}, {2,3}, {2,3,4}};
    vector<bool> visited;

    void printGraph() {
        for (int i = 0; i < vertices; i++) {
            cout << i << " -> ";
            for (int j : graph[i]) 
                cout << j << " ";
            cout << endl;
        }
    }

    void initialize_visited() {
        visited.assign(vertices, false);
    }

    // ── Sequential DFS ──────────────────────────────────────
    void dfs(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current = s.top();
            s.pop();
            cout << current << " ";

            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    s.push(neighbor);
                }
            }
        }
    }

    // ── Parallel DFS (OpenMP) ────────────────────────────────
    // Uses #pragma omp parallel for to explore neighbors
    // and #pragma omp critical to protect shared stack & visited[]
    void parallel_dfs(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current;
            #pragma omp critical {
                current = s.top();
                s.pop();
            }
            cout << current << " ";

            #pragma omp parallel for
            for (int i = 0; i < (int)graph[current].size(); i++) {
                int neighbor = graph[current][i];
                #pragma omp critical {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        s.push(neighbor);
                    }
                }
            }
        }
    }

    // ── Sequential BFS ──────────────────────────────────────
    void bfs(int start) {
        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";

            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    // ── Parallel BFS (OpenMP) ────────────────────────────────
    // Uses #pragma omp parallel for to explore neighbors in parallel
    // and #pragma omp critical to protect the shared queue & visited[]
    void parallel_bfs(int start) {
        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int current;
            #pragma omp critical {
                current = q.front();
                q.pop();
            }
            cout << current << " ";

            #pragma omp parallel for
            for (int i = 0; i < (int)graph[current].size(); i++) {
                int neighbor = graph[current][i];
                #pragma omp critical {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
};

int main() {
    Graph g;
    cout << "Adjacency List:\n";
    g.printGraph();

    cout << "\nSequential DFS: ";
    g.initialize_visited();
    auto start = chrono::high_resolution_clock::now();
    g.dfs(0);
    auto end = chrono::high_resolution_clock::now();
    cout << "\nTime: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    cout << "\nParallel DFS: ";
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.parallel_dfs(0);
    end = chrono::high_resolution_clock::now();
    cout << "\nTime: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    cout << "\nSequential BFS: ";
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.bfs(0);
    end = chrono::high_resolution_clock::now();
    cout << "\nTime: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    cout << "\nParallel BFS: ";
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.parallel_bfs(0);
    end = chrono::high_resolution_clock::now();
    cout << "\nTime: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    return 0;
}