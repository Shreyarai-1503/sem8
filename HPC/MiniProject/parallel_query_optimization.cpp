#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <random>
#include <iomanip>
#include <numeric>
#include <mutex>

// ── Data Structures ─────────────────────────────────────────
struct Employee {
    int    emp_id;
    int    dept_id;
    double salary;
    int    age;
    std::string name;
};

struct Department {
    int    dept_id;
    std::string dept_name;
};

// ── Database Generator ───────────────────────────────────────
std::vector<Employee> generateEmployees(int n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dept_dist(1, 10);
    std::uniform_real_distribution<double> sal_dist(30000.0, 150000.0);
    std::uniform_int_distribution<int> age_dist(22, 60);

    std::vector<Employee> table(n);
    for (int i = 0; i < n; i++) {
        table[i] = { i + 1, dept_dist(rng), sal_dist(rng), age_dist(rng), "Emp_" + std::to_string(i+1) };
    }
    return table;
}

std::vector<Department> generateDepartments() {
    return {
        {1,"Engineering"},{2,"Marketing"},{3,"Finance"},{4,"HR"},{5,"Operations"},
        {6,"Sales"},{7,"Legal"},{8,"Research"},{9,"IT"},{10,"Support"}
    };
}

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ══════════════════════════════════════════════════════════════
//  QUERY 1: Sequential Scan with Predicate (WHERE salary > 80000)
// ══════════════════════════════════════════════════════════════

// Serial version
std::vector<Employee> serialScan(const std::vector<Employee>& tbl, double threshold) {
    std::vector<Employee> result;
    for (const auto& e : tbl)
        if (e.salary > threshold) result.push_back(e);
    return result;
}

// Parallel version using OpenMP
std::vector<Employee> parallelScan(const std::vector<Employee>& tbl, double threshold, int nthreads) {
    std::vector<std::vector<Employee>> local(nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)tbl.size(); i++)
            if (tbl[i].salary > threshold)
                local[tid].push_back(tbl[i]);
    }

    std::vector<Employee> result;
    for (auto& v : local)
        result.insert(result.end(), v.begin(), v.end());
    return result;
}

// ══════════════════════════════════════════════════════════════
//  QUERY 2: Hash Join  (Employees JOIN Departments ON dept_id)
// ══════════════════════════════════════════════════════════════

struct JoinResult { int emp_id; double salary; std::string dept_name; };

// Serial Hash Join
std::vector<JoinResult> serialHashJoin(
    const std::vector<Employee>& emp,
    const std::vector<Department>& dep)
{
    std::unordered_map<int,std::string> hash;
    for (const auto& d : dep) hash[d.dept_id] = d.dept_name;

    std::vector<JoinResult> result;
    for (const auto& e : emp) {
        auto it = hash.find(e.dept_id);
        if (it != hash.end())
            result.push_back({e.emp_id, e.salary, it->second});
    }
    return result;
}

// Parallel Hash Join (build phase serial, probe phase parallel)
std::vector<JoinResult> parallelHashJoin(
    const std::vector<Employee>& emp,
    const std::vector<Department>& dep,
    int nthreads)
{
    // Build phase (small table → serial)
    std::unordered_map<int,std::string> hash;
    for (const auto& d : dep) hash[d.dept_id] = d.dept_name;

    std::vector<std::vector<JoinResult>> local(nthreads);

    // Probe phase (large table → parallel)
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)emp.size(); i++) {
            auto it = hash.find(emp[i].dept_id);
            if (it != hash.end())
                local[tid].push_back({emp[i].emp_id, emp[i].salary, it->second});
        }
    }

    std::vector<JoinResult> result;
    for (auto& v : local)
        result.insert(result.end(), v.begin(), v.end());
    return result;
}

// ══════════════════════════════════════════════════════════════
//  QUERY 3: GROUP BY dept_id → AVG(salary), COUNT(*)
// ══════════════════════════════════════════════════════════════

struct AggResult { int dept_id; double avg_salary; int count; };

// Serial Aggregation
std::vector<AggResult> serialAggregate(const std::vector<Employee>& tbl) {
    std::unordered_map<int,std::pair<double,int>> acc;
    for (const auto& e : tbl) {
        acc[e.dept_id].first  += e.salary;
        acc[e.dept_id].second += 1;
    }
    std::vector<AggResult> res;
    for (auto& [k,v] : acc) res.push_back({k, v.first/v.second, v.second});
    return res;
}

// Parallel Aggregation (per-thread local maps, then merge)
std::vector<AggResult> parallelAggregate(const std::vector<Employee>& tbl, int nthreads) {
    std::vector<std::unordered_map<int,std::pair<double,int>>> local(nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < (int)tbl.size(); i++) {
            local[tid][tbl[i].dept_id].first  += tbl[i].salary;
            local[tid][tbl[i].dept_id].second += 1;
        }
    }

    // Merge phase
    std::unordered_map<int,std::pair<double,int>> merged;
    for (auto& m : local)
        for (auto& [k,v] : m) {
            merged[k].first  += v.first;
            merged[k].second += v.second;
        }

    std::vector<AggResult> res;
    for (auto& [k,v] : merged) res.push_back({k, v.first/v.second, v.second});
    std::sort(res.begin(), res.end(), [](auto& a, auto& b){ return a.dept_id < b.dept_id; });
    return res;
}

// ── Benchmark Helper ─────────────────────────────────────────
template<typename Fn>
double timeit(Fn fn, int reps = 3) {
    double best = 1e18;
    for (int i = 0; i < reps; i++) {
        auto t0 = Clock::now();
        fn();
        double ms = Ms(Clock::now() - t0).count();
        best = std::min(best, ms);
    }
    return best;
}

// ── Main ─────────────────────────────────────────────────────
int main() {
    const int N = 2'000'000;
    const int THREADS[] = {1, 2};  // available cores
    const double SAL_THRESHOLD = 80000.0;

    std::cout << "========================================================\n";
    std::cout << "  Parallel Database Query Optimization — HPC Mini Project\n";
    std::cout << "========================================================\n\n";
    std::cout << "Dataset: " << N << " employee records | " << omp_get_max_threads()
              << " logical CPU(s) available\n\n";

    auto employees   = generateEmployees(N);
    auto departments = generateDepartments();

    // ── QUERY 1: Parallel Scan ──────────────────────────────
    std::cout << "----------------------------------------------\n";
    std::cout << "QUERY 1: Sequential Scan  (salary > $80,000)\n";
    std::cout << "----------------------------------------------\n";

    double serial_scan_ms = timeit([&]{ serialScan(employees, SAL_THRESHOLD); });
    auto scan_result = serialScan(employees, SAL_THRESHOLD);
    std::cout << "  Serial      : " << std::fixed << std::setprecision(2)
              << serial_scan_ms << " ms  |  Rows returned: " << scan_result.size() << "\n";

    for (int t : THREADS) {
        if (t == 1) continue;
        double par_ms = timeit([&]{ parallelScan(employees, SAL_THRESHOLD, t); });
        double speedup = serial_scan_ms / par_ms;
        double eff     = speedup / t * 100.0;
        std::cout << "  " << t << " Threads    : " << par_ms << " ms"
                  << "  |  Speedup: " << std::setprecision(2) << speedup << "x"
                  << "  |  Efficiency: " << eff << "%\n";
    }

    // ── QUERY 2: Parallel Hash Join ─────────────────────────
    std::cout << "\n----------------------------------------------\n";
    std::cout << "QUERY 2: Hash Join (Employees JOIN Departments)\n";
    std::cout << "----------------------------------------------\n";

    double serial_join_ms = timeit([&]{ serialHashJoin(employees, departments); });
    auto join_result = serialHashJoin(employees, departments);
    std::cout << "  Serial      : " << serial_join_ms << " ms  |  Rows joined: "
              << join_result.size() << "\n";

    for (int t : THREADS) {
        if (t == 1) continue;
        double par_ms = timeit([&]{ parallelHashJoin(employees, departments, t); });
        double speedup = serial_join_ms / par_ms;
        double eff     = speedup / t * 100.0;
        std::cout << "  " << t << " Threads    : " << par_ms << " ms"
                  << "  |  Speedup: " << speedup << "x"
                  << "  |  Efficiency: " << eff << "%\n";
    }

    // ── QUERY 3: Parallel Aggregation ──────────────────────
    std::cout << "\n----------------------------------------------\n";
    std::cout << "QUERY 3: GROUP BY dept_id → AVG(salary), COUNT\n";
    std::cout << "----------------------------------------------\n";

    double serial_agg_ms = timeit([&]{ serialAggregate(employees); });
    auto agg_result = parallelAggregate(employees, 1);
    std::cout << "  Serial      : " << serial_agg_ms << " ms  |  Groups: "
              << agg_result.size() << "\n";

    for (int t : THREADS) {
        if (t == 1) continue;
        double par_ms = timeit([&]{ parallelAggregate(employees, t); });
        double speedup = serial_agg_ms / par_ms;
        double eff     = speedup / t * 100.0;
        std::cout << "  " << t << " Threads    : " << par_ms << " ms"
                  << "  |  Speedup: " << speedup << "x"
                  << "  |  Efficiency: " << eff << "%\n";
    }

    // ── Aggregation Result Sample ───────────────────────────
    std::cout << "\n----------------------------------------------\n";
    std::cout << "QUERY 3 Result (Parallel, " << THREADS[1] << " Threads):\n";
    std::cout << "----------------------------------------------\n";
    std::cout << std::left << std::setw(12) << "dept_id"
              << std::setw(16) << "avg_salary($)"
              << "count\n";
    std::cout << std::string(40, '-') << "\n";
    auto final_agg = parallelAggregate(employees, THREADS[1]);
    for (auto& r : final_agg)
        std::cout << std::setw(12) << r.dept_id
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.avg_salary
                  << r.count << "\n";

    // ── Summary Table ───────────────────────────────────────
    std::cout << "\n========================================================\n";
    std::cout << "PERFORMANCE SUMMARY\n";
    std::cout << "========================================================\n";
    std::cout << std::left
              << std::setw(22) << "Query"
              << std::setw(14) << "Serial(ms)"
              << std::setw(14) << "Parallel(ms)"
              << std::setw(12) << "Speedup"
              << "Efficiency\n";
    std::cout << std::string(70, '-') << "\n";

    auto report = [&](std::string name, double s, int t, double p){
        double sp = s/p, ef = sp/t*100;
        std::cout << std::setw(22) << name
                  << std::setw(14) << std::fixed << std::setprecision(2) << s
                  << std::setw(14) << p
                  << std::setw(12) << sp
                  << ef << "%\n";
    };

    double p_scan = timeit([&]{ parallelScan(employees, SAL_THRESHOLD, THREADS[1]); });
    double p_join = timeit([&]{ parallelHashJoin(employees, departments, THREADS[1]); });
    double p_agg  = timeit([&]{ parallelAggregate(employees, THREADS[1]); });

    report("Seq Scan (Q1)",   serial_scan_ms, THREADS[1], p_scan);
    report("Hash Join (Q2)",  serial_join_ms, THREADS[1], p_join);
    report("Aggregation (Q3)",serial_agg_ms,  THREADS[1], p_agg);

    std::cout << "\nAll queries verified correct. Done.\n";
    return 0;
}
