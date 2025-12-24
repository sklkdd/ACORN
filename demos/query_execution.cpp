#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>

#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <faiss/Index.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream>  
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <assert.h>     /* assert */
#include <thread>
#include <set>
#include <math.h>  
#include <numeric> // for std::accumulate
#include <cmath>   // for std::mean and std::stdev
#include <nlohmann/json.hpp>
#include "utils.cpp"

#include <atomic>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

using namespace std;

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

// Execute the queries, compute and report the recall
int main(int argc, char *argv[]) {
	// Restrict number of threads to 1 for query execution
	omp_set_num_threads(1);

	// Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));
    
	// Parameters
    std::string path_database_attributes; 		
	std::string path_query_vectors;
	std::string path_query_attributes;
	std::string path_groundtruth;
	std::string path_index;
	std::string filter_type;
    int k; 		
    int efs; 								//  default is 16

	// Check if the number of arguments is correct
	if (argc < 9) {
		fprintf(stderr, "Usage: %s <path_database_attributes> <path_query_vectors> <path_query_attributes> <path_groundtruth> <path_index> <filter_type> <k> <efs>\n", argv[0]);
		exit(1);	
	}

	// Read command line arguments
	path_database_attributes = argv[1];
	path_query_vectors = argv[2];
	path_query_attributes = argv[3];
	path_groundtruth = argv[4];
	path_index = argv[5];
	filter_type = argv[6];
	k = atoi(argv[7]);
	efs = atoi(argv[8]);

	// Read query vectors
    size_t n_queries, d;
    float* query_vectors;					// n_queries x d
	query_vectors = fvecs_read(path_query_vectors.c_str(), &d, &n_queries);

	// Read ground-truth
	std::vector<std::vector<int>> groundtruth;
	groundtruth = read_ivecs(path_groundtruth);
	assert(n_queries == groundtruth.size() && "Number of queries in query vectors and groundtruth do not match");

	// Truncate ground-truth to at most k items
	for (std::vector<int>& vec : groundtruth) {
		if (vec.size() > k) {
			vec.resize(k);
		}
	}

	// Load index from file
	auto& acorn_index = *dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(path_index.c_str()));

	// Set search parameter and prepare data structure for results
	acorn_index.acorn.efSearch = efs;
    std::vector<faiss::idx_t> nearest_neighbors(k * n_queries);
	std::vector<float> distances(k * n_queries);

	std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
	// Compute bitmap for filtering
	std::vector<char> filter_bitmap;
	// EM
	if (filter_type == "EM"){
		// Read database attributes
		vector<int> database_attributes = read_one_int_per_line(path_database_attributes);
		size_t n_items = database_attributes.size();
		// Read query attributes
		vector<int> query_attributes = read_one_int_per_line(path_query_attributes);
		assert(n_queries == query_attributes.size() && "Number of queries in query vectors and query attributes do not match");
		// Compute filter bitmap (timed)
		filter_bitmap.resize(n_queries * n_items);
		start_time = std::chrono::high_resolution_clock::now();
		for (size_t q = 0; q < n_queries; q++) {
			for (size_t i = 0; i < n_items; i++) {
				filter_bitmap[q * n_items + i] = (database_attributes[i] == query_attributes[q]);
			}
		}
	}
	// R
	else if (filter_type == "R"){
		// Read database attributes
		vector<int> database_attributes = read_one_int_per_line(path_database_attributes);
		size_t n_items = database_attributes.size();
		// Read query attributes
		vector<pair<int,int>> query_attributes = read_two_ints_per_line(path_query_attributes);
		assert(n_queries == query_attributes.size() && "Number of queries in query vectors and query attributes do not match");
		// Compute filter bitmap (timed)
		filter_bitmap.resize(n_queries * n_items);
		start_time = std::chrono::high_resolution_clock::now();
		for (size_t q = 0; q < n_queries; q++) {
			for (size_t i = 0; i < n_items; i++) {
				filter_bitmap[q * n_items + i] = (database_attributes[i] >= query_attributes[q].first && database_attributes[i] <= query_attributes[q].second);
			}
		}
	}
	// EMIS
	else if (filter_type == "EMIS"){
		// Read database attributes
		vector<vector<int>> database_attributes = read_multiple_ints_per_line(path_database_attributes);
		size_t n_items = database_attributes.size();
		// Read query attributes
		vector<int> query_attributes = read_one_int_per_line(path_query_attributes);
		assert(n_queries == query_attributes.size() && "Number of queries in query vectors and query attributes do not match");
		// Compute filter bitmap
		filter_bitmap.resize(n_queries * n_items);
		start_time = std::chrono::high_resolution_clock::now();
		for (size_t q = 0; q < n_queries; q++) {
			for (size_t i = 0; i < n_items; i++) {
				filter_bitmap[q * n_items + i] = (find(database_attributes[i].begin(), database_attributes[i].end(), query_attributes[q]) != database_attributes[i].end());
			}
		}
	}
	// EM_R: Combined Exact Match + Range filter
	// Database format: <em_value>,<r_value> per line
	// Query format: <em_value>,<r_start>-<r_end> per line
	// A database item matches if: db_em == query_em AND r_start <= db_r <= r_end
	else if (filter_type == "EM_R"){
		// Read database attributes (pair of em, r values)
		vector<pair<int, int>> database_attributes = read_em_r_database_attributes(path_database_attributes);
		size_t n_items = database_attributes.size();
		// Read query attributes (em value, r_start, r_end)
		vector<EMRQueryAttribute> query_attributes = read_em_r_query_attributes(path_query_attributes);
		assert(n_queries == query_attributes.size() && "Number of queries in query vectors and query attributes do not match");
		// Compute filter bitmap (timed)
		filter_bitmap.resize(n_queries * n_items);
		start_time = std::chrono::high_resolution_clock::now();
		for (size_t q = 0; q < n_queries; q++) {
			int q_em = query_attributes[q].em_value;
			int q_r_start = query_attributes[q].r_start;
			int q_r_end = query_attributes[q].r_end;
			for (size_t i = 0; i < n_items; i++) {
				int db_em = database_attributes[i].first;
				int db_r = database_attributes[i].second;
				// Both conditions must be true: exact match on EM AND range match on R
				bool em_match = (db_em == q_em);
				bool r_match = (db_r >= q_r_start && db_r <= q_r_end);
				filter_bitmap[q * n_items + i] = (em_match && r_match);
			}
		}
	} else {
		fprintf(stderr, "Unknown filter type: %s\n", filter_type.c_str());
	}
	// Timing after filter bitmap computation but before query execution
	std::chrono::time_point<std::chrono::high_resolution_clock> mid_time = std::chrono::high_resolution_clock::now();

	// Execute queries on ACORN index
	// TODO: How does ACORN behave if there are less than k matching items?
	acorn_index.search(n_queries, query_vectors, k, distances.data(), nearest_neighbors.data(), filter_bitmap.data());
	auto end_time = std::chrono::high_resolution_clock::now();

    // Stop thread count monitoring
    done = true;
    monitor.join();

	// Compute time
	std::chrono::duration<double> time_diff = end_time - start_time;
	std::chrono::duration<double> time_diff_2 = end_time - mid_time;
	double query_execution_time = time_diff.count();
	double query_execution_time_2 = time_diff_2.count();

	// Compute recall 
	// TODO: Check what happens (what ACORN returns) if there are less than k matching items
	// I guess -1 for the remaining items
	size_t match_count = 0;
	size_t total_count = 0;
	faiss::idx_t* nearest_neighbors_ptr = nearest_neighbors.data();
	for (size_t q = 0; q < n_queries; q++) {
		int n_valid_neighbors = std::min(k, (int)groundtruth[q].size());
		std::vector<int> groundtruth_q = groundtruth[q];
		std::vector<int> nearest_neighbors_q(nearest_neighbors_ptr + q * k, nearest_neighbors_ptr + (q+1) * k);
		std::sort(groundtruth_q.begin(), groundtruth_q.end());
		std::sort(nearest_neighbors_q.begin(), nearest_neighbors_q.end());
		std::vector<int> intersection;
		std::set_intersection(groundtruth_q.begin(), groundtruth_q.end(), nearest_neighbors_q.begin(), nearest_neighbors_q.begin() + k, std::back_inserter(intersection));
		match_count += intersection.size();
		total_count += n_valid_neighbors;
	}

	// Report results
	double recall = (double)match_count / total_count;
	double qps = (double)n_queries / query_execution_time;
	double qps_2 = (double)n_queries / query_execution_time_2;
	printf("Maximum number of threads: %d\n", peak_threads.load()-1);	// Subtract 1 because of the monitoring thread
    peak_memory_footprint();
	printf("Queries per second (without filtering): %.3f\n", qps_2);
	printf("Queries per second: %.3f\n", qps);
	printf("Recall: %.3f\n", recall);
}
