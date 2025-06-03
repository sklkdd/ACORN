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
#include <chrono>

#include <atomic>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"


// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

// Create index, write it to file and collect statistics
int main(int argc, char *argv[]) {
	// Prepare thread monitoring
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

	// Parameters 
    std::string path_database_vectors;
	std::string path_index;
    int M; 				
    int gamma;
    int M_beta; 		

	// Parse arguments
	if (argc != 6) {
		fprintf(stderr, "Usage: %s <path_database_vectors> <path_index> <M> <gamma> <M_beta>\n", argv[0]);
		exit(1);
	}

	// Store parameters
	path_database_vectors = argv[1];
	path_index = argv[2];
	M = atoi(argv[3]);
	gamma = atoi(argv[4]);
	M_beta = atoi(argv[5]);

	// Read vectors from file
	size_t n_items, d;
	// fvecs_read is from utils.cpp and returns a float*, read_fvecs is from fanns_survey_helpers.cpp and returns a vector<vector<float>>
	float* database_vectors = fvecs_read(path_database_vectors.c_str(), &d, &n_items);

	// According to the following GitHub issue, the metadata does not influence the performance of ACORN:
	// https://github.com/TAG-Research/ACORN/issues/2
	// Therefore, we leave the metadata empty
	std::vector<int> metadata(n_items,0);

    // Initialize the ACORN index
    faiss::IndexACORNFlat acorn_index(d, M, gamma, metadata, M_beta);

	// Add vectors to index: This is the part that is timed for the index construction time
	auto start_time = std::chrono::high_resolution_clock::now();	
	acorn_index.add(n_items, database_vectors);
	auto end_time = std::chrono::high_resolution_clock::now();
	// Stop thread monitoring
    done = true;
    monitor.join();

	// Print statistics
	std::chrono::duration<double> time_diff = end_time - start_time;
	double duration = time_diff.count();
	printf("Maximum number of threads: %d\n", peak_threads.load()-1);	// Subtract 1 because of the monitoring thread
	printf("Index construction time: %.3f s\n", duration);
	peak_memory_footprint();
	delete[] database_vectors;

	// Write index to file
	write_index(&acorn_index, path_index.c_str());
}
