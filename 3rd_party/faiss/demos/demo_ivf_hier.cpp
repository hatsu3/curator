#include <faiss/MultiTenantIndexIVFHierarchical.h>
#include <faiss/MetricType.h>


int main() {
    faiss::IndexFlatL2 quantizer(2);
    faiss::MultiTenantIndexIVFHierarchical index(&quantizer, /*d=*/2, /*nlist=*/3, faiss::METRIC_L2);
    float X[2][2] = {{1, 2}, {3, 4}};
    faiss::idx_t xids[2] = {0, 1};
    faiss::tid_t tids[2] = {1, 2};
    index.train(/*n=*/2, /*x=*/X[0], /*tid=*/0);
    index.add_with_ids(2, X[0], xids, tids, /*n_tids=*/2);
    // index.search(1, query, k, distances, labels);
}
