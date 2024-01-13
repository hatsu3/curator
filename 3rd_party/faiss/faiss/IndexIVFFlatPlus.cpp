// -*- c++ -*-

#include <faiss/IndexIVFFlatPlus.h>

namespace faiss {

IndexIVFFlatPlus::IndexIVFFlatPlus(
        Index* quantizer,
        size_t d,
        size_t nlist_,
        MetricType metric)
        : IndexIVFFlat(quantizer, d, nlist_, metric) {
    printf("Instantiate IndexIVFFlatPlus\n");
}

void IndexIVFFlatPlus::train(idx_t n, const float* x) {
    printf("Train IndexIVFFlatPlus on %ld vectors\n", n);
    IndexIVFFlat::train(n, x);
}

void IndexIVFFlatPlus::add(idx_t n, const float* x) {
    printf("Add %ld vectors to IndexIVFFlatPlus\n", n);
    IndexIVFFlat::add(n, x);
}

void IndexIVFFlatPlus::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    printf("Search IndexIVFFlatPlus\n");
    IndexIVFFlat::search(n, x, k, distances, labels, params);
}

} // namespace faiss