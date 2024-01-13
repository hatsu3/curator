// -*- c++ -*-

#include <faiss/MultiTenantIndexIVFFlatBF.h>

#include <omp.h>

#include <cinttypes>
#include <cstdio>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

MultiTenantIndexIVFFlatBF::MultiTenantIndexIVFFlatBF(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric, 
        size_t bf_capacity,
        float bf_false_pos, 
        float gamma)
        : MultiTenantIndexIVFFlat(quantizer, d, nlist, metric) {
    code_size = sizeof(float) * d;
    by_residual = false;

    this->bf_capacity = bf_capacity;
    this->bf_false_pos = bf_false_pos;
    bf_params.projected_element_count = bf_capacity;
    bf_params.false_positive_probability = bf_false_pos;
    bf_params.random_seed = 0xA5A5A5A5;
    FAISS_THROW_IF_NOT_MSG(bf_params, "Invalid bloom filter parameters");
    bf_params.compute_optimal_parameters();

    this->gamma = gamma;
}

MultiTenantIndexIVFFlatBF::MultiTenantIndexIVFFlatBF() {
    by_residual = false;
}

void MultiTenantIndexIVFFlatBF::train(idx_t n, const float* x, tid_t tid) {
    MultiTenantIndexIVFFlat::train(n, x, tid);
    
    // create a bloom filter for each inverted list
    ivf_bfs.resize(nlist);
    for (size_t i = 0; i < nlist; i++) {
        ivf_bfs[i] = bloom_filter(bf_params);
    }

    // create an entry in access_map.tid_to_vids for each inverted list
    access_map.tid_to_vids.resize(nlist);
}

void MultiTenantIndexIVFFlatBF::add_vector_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids,
        tid_t tid) {
    // insert vectors into the corresponding inverted lists
    idx_t* coarse_idx = new idx_t[n];
    quantizer->assign(n, x, coarse_idx);
    
    for (idx_t i = 0; i < n; i++) {
        size_t xid = xids ? xids[i] : ntotal + i;
        idx_t bucket_idx = coarse_idx[i];
        access_map.vid_to_bid[xid] = bucket_idx;
        access_map.tid_to_vids[bucket_idx][tid].insert(xid);
        vector_owners[xid] = tid;
    }
    
    add_core(n, x, xids, coarse_idx);

    // update the bloom filters
    for (idx_t i = 0; i < n; i++) {
        ivf_bfs[coarse_idx[i]].insert(tid);
    }
}

void MultiTenantIndexIVFFlatBF::grant_access(idx_t xid, tid_t tid) {
    // update the vector-to-tenant mapping
    idx_t bucket_idx = access_map.vid_to_bid[xid];
    auto& ttv = access_map.tid_to_vids[bucket_idx];
    if (ttv.find(tid) == ttv.end()) {
        ttv.emplace(tid, std::unordered_set<idx_t>());
    }
    ttv[tid].insert(xid);
}

bool MultiTenantIndexIVFFlatBF::remove_vector(idx_t xid, tid_t tid) {
    // check if the vector is accessible by the tenant
    if (!vector_owners.at(xid) == tid) {
        return false;
    }
    vector_owners.erase(xid);

    // remove the vector from the direct map
    if (!direct_map.no()) {
        assert(direct_map.type == DirectMap::Hashtable);
        direct_map.hashtable.erase(xid);
    }

    // remove the vector from the access map
    idx_t bucket_idx = access_map.vid_to_bid.at(xid);
    access_map.vid_to_bid.erase(xid);

    bool update_bf = false;
    auto& tid_to_vids = access_map.tid_to_vids[bucket_idx];
    for (auto it = tid_to_vids.begin(); it != tid_to_vids.end(); ) {
        if (it->second.erase(xid) && it->second.empty()) {
            it = tid_to_vids.erase(it);
            update_bf = true;
        } else {
            ++it;
        }
    }

    // recompute the bloom filter if necessary
    if (update_bf) {
        bloom_filter new_bf(bf_params);
        for (const auto& tid : tid_to_vids) {
            new_bf.insert(tid.first);
        }
        ivf_bfs[bucket_idx] = new_bf;
    }

    ntotal -= 1;
    return true;
}

bool MultiTenantIndexIVFFlatBF::revoke_access(idx_t xid, tid_t tid) {
    idx_t bucket_idx = access_map.vid_to_bid.at(xid);
    auto& tid_to_vids = access_map.tid_to_vids[bucket_idx];
    if (tid_to_vids.find(tid) == tid_to_vids.end()) {
        return false;
    }
    
    bool update_bf = false;
    bool success = tid_to_vids[tid].erase(xid);
    if (success && tid_to_vids[tid].empty()) {
        tid_to_vids.erase(tid);
        update_bf = true;
    }

    // recompute the bloom filter if necessary
    if (update_bf) {
        bloom_filter new_bf(bf_params);
        for (const auto& tid : tid_to_vids) {
            new_bf.insert(tid.first);
        }
        ivf_bfs[bucket_idx] = new_bf;
    }
    
    return success;
}

void MultiTenantIndexIVFFlatBF::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    
    FAISS_THROW_IF_NOT(k > 0);
    
    // TODO: we do not support overriding the gamma parameter for now
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }

    // search function for a subset of queries
    auto sub_search_func = [this, k, tid, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nlist]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nlist]);

        double t0 = getmillisecs();
        quantizer->search(
                n,
                x,
                nlist,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        // filter out inverted lists that do not contain the tenant
        // by setting their coarse distances to -1
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < nlist; j++) {
                idx_t list_no = idx[i * nlist + j];
                if (!ivf_bfs[list_no].contains(tid)) {
                    coarse_dis[i * nlist + j] = -1;
                }
            }
        }

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nlist);

        search_preassigned(
                n,
                x,
                k,
                tid,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats);
        
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle paralellization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void MultiTenantIndexIVFFlatBF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    // we potentially scan all inverted lists
    idx_t nprobe = nlist;

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    MultiTenantIDSelector mt_sel(tid, &access_map, sel);

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    FAISS_THROW_IF_NOT_MSG(
            pmode == 0 || pmode == 3,
            "search_preassigned only supports parallel_mode = 0 or 3");

    bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        InvertedListScanner* scanner =
                get_InvertedListScanner(store_pairs, &mt_sel);
        ScopeDeleter1<InvertedListScanner> del(scanner);

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 idx_t list_size_max) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key)) {
                return (size_t)0;
            }

            // the inverted list does not contain any vectors accessible by the tenant
            if (coarse_dis_i < 0) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key));

                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);

                    return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids.reset(new InvertedLists::ScopedIds(invlists, key));
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->scan_codes(
                            list_size, codes, ids, simi, idxi, k);

                    size_t list_size_tenant = 0;
                    if (!scanner->sel) {
                        list_size_tenant = list_size;
                    } else {
                        list_size_tenant = access_map.tid_to_vids[key].at(tid).size();
                    }

                    return list_size_tenant;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    size_t x = scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            max_codes - nscan);
                    nscan += x;

                    if (nscan >= int(this->gamma * k)) {
                        break;
                    }

                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data(),
                            unlimited_list_size);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;
                size_t j = ij % nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size);
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats) {
        ivf_stats->nq += n;
        ivf_stats->nlist += nlistv;
        ivf_stats->ndis += ndis;
        ivf_stats->nheap_updates += nheap;
    }
}

} // namespace faiss
