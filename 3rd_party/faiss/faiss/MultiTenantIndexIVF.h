// -*- c++ -*-

#ifndef FAISS_MULTI_TENANT_INDEX_IVF_H
#define FAISS_MULTI_TENANT_INDEX_IVF_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include <faiss/Clustering.h>
#include <faiss/MultiTenantIndex.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

namespace faiss {

struct MultiTenantIndexIVFInterface : Level1Quantizer {
    size_t nprobe = 1;    ///< number of probes at query time
    size_t max_codes = 0; ///< max nb of codes to visit to do a query

    explicit MultiTenantIndexIVFInterface(Index* quantizer = nullptr, size_t nlist = 0)
            : Level1Quantizer(quantizer, nlist) {}

    /** search a set of vectors, that are pre-quantized by the IVF
     *  quantizer. Fill in the corresponding heaps with the query
     *  results. The default implementation uses InvertedListScanners
     *  to do the search.
     *
     * @param n      nb of vectors to query
     * @param x      query vectors, size nx * d
     * @param tid    id of the tenant that is searching
     * @param assign coarse quantization indices, size nx * nprobe
     * @param centroid_dis
     *               distances to coarse centroids, size nx * nprobe
     * @param distance
     *               output distances, size n * k
     * @param labels output labels, size n * k
     * @param store_pairs store inv list index + inv list offset
     *                     instead in upper/lower 32 bit of result,
     *                     instead of ids (used for reranking).
     * @param params used to override the object's search parameters
     * @param stats  search stats to be updated (can be null)
     */
    virtual void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const = 0;

    /** Range search a set of vectors, that are pre-quantized by the IVF
     *  quantizer. Fill in the RangeSearchResults results. The default
     * implementation uses InvertedListScanners to do the search.
     *
     * @param n      nb of vectors to query
     * @param x      query vectors, size nx * d
     * @param tid    id of the tenant that is searching
     * @param assign coarse quantization indices, size nx * nprobe
     * @param centroid_dis
     *               distances to coarse centroids, size nx * nprobe
     * @param result Output results
     * @param store_pairs store inv list index + inv list offset
     *                     instead in upper/lower 32 bit of result,
     *                     instead of ids (used for reranking).
     * @param params used to override the object's search parameters
     * @param stats  search stats to be updated (can be null)
     */
    virtual void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            tid_t tid,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const = 0;

    virtual ~MultiTenantIndexIVFInterface() {}
};

/** Index based on a inverted file (IVF)
 *
 * In the inverted file, the quantizer (an Index instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is stored.
 *
 * The inverted list object is required only after trainng. If none is
 * set externally, an ArrayInvertedLists is used automatically.
 *
 * At search time, the vector to be searched is also quantized, and
 * only the list corresponding to the quantization index is
 * searched. This speeds up the search by making it
 * non-exhaustive. This can be relaxed using multi-probe search: a few
 * (nprobe) quantization indices are selected and several inverted
 * lists are visited.
 *
 * Sub-classes implement a post-filtering of the index that refines
 * the distance estimation from the query to databse vectors.
 */
struct MultiTenantIndexIVF : MultiTenantIndex, MultiTenantIndexIVFInterface {
    /// Access to the actual data
    InvertedLists* invlists = nullptr;
    bool own_invlists = false;

    size_t code_size = 0; ///< code size per vector in bytes

    /** Parallel mode determines how queries are parallelized with OpenMP
     *
     * 0 (default): split over queries
     * 1: parallelize over inverted lists
     * 2: parallelize over both
     * 3: split over queries with a finer granularity
     *
     * PARALLEL_MODE_NO_HEAP_INIT: binary or with the previous to
     * prevent the heap to be initialized and finalized
     */
    int parallel_mode = 0;
    const int PARALLEL_MODE_NO_HEAP_INIT = 1024;

    /** optional map that maps back ids to invlist entries. This
     *  enables reconstruct() */
    DirectMap direct_map;

    /// do the codes in the invlists encode the vectors relative to the
    /// centroids?
    bool by_residual = true;

    /// maintains the access metadata of vectors
    AccessMap access_map;

    /// keeps track of the creator of each vector
    std::unordered_map<idx_t, tid_t> vector_owners;

    /** The Inverted file takes a quantizer (an Index) on input,
     * which implements the function mapping a vector to a list
     * identifier.
     */
    MultiTenantIndexIVF(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2);

    void reset() override;

    /// Trains the quantizer and calls train_encoder to train sub-quantizers
    void train(idx_t n, const float* x, tid_t tid) override;

    /// Calls add_with_ids with NULL ids
    void add_vector(idx_t n, const float* x, tid_t tid) override;

    /// default implementation that calls encode_vectors
    void add_vector_with_ids(
            idx_t n,
            const float* x,
            const idx_t* xids,
            tid_t tid) override;

    void grant_access(idx_t xid, tid_t tid) override;

    /** Implementation of vector addition where the vector assignments are
     * predefined. The default implementation hands over the code extraction to
     * encode_vectors.
     *
     * @param precomputed_idx    quantization indices for the input vectors
     * (size n)
     */
    virtual void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx);

    /** Encodes a set of vectors as they would appear in the inverted lists
     *
     * @param list_nos   inverted list ids as returned by the
     *                   quantizer (size n). -1s are ignored.
     * @param codes      output codes, size n * code_size
     * @param include_listno
     *                   include the list ids in the code (in this case add
     *                   ceil(log8(nlist)) to the code size)
     */
    virtual void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const = 0;

    /** Add vectors that are computed with the standalone codec
     *
     * @param codes  codes to add size n * sa_code_size()
     * @param xids   corresponding ids, size n
     */
    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids);

    /** Train the encoder for the vectors.
     *
     * If by_residual then it is called with residuals and corresponding assign
     * array, otherwise x is the raw training vectors and assign=nullptr */
    virtual void train_encoder(idx_t n, const float* x, const idx_t* assign);

    /// can be redefined by subclasses to indicate how many training vectors
    /// they need
    virtual idx_t train_encoder_num_vectors() const;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            tid_t tid,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    /** assign the vectors, then call search_preassign */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            tid_t tid,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /** Get a scanner for this index (store_pairs means ignore labels)
     *
     * The default search implementation uses this to compute the distances
     */
    virtual InvertedListScanner* get_InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr) const;

    /** reconstruct a vector. Works only if maintain_direct_map is set to 1 or 2
     */
    void reconstruct(idx_t key, float* recons) const override;

    /** Update a subset of vectors.
     *
     * The index must have a direct_map
     *
     * @param nv     nb of vectors to update
     * @param idx    vector indices to update, size nv
     * @param v      vectors of new values, size nv*d
     * @param tid    id of the tenant that is updating
     */
    virtual void update_vectors(int nv, const idx_t* idx, const float* v, tid_t tid);

    /** Reconstruct a subset of the indexed vectors.
     *
     * Overrides default implementation to bypass reconstruct() which requires
     * direct_map to be maintained.
     *
     * @param i0     first vector to reconstruct
     * @param ni     nb of vectors to reconstruct
     * @param recons output array of reconstructed vectors, size ni * d
     */
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /** Similar to search, but also reconstructs the stored vectors (or an
     * approximation in the case of lossy coding) for the search results.
     *
     * Overrides default implementation to avoid having to maintain direct_map
     * and instead fetch the code offsets through the `store_pairs` flag in
     * search_preassigned().
     *
     * @param recons      reconstructed vectors size (n, k, d)
     */
    void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const override;

    /** Reconstruct a vector given the location in terms of (inv list index +
     * inv list offset) instead of the id.
     *
     * Useful for reconstructing when the direct_map is not maintained and
     * the inv list offset is computed by search_preassigned() with
     * `store_pairs` set.
     */
    virtual void reconstruct_from_offset(
            int64_t list_no,
            int64_t offset,
            float* recons) const;

    /// Dataset manipulation functions

    bool remove_vector(idx_t xid, tid_t tid) override;

    bool revoke_access(idx_t xid, tid_t tid) override;

    void check_compatible_for_merge(const MultiTenantIndex& otherIndex) const override;

    virtual void merge_from(MultiTenantIndex& otherIndex, idx_t add_id) override;

    // returns a new instance of a CodePacker
    virtual CodePacker* get_CodePacker() const;

    /** copy a subset of the entries index to the other index
     * see Invlists::copy_subset_to for the meaning of subset_type
     */
    virtual void copy_subset_to(
            IndexIVF& other,
            InvertedLists::subset_type_t subset_type,
            idx_t a1,
            idx_t a2) const;

    ~MultiTenantIndexIVF() override;

    size_t get_list_size(size_t list_no) const {
        return invlists->list_size(list_no);
    }

    /// are the ids sorted?
    bool check_ids_sorted() const;

    /** initialize a direct map
     *
     * @param new_maintain_direct_map    if true, create a direct map,
     *                                   else clear it
     */
    void make_direct_map(bool new_maintain_direct_map = true);

    void set_direct_map_type(DirectMap::Type type);

    /// replace the inverted lists, old one is deallocated if own_invlists
    void replace_invlists(InvertedLists* il, bool own = false);

    /* The standalone codec interface (except sa_decode that is specific) */
    size_t sa_code_size() const override;
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    MultiTenantIndexIVF();
};

} // namespace faiss

#endif
