// -*- c++ -*-

#ifndef FAISS_MULTI_TENANT_INDEX_H
#define FAISS_MULTI_TENANT_INDEX_H

#include <faiss/MetricType.h>
#include <faiss/Index.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <typeinfo>

namespace faiss {

/** Abstract structure for a multi-tenant index, supports adding vectors and searching them.
 *
 * All vectors provided at add or search time are 32-bit float arrays,
 * although the internal representation may vary.
 */
struct MultiTenantIndex {
    using component_t = float;
    using distance_t = float;

    int d;        ///< vector dimension
    idx_t ntotal; ///< total nb of indexed vectors
    bool verbose; ///< verbosity level

    /// set if the Index does not require training, or if training is
    /// done already
    bool is_trained;

    /// type of metric this index uses for search
    MetricType metric_type;
    float metric_arg; ///< argument of the metric type

    explicit MultiTenantIndex(idx_t d = 0, MetricType metric = METRIC_L2)
            : d(d),
              ntotal(0),
              verbose(false),
              is_trained(true),
              metric_type(metric),
              metric_arg(0) {}

    virtual ~MultiTenantIndex();

    /** Perform training on a representative set of vectors
     *
     * @param n      nb of training vectors
     * @param x      training vecors, size n * d
     * @param tid    id of the tenant that is training
     */
    virtual void train(idx_t n, const float* x, tid_t tid);

    /** Add n vectors of dimension d to the index.
     *
     * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
     * This function slices the input vectors in chunks smaller than
     * blocksize_add and calls add_core.
     * @param x      input matrix, size n * d
     * @param tid    id of the creator of the vectors
     */
    virtual void add_vector(idx_t n, const float* x, tid_t tid) = 0;

    /** Same as add, but stores xids instead of sequential ids.
     *
     * The default implementation fails with an assertion, as it is
     * not supported by all indexes.
     *
     * @param xids if non-null, ids to store for the vectors (size n)
     */
    virtual void add_vector_with_ids(idx_t n, const float* x, const idx_t* xids, tid_t tid) = 0;

    /** Grant access to a tenant to a vector.
     * 
     * @param xid    id of the vector
     * @param tid    id of the tenant to grant access to
     */
    virtual void grant_access(idx_t xid, tid_t tid) = 0;

    /** query n vectors of dimension d to the index.
     *
     * return at most k vectors. If there are not enough results for a
     * query, the result array is padded with -1s.
     *
     * @param x           input vectors to search, size n * d
     * @param tid         id of the tenant that is searching
     * @param labels      output labels of the NNs, size n*k
     * @param distances   output pairwise distances, size n*k
     */
    virtual void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const = 0;

    /** query n vectors of dimension d to the index.
     *
     * return all vectors with distance < radius. Note that many
     * indexes do not implement the range_search (only the k-NN search
     * is mandatory).
     *
     * @param x           input vectors to search, size n * d
     * @param tid         id of the tenant that is searching
     * @param radius      search radius
     * @param result      result table
     */
    virtual void range_search(
            idx_t n,
            const float* x,
            float radius,
            tid_t tid,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const;

    /** return the indexes of the k vectors closest to the query x.
     *
     * This function is identical as search but only return labels of neighbors.
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     * @param tid         id of the tenant that is searching
     */
    virtual void assign(idx_t n, const float* x, tid_t tid, idx_t* labels, idx_t k = 1)
            const;

    /// removes all elements from the database.
    virtual void reset() = 0;

    /** Removes a vector from the index.
     *
     * The default implementation fails with an assertion, as it is
     * not supported by all indexes.
     *
     * @param xid    id of the vector to remove
     * @param tid    id of the tenant that is removing the vector
     */
    virtual bool remove_vector(idx_t xid, tid_t tid) = 0;

    /** Revokes access to a vector from a tenant.
     *
     * The default implementation fails with an assertion, as it is
     * not supported by all indexes.
     *
     * @param xid    id of the vector to revoke access to
     * @param tid    id of the tenant to revoke access from
     */
    virtual bool revoke_access(idx_t xid, tid_t tid) = 0;

    /** Reconstruct a stored vector (or an approximation if lossy coding)
     *
     * this function may not be defined for some indexes
     * @param key         id of the vector to reconstruct
     * @param recons      reconstucted vector (size d)
     */
    virtual void reconstruct(idx_t key, float* recons) const;

    /** Reconstruct several stored vectors (or an approximation if lossy coding)
     *
     * this function may not be defined for some indexes
     * @param n        number of vectors to reconstruct
     * @param keys        ids of the vectors to reconstruct (size n)
     * @param recons      reconstucted vector (size n * d)
     */
    virtual void reconstruct_batch(idx_t n, const idx_t* keys, float* recons)
            const;

    /** Reconstruct vectors i0 to i0 + ni - 1
     *
     * this function may not be defined for some indexes
     * @param recons      reconstucted vector (size ni * d)
     */
    virtual void reconstruct_n(idx_t i0, idx_t ni, float* recons) const;

    /** Similar to search, but also reconstructs the stored vectors (or an
     * approximation in the case of lossy coding) for the search results.
     *
     * If there are not enough results for a query, the resulting arrays
     * is padded with -1s.
     *
     * @param recons      reconstructed vectors size (n, k, d)
     **/
    virtual void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const;

    /** Computes a residual vector after indexing encoding.
     *
     * The residual vector is the difference between a vector and the
     * reconstruction that can be decoded from its representation in
     * the index. The residual can be used for multiple-stage indexing
     * methods, like IndexIVF's methods.
     *
     * @param x           input vector, size d
     * @param residual    output residual vector, size d
     * @param key         encoded index, as returned by search and assign
     */
    virtual void compute_residual(const float* x, float* residual, idx_t key)
            const;

    /** Computes a residual vector after indexing encoding (batch form).
     * Equivalent to calling compute_residual for each vector.
     *
     * The residual vector is the difference between a vector and the
     * reconstruction that can be decoded from its representation in
     * the index. The residual can be used for multiple-stage indexing
     * methods, like IndexIVF's methods.
     *
     * @param n           number of vectors
     * @param xs          input vectors, size (n x d)
     * @param residuals   output residual vectors, size (n x d)
     * @param keys        encoded index, as returned by search and assign
     */
    virtual void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const;

    /** Get a DistanceComputer (defined in AuxIndexStructures) object
     * for this kind of index.
     *
     * DistanceComputer is implemented for indexes that support random
     * access of their vectors.
     */
    virtual DistanceComputer* get_distance_computer() const;

    /* The standalone codec interface */

    /** size of the produced codes in bytes */
    virtual size_t sa_code_size() const;

    /** encode a set of vectors
     *
     * @param n       number of vectors
     * @param x       input vectors, size n * d
     * @param bytes   output encoded vectors, size n * sa_code_size()
     */
    virtual void sa_encode(idx_t n, const float* x, uint8_t* bytes) const;

    /** decode a set of vectors
     *
     * @param n       number of vectors
     * @param bytes   input encoded vectors, size n * sa_code_size()
     * @param x       output vectors, size n * d
     */
    virtual void sa_decode(idx_t n, const uint8_t* bytes, float* x) const;

    /** moves the entries from another dataset to self.
     * On output, other is empty.
     * add_id is added to all moved ids
     * (for sequential ids, this would be this->ntotal) */
    virtual void merge_from(MultiTenantIndex& otherIndex, idx_t add_id = 0);

    /** check that the two indexes are compatible (ie, they are
     * trained in the same way and have the same
     * parameters). Otherwise throw. */
    virtual void check_compatible_for_merge(const MultiTenantIndex& otherIndex) const;
};

} // namespace faiss

#endif
