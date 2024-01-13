#ifndef MULTI_TENANT_INDEX_IVF_HIERARCHICAL_H
#define MULTI_TENANT_INDEX_IVF_HIERARCHICAL_H

#include <stdint.h>
#include <unordered_map>

#include <faiss/MetricType.h>
#include <faiss/IndexFlat.h>
#include <faiss/MultiTenantIndexIVFFlat.h>
#include <faiss/BloomFilter.h>

namespace faiss {

typedef size_t vid_t;
typedef idx_t label_t;

struct IdAllocator {
    std::unordered_set<vid_t> free_list;
    std::unordered_map<label_t, vid_t> label_to_id;
    std::vector<label_t> id_to_label;

    vid_t allocate_id(label_t label);

    void free_id(label_t label);

    const vid_t get_id(label_t label) const;

    const label_t get_label(vid_t vid) const;
};

struct VectorStore {
    size_t d;
    std::vector<float> vecs;

    VectorStore(size_t d);

    void add_vector(const float* vec, vid_t vid);

    void remove_vector(vid_t vid);

    const float* get_vec(vid_t vid) const;
};

struct AccessMatrix {
    std::vector<std::unordered_set<tid_t>> access_matrix;

    void add_vector(vid_t vid, tid_t tid);

    void remove_vector(vid_t vid, tid_t tid);

    void grant_access(vid_t vid, tid_t tid);

    void revoke_access(vid_t vid, tid_t tid);

    bool has_access(vid_t vid, tid_t tid) const;
};

struct TreeNode {
    /* information about the tree structure */
    size_t level;       // the level of this node in the tree
    size_t sibling_id;  // the id of this node among its siblings
    TreeNode* parent;
    size_t n_clusters;  // number of children nodes
    std::vector<TreeNode*> children;

    /* information about the cluster */
    float* centroid;
    IndexFlatL2 quantizer;
    
    /* available for all nodes */
    bloom_filter bf;
    std::unordered_map<tid_t, std::vector<vid_t>> shortlists;

    /* only for leaf nodes */
    std::vector<vid_t> vector_indices;  // vectors assigned to this leaf node
    std::unordered_map<tid_t, size_t> n_vectors_per_tenant;

    TreeNode(
            size_t level,
            size_t sibling_id,
            TreeNode* parent,
            float* centroid,
            size_t d,
            size_t n_clusters,
            size_t bf_capacity,
            float bf_false_pos);

    ~TreeNode();
};

struct MultiTenantIndexIVFHierarchical : MultiTenantIndexIVFFlat {
    // TODO: try to remove these constants
    static constexpr size_t MIN_POINTS_PER_CENTROID = 8;
    static constexpr size_t MAX_LEVEL = 8;
    static constexpr float N_CLUSTER_DIVISOR = 2.0;
    static constexpr size_t MIN_N_CLUSTERS = 4;
    static constexpr size_t BF_UPDATE_INTERVAL = 100;

    /* construction parameters */
    size_t bf_capacity;
    float bf_false_pos;
    size_t max_sl_size;

    /* search parameters */
    float gamma1, gamma2;

    /* main data structures */
    TreeNode tree_root;
    IdAllocator id_allocator;
    VectorStore vec_store;
    AccessMatrix access_matrix;
    
    /* auxiliary data structures */
    size_t update_bf_after;
    std::unordered_map<label_t, TreeNode*> label_to_leaf;

    MultiTenantIndexIVFHierarchical(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType metric = METRIC_L2, 
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            float gamma1 = 16,
            float gamma2 = 256,
            size_t max_sl_size = 128);

    /*
     * API functions
    */

    void train(idx_t n, const float* x, tid_t tid) override;

    void train_helper(TreeNode* node, idx_t n, const float* x);

    void add_vector_with_ids(
            idx_t n,
            const float* x,
            const label_t* labels,
            tid_t tid) override;

    void grant_access(label_t label, tid_t tid) override;

    void grant_access_helper(TreeNode* node, label_t label, tid_t tid, std::vector<idx_t>& path);

    bool remove_vector(label_t label, tid_t tid) override;

    bool revoke_access(label_t label, tid_t tid) override;

    void update_shortlists_helper(TreeNode* leaf, vid_t vid, std::unordered_set<tid_t>& tenants);

    void update_bf_helper(TreeNode* leaf);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            label_t* labels,
            const SearchParameters* params = nullptr) const override;

     void search_one(
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            label_t* labels,
            const SearchParameters* params = nullptr) const;

    /*
     * Helper functions
    */

    TreeNode* assign_vec_to_leaf(const float* x);

    std::vector<idx_t> get_vector_path(label_t label) const;

    void split_short_list(TreeNode* node, tid_t tid);

    bool merge_short_list(TreeNode* node, tid_t tid);

    bool merge_short_list_recursively(TreeNode* node, tid_t tid);
};

} // namespace faiss

#endif
