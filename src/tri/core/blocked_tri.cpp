/**
 * @file blocked_tri.cpp
 * @brief Blocked lower triangular matrix implementation
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/core/blocked_tri.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"
#include "tri/mem/access_count_policy.hpp"
#include "tri/mem/block_key.hpp"
#include "tri/mem/block_manager.hpp"
#include "tri/mem/block_storage.hpp"
#include "tri/mem/lru_policy.hpp"

namespace tri {
namespace core {

// Data persistance implementations
template <typename T>
std::string BlockedTriMatrix<T>::generate_matrix_id() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    std::ostringstream oss;
    oss << "matrix_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << "_" << dis(gen);
    return oss.str();
}

template <typename T>
bool BlockedTriMatrix<T>::set_storage_backend(StorageType type, const std::string& disk_path) {
    if (matrix_id_.empty()) {
        matrix_id_ = generate_matrix_id();
    }

    switch (type) {
        case StorageType::Memory:
            storage_backend_ = std::make_shared<mem::MemoryStorage<T>>();
            break;

        case StorageType::Disk:
            storage_backend_ = std::make_shared<mem::DiskStorage<T>>(disk_path, matrix_id_, false);
            break;

        case StorageType::Hybrid:
            storage_backend_ = std::make_shared<mem::HybridStorage<T>>(
                100, disk_path, matrix_id_);  // Cache up to 100 blocks in memory
            break;

        default:
            return false;
    }

    return true;
}

// Update load_block_callback to use storage backend:
template <typename T>
bool BlockedTriMatrix<T>::load_block_callback(const mem::BlockKey& key) {
    size_type idx = get_linear_block_index(key.block_row, key.block_col);
    if (idx >= blocks_.size()) return false;

    if (!blocks_[idx].is_loaded) {
        size_type rows = get_block_actual_rows(key.block_row);
        size_type cols = get_block_actual_cols(key.block_col);
        size_type block_elements = rows * cols;

        blocks_[idx].data = std::make_unique<T[]>(block_elements);

        // Try to load from storage backend
        bool loaded = false;
        if (storage_backend_ && storage_backend_->exists(key)) {
            loaded = storage_backend_->load(key, blocks_[idx].data.get(), block_elements);
        }

        if (!loaded) {
            // Initialize with zeros for new blocks
            std::fill_n(blocks_[idx].data.get(), block_elements, T{0});
        }

        blocks_[idx].is_loaded = true;
    }
    return true;
}

// Update unload_block_callback to use storage backend:
template <typename T>
bool BlockedTriMatrix<T>::unload_block_callback(const mem::BlockKey& key) {
    size_type idx = get_linear_block_index(key.block_row, key.block_col);
    if (idx >= blocks_.size()) return false;

    if (blocks_[idx].is_loaded && blocks_[idx].data) {
        // Save to storage backend before unloading
        if (storage_backend_) {
            size_type rows = get_block_actual_rows(key.block_row);
            size_type cols = get_block_actual_cols(key.block_col);
            size_type block_elements = rows * cols;

            storage_backend_->store(key, blocks_[idx].data.get(), block_elements);
        }

        // Now unload from memory
        blocks_[idx].data.reset();
        blocks_[idx].is_loaded = false;
    }
    return true;
}

// Update clear() to clean up storage:
template <typename T>
void BlockedTriMatrix<T>::clear() noexcept {
    n_ = 0;
    block_size_ = 0;
    num_blocks_ = 0;
    blocks_.clear();
    block_remap_.clear();
    pinned_blocks_ = 0;

    // Clear storage backend
    if (storage_backend_) {
        storage_backend_->clear();
    }
}

// Block implementation
template <typename T>
BlockedTriMatrix<T>::Block::Block(size_type size)
    : data(std::make_unique<T[]>(size)), is_loaded(true) {
    std::fill_n(data.get(), size, T{0});
}

// BlockedTriMatrix implementation
template <typename T>
void BlockedTriMatrix<T>::init(size_type n, size_type block_size) {
    n_ = n;
    block_size_ = block_size;
    if (block_size_ == 0) {
        block_size_ = tri::config::DEFAULT_BLOCK_SIZE;
    }

    // Calculate number of blocks in each dimension
    num_blocks_ = (n_ + block_size_ - 1) / block_size_;

    // Allocate blocks (only lower triangular)
    allocate_blocks();
}

// Matrix constructor implementation
template <typename T>
template <typename MatrixType, typename>
BlockedTriMatrix<T>::BlockedTriMatrix(const MatrixType& dense, size_type block_size) {
    if (dense.rows() != dense.cols()) {
        throw std::invalid_argument("Matrix must be square for BlockedTriMatrix");
    }

    init(dense.rows(), block_size);

    // Copy data from dense matrix
    for (size_type i = 0; i < n_; ++i) {
        for (size_type j = 0; j <= i; ++j) {
            set(i, j, static_cast<T>(dense(i, j)));
        }
    }
}

template <typename T>
BlockedTriMatrix<T>::BlockedTriMatrix(const BlockedTriMatrix& other)
    : n_(other.n_), block_size_(other.block_size_), num_blocks_(other.num_blocks_) {
    // Deep copy blocks
    blocks_.resize(other.blocks_.size());
    for (size_type i = 0; i < blocks_.size(); ++i) {
        if (other.blocks_[i].is_loaded && other.blocks_[i].data) {
            size_type block_linear_idx = i;
            size_type block_row = 0, block_col = 0, blocks_seen = 0;

            // Find block_row and block_col from linear index
            for (size_type br = 0; br < num_blocks_; ++br) {
                size_type blocks_in_row = br + 1;
                if (blocks_seen + blocks_in_row > block_linear_idx) {
                    block_row = br;
                    block_col = block_linear_idx - blocks_seen;
                    break;
                }
                blocks_seen += blocks_in_row;
            }

            size_type rows = get_block_actual_rows(block_row);
            size_type cols =
                get_block_actual_cols(block_col);  // Fixed: use block_col instead of block_row
            size_type block_elements = rows * cols;

            blocks_[i].data = std::make_unique<T[]>(block_elements);
            std::copy_n(other.blocks_[i].data.get(), block_elements, blocks_[i].data.get());
            blocks_[i].access_count = other.blocks_[i].access_count;
            blocks_[i].is_loaded = true;
        }
    }
}

template <typename T>
BlockedTriMatrix<T>& BlockedTriMatrix<T>::operator=(const BlockedTriMatrix& other) {
    if (this != &other) {
        BlockedTriMatrix temp(other);
        swap(temp);
    }
    return *this;
}

template <typename T>
void BlockedTriMatrix<T>::allocate_blocks() {
    // Calculate total number of lower triangular blocks
    size_type total = total_blocks();
    blocks_.resize(total);

    // Initialize all blocks
    for (size_type block_row = 0; block_row < num_blocks_; ++block_row) {
        for (size_type block_col = 0; block_col <= block_row; ++block_col) {
            allocate_block(block_row, block_col);
        }
    }
}

template <typename T>
void BlockedTriMatrix<T>::allocate_block(size_type block_row, size_type block_col) {
    if (!is_block_in_lower_triangle(block_row, block_col)) {
        return;
    }

    size_type idx = get_linear_block_index(block_row, block_col);
    if (idx >= blocks_.size()) {
        return;
    }

    size_type rows = get_block_actual_rows(block_row);
    size_type cols = get_block_actual_cols(block_col);
    size_type block_elements = rows * cols;

    blocks_[idx] = Block(block_elements);
}

template <typename T>
void BlockedTriMatrix<T>::deallocate_block(size_type block_row, size_type block_col) {
    if (!is_block_in_lower_triangle(block_row, block_col)) {
        return;
    }

    size_type idx = get_linear_block_index(block_row, block_col);
    if (idx < blocks_.size()) {
        blocks_[idx].data.reset();
        blocks_[idx].is_loaded = false;
    }
}

// Updated operator() to use block manager
template <typename T>
typename BlockedTriMatrix<T>::reference BlockedTriMatrix<T>::operator()(size_type i, size_type j) {
    if (i >= n_ || j >= n_) {
        throw std::out_of_range("BlockedTriMatrix: Index out of range");
    }

    if (j > i) {
        throw std::logic_error("Cannot modify upper triangular elements in BlockedTriMatrix");
    }

    BlockIndex idx = compute_block_index(i, j);

    // Notify block manager of access
    if (block_manager_) {
        mem::BlockKey key(idx.block_row, idx.block_col);
        block_manager_->on_access(key);
    }

    Block* block = get_block(idx.block_row, idx.block_col);

    if (!block || !block->data) {
        if (!block_manager_) {
            throw std::runtime_error("Block not allocated and no block manager available");
        }
        // Block should be loaded by block_manager_->on_access
        block = get_block(idx.block_row, idx.block_col);
        if (!block || !block->data) {
            throw std::runtime_error("Block failed to load");
        }
    }

    block->access_count++;

    // Calculate actual block dimensions for proper indexing
    size_type block_cols = get_block_actual_cols(idx.block_col);
    size_type index = idx.row_offset * block_cols + idx.col_offset;

    return block->data[index];
}

template <typename T>
typename BlockedTriMatrix<T>::const_reference BlockedTriMatrix<T>::operator()(size_type i,
                                                                              size_type j) const {
    if (i >= n_ || j >= n_) {
        throw std::out_of_range("BlockedTriMatrix: Index out of range");
    }

    if (j > i) {
        return zero_value_;
    }

    BlockIndex idx = compute_block_index(i, j);

    // Notify block manager of access (const_cast is safe here as we only update metadata)
    if (block_manager_) {
        mem::BlockKey key(idx.block_row, idx.block_col);
        const_cast<BlockedTriMatrix*>(this)->block_manager_->on_access(key);
    }

    const Block* block = get_block(idx.block_row, idx.block_col);

    if (!block || !block->data) {
        if (block_manager_) {
            // Try to get block again after manager access
            block = get_block(idx.block_row, idx.block_col);
        }
        if (!block || !block->data) {
            return zero_value_;
        }
    }

    const_cast<Block*>(block)->access_count++;

    // Calculate actual block dimensions for proper indexing
    size_type block_cols = get_block_actual_cols(idx.block_col);
    size_type index = idx.row_offset * block_cols + idx.col_offset;

    return block->data[index];
}

template <typename T>
T BlockedTriMatrix<T>::at(size_type i, size_type j) const {
    if (j > i || i >= n_ || j >= n_) {
        return T{0};
    }

    // For const access with block manager, we need to cast away constness
    // This is safe because block loading doesn't change the logical state
    if (block_manager_) {
        return const_cast<BlockedTriMatrix<T>*>(this)->operator()(i, j);
    }

    BlockIndex idx = compute_block_index(i, j);
    const Block* block = get_block(idx.block_row, idx.block_col);

    if (!block || !block->data) {
        return T{0};
    }

    // Calculate actual block dimensions for proper indexing
    size_type block_cols = get_block_actual_cols(idx.block_col);
    size_type index = idx.row_offset * block_cols + idx.col_offset;

    return block->data[index];
}

template <typename T>
void BlockedTriMatrix<T>::set(size_type i, size_type j, const T& value) {
    if (i >= n_ || j >= n_) {
        throw std::out_of_range("BlockedTriMatrix: Index out of range");
    }

    if (j > i) {
        throw std::logic_error("Cannot set upper triangular elements in BlockedTriMatrix");
    }

    BlockIndex idx = compute_block_index(i, j);
    Block* block = get_block(idx.block_row, idx.block_col);

    if (!block || !block->data) {
        allocate_block(idx.block_row, idx.block_col);
        block = get_block(idx.block_row, idx.block_col);
    }

    // Calculate actual block dimensions for proper indexing
    size_type block_cols = get_block_actual_cols(idx.block_col);
    size_type index = idx.row_offset * block_cols + idx.col_offset;

    block->data[index] = value;
}

template <typename T>
typename BlockedTriMatrix<T>::Block* BlockedTriMatrix<T>::get_block(size_type block_row,
                                                                    size_type block_col) {
    if (!is_block_in_lower_triangle(block_row, block_col)) {
        return nullptr;
    }

    size_type idx = get_linear_block_index(block_row, block_col);
    if (idx >= blocks_.size()) {
        return nullptr;
    }

    return &blocks_[idx];
}

template <typename T>
const typename BlockedTriMatrix<T>::Block* BlockedTriMatrix<T>::get_block(
    size_type block_row, size_type block_col) const {
    if (!is_block_in_lower_triangle(block_row, block_col)) {
        return nullptr;
    }

    size_type idx = get_linear_block_index(block_row, block_col);
    if (idx >= blocks_.size()) {
        return nullptr;
    }

    return &blocks_[idx];
}

template <typename T>
typename BlockedTriMatrix<T>::pointer BlockedTriMatrix<T>::get_block_data(size_type block_row,
                                                                          size_type block_col) {
    Block* block = get_block(block_row, block_col);
    return block ? block->data.get() : nullptr;
}

template <typename T>
typename BlockedTriMatrix<T>::const_pointer BlockedTriMatrix<T>::get_block_data(
    size_type block_row, size_type block_col) const {
    const Block* block = get_block(block_row, block_col);
    return block ? block->data.get() : nullptr;
}

template <typename T>
typename BlockedTriMatrix<T>::BlockIndex BlockedTriMatrix<T>::compute_block_index(
    size_type i, size_type j) const noexcept {
    if (j > i) {
        return BlockIndex::invalid();
    }

    BlockIndex result;
    result.block_row = i / block_size_;
    result.block_col = j / block_size_;
    result.row_offset = i % block_size_;
    result.col_offset = j % block_size_;

    return result;
}

template <typename T>
typename BlockedTriMatrix<T>::size_type BlockedTriMatrix<T>::get_linear_block_index(
    size_type block_row, size_type block_col) const noexcept {
    if (block_col > block_row || block_row >= num_blocks_) {
        return static_cast<size_type>(-1);
    }

    // For lower triangular storage: block at (r,c) has index r*(r+1)/2 + c
    return block_row * (block_row + 1) / 2 + block_col;
}

template <typename T>
bool BlockedTriMatrix<T>::is_block_in_lower_triangle(size_type block_row,
                                                     size_type block_col) const noexcept {
    return block_col <= block_row && block_row < num_blocks_ && block_col < num_blocks_;
}

template <typename T>
typename BlockedTriMatrix<T>::size_type BlockedTriMatrix<T>::total_blocks() const noexcept {
    // Total lower triangular blocks: n*(n+1)/2
    return num_blocks_ * (num_blocks_ + 1) / 2;
}

template <typename T>
typename BlockedTriMatrix<T>::size_type BlockedTriMatrix<T>::allocated_blocks() const noexcept {
    size_type count = 0;
    for (const auto& block : blocks_) {
        if (block.is_loaded && block.data) {
            count++;
        }
    }
    return count;
}

template <typename T>
typename BlockedTriMatrix<T>::size_type BlockedTriMatrix<T>::get_block_actual_rows(
    size_type block_row) const noexcept {
    size_type start_row = block_row * block_size_;
    size_type end_row = std::min(start_row + block_size_, n_);
    return end_row - start_row;
}

template <typename T>
typename BlockedTriMatrix<T>::size_type BlockedTriMatrix<T>::get_block_actual_cols(
    size_type block_col) const noexcept {
    size_type start_col = block_col * block_size_;
    size_type end_col = std::min(start_col + block_size_, n_);
    return end_col - start_col;
}

template <typename T>
void BlockedTriMatrix<T>::export_to_dense(pointer dest, size_type ld) const {
    if (!dest) {
        throw std::invalid_argument("Destination pointer is null");
    }

    // Initialize to zero
    for (size_type i = 0; i < n_; ++i) {
        for (size_type j = 0; j < n_; ++j) {
            dest[i * ld + j] = T{0};
        }
    }

    // Copy lower triangular elements
    for (size_type i = 0; i < n_; ++i) {
        for (size_type j = 0; j <= i; ++j) {
            dest[i * ld + j] = at(i, j);
        }
    }
}

template <typename T>
std::vector<T> BlockedTriMatrix<T>::to_dense_vector() const {
    std::vector<T> result(n_ * n_, T{0});
    export_to_dense(result.data(), n_);
    return result;
}

// Updated load_block to use block manager
template <typename T>
void BlockedTriMatrix<T>::load_block(size_type block_row, size_type block_col) {
    if (!is_block_in_lower_triangle(block_row, block_col)) {
        return;
    }

    if (block_manager_) {
        mem::BlockKey key(block_row, block_col);
        block_manager_->load(key);
    } else {
        size_type idx = get_linear_block_index(block_row, block_col);
        if (idx < blocks_.size() && !blocks_[idx].is_loaded) {
            allocate_block(block_row, block_col);
        }
    }
}

// Updated unload_block to use block manager
template <typename T>
void BlockedTriMatrix<T>::unload_block(size_type block_row, size_type block_col) {
    if (!is_block_in_lower_triangle(block_row, block_col)) {
        return;
    }

    if (block_manager_) {
        mem::BlockKey key(block_row, block_col);
        block_manager_->evict(key);
    } else {
        deallocate_block(block_row, block_col);
    }
}

// Updated pin_block to use block manager
template <typename T>
void BlockedTriMatrix<T>::pin_block(size_type block_row, size_type block_col) {
    if (is_block_in_lower_triangle(block_row, block_col)) {
        if (block_manager_) {
            mem::BlockKey key(block_row, block_col);
            block_manager_->pin(key);
        }
        pinned_blocks_++;
    }
}

// Updated unpin_block to use block manager
template <typename T>
void BlockedTriMatrix<T>::unpin_block(size_type block_row, size_type block_col) {
    if (is_block_in_lower_triangle(block_row, block_col) && pinned_blocks_ > 0) {
        if (block_manager_) {
            mem::BlockKey key(block_row, block_col);
            block_manager_->unpin(key);
        }
        pinned_blocks_--;
    }
}

template <typename T>
typename BlockedTriMatrix<T>::size_type BlockedTriMatrix<T>::get_block_access_count(
    size_type block_row, size_type block_col) const {
    const Block* block = get_block(block_row, block_col);
    return block ? block->access_count : 0;
}

template <typename T>
void BlockedTriMatrix<T>::reset_access_counts() noexcept {
    for (auto& block : blocks_) {
        block.access_count = 0;
    }
}

template <typename T>
void BlockedTriMatrix<T>::resize(size_type new_n, size_type new_block_size) {
    if (new_block_size == 0) {
        new_block_size = block_size_ > 0 ? block_size_ : tri::config::DEFAULT_BLOCK_SIZE;
    }

    clear();
    init(new_n, new_block_size);
}

template <typename T>
void BlockedTriMatrix<T>::fill(const T& value) {
    for (size_type block_row = 0; block_row < num_blocks_; ++block_row) {
        for (size_type block_col = 0; block_col <= block_row; ++block_col) {
            fill_block(block_row, block_col, value);
        }
    }
}

template <typename T>
void BlockedTriMatrix<T>::fill_block(size_type block_row, size_type block_col, const T& value) {
    Block* block = get_block(block_row, block_col);
    if (block && block->data) {
        size_type rows = get_block_actual_rows(block_row);
        size_type cols = get_block_actual_cols(block_col);
        std::fill_n(block->data.get(), rows * cols, value);
    }
}

template <typename T>
void BlockedTriMatrix<T>::set_diagonal(const T& value) {
    for (size_type i = 0; i < n_; ++i) {
        set(i, i, value);
    }
}

template <typename T>
void BlockedTriMatrix<T>::swap(BlockedTriMatrix& other) noexcept {
    std::swap(n_, other.n_);
    std::swap(block_size_, other.block_size_);
    std::swap(num_blocks_, other.num_blocks_);
    blocks_.swap(other.blocks_);
    block_remap_.swap(other.block_remap_);
    std::swap(pinned_blocks_, other.pinned_blocks_);
    std::swap(max_memory_blocks_, other.max_memory_blocks_);
}

template <typename T>
BlockedTriMatrix<T> BlockedTriMatrix<T>::identity(size_type n, size_type block_size) {
    BlockedTriMatrix result(static_cast<int>(n), block_size);  // Explicitly use int constructor
    result.set_diagonal(T{1});
    return result;
}

template <typename T>
BlockedTriMatrix<T> BlockedTriMatrix<T>::zeros(size_type n, size_type block_size) {
    return BlockedTriMatrix(static_cast<int>(n), block_size,
                            T{0});  // Explicitly use int constructor
}

template <typename T>
void swap(BlockedTriMatrix<T>& lhs, BlockedTriMatrix<T>& rhs) noexcept {
    lhs.swap(rhs);
}

// New methods for block manager support
template <typename T>
void BlockedTriMatrix<T>::set_block_manager(std::shared_ptr<mem::BlockManager> manager) {
    block_manager_ = manager;
}

template <typename T>
bool BlockedTriMatrix<T>::create_block_manager(size_type max_blocks,
                                               const std::string& policy_name) {
    using namespace mem;

    // Set up default storage backend if not already set
    if (!storage_backend_) {
        set_storage_backend(StorageType::Memory);
    }

    auto load_cb = [this](const BlockKey& key) { return this->load_block_callback(key); };

    auto unload_cb = [this](const BlockKey& key) { return this->unload_block_callback(key); };

    if (policy_name == "LRU") {
        block_manager_ = make_lru_manager(max_blocks, load_cb, unload_cb);
    } else if (policy_name == "AccessCount") {
        mem::AccessCountPolicyConfig config;
        block_manager_ = make_access_count_manager(max_blocks, load_cb, unload_cb, config);
    } else {
        return false;
    }

    return true;
}

// Explicit instantiations
template class BlockedTriMatrix<float>;
template class BlockedTriMatrix<double>;
template class BlockedTriMatrix<int>;
template class BlockedTriMatrix<long>;

template void swap(BlockedTriMatrix<float>&, BlockedTriMatrix<float>&) noexcept;
template void swap(BlockedTriMatrix<double>&, BlockedTriMatrix<double>&) noexcept;
template void swap(BlockedTriMatrix<int>&, BlockedTriMatrix<int>&) noexcept;
template void swap(BlockedTriMatrix<long>&, BlockedTriMatrix<long>&) noexcept;

// Explicit instantiation of template constructors
template BlockedTriMatrix<float>::BlockedTriMatrix(const DenseRM<float>&, size_type);
template BlockedTriMatrix<double>::BlockedTriMatrix(const DenseRM<double>&, size_type);
template BlockedTriMatrix<int>::BlockedTriMatrix(const DenseRM<int>&, size_type);
template BlockedTriMatrix<long>::BlockedTriMatrix(const DenseRM<long>&, size_type);

template BlockedTriMatrix<float>::BlockedTriMatrix(const LowerTriangularRM<float>&, size_type);
template BlockedTriMatrix<double>::BlockedTriMatrix(const LowerTriangularRM<double>&, size_type);
template BlockedTriMatrix<int>::BlockedTriMatrix(const LowerTriangularRM<int>&, size_type);
template BlockedTriMatrix<long>::BlockedTriMatrix(const LowerTriangularRM<long>&, size_type);

// Explicit instantiation for integral type constructors
template BlockedTriMatrix<float>::BlockedTriMatrix<int>(int, size_type);
template BlockedTriMatrix<float>::BlockedTriMatrix<int>(int, size_type, const float&);
template BlockedTriMatrix<double>::BlockedTriMatrix<int>(int, size_type);
template BlockedTriMatrix<double>::BlockedTriMatrix<int>(int, size_type, const double&);
template BlockedTriMatrix<float>::BlockedTriMatrix<unsigned int>(unsigned int, size_type);
template BlockedTriMatrix<float>::BlockedTriMatrix<unsigned int>(unsigned int, size_type,
                                                                 const float&);
template BlockedTriMatrix<double>::BlockedTriMatrix<unsigned int>(unsigned int, size_type);
template BlockedTriMatrix<double>::BlockedTriMatrix<unsigned int>(unsigned int, size_type,
                                                                  const double&);

// Storage backend settings are implicitly instantiated as needed

// Explicit instantiations for block manager methods
// template void BlockedTriMatrix<float>::set_block_manager(std::shared_ptr<mem::BlockManager>);
// template void BlockedTriMatrix<double>::set_block_manager(std::shared_ptr<mem::BlockManager>);
// template void BlockedTriMatrix<int>::set_block_manager(std::shared_ptr<mem::BlockManager>);
// template void BlockedTriMatrix<long>::set_block_manager(std::shared_ptr<mem::BlockManager>);

// template bool BlockedTriMatrix<float>::create_block_manager(size_type, const std::string&);
// template bool BlockedTriMatrix<double>::create_block_manager(size_type, const std::string&);
// template bool BlockedTriMatrix<int>::create_block_manager(size_type, const std::string&);
// template bool BlockedTriMatrix<long>::create_block_manager(size_type, const std::string&);

}  // namespace core
}  // namespace tri