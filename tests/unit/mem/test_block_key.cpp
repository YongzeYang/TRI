/**
 * @file test_block_key.cpp
 * @brief Unit tests for BlockKey
 * @author Yongze
 * @date 2025-08-14
 */

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "tri/mem/block_key.hpp"

namespace tri::test {

TEST(BlockKey, DefaultConstruction) {
    mem::BlockKey key;
    ASSERT_EQ(0u, key.block_row);
    ASSERT_EQ(0u, key.block_col);
}

TEST(BlockKey, ParameterizedConstruction) {
    mem::BlockKey key(3, 5);
    ASSERT_EQ(3u, key.block_row);
    ASSERT_EQ(5u, key.block_col);
}

TEST(BlockKey, EqualityOperator) {
    mem::BlockKey key1(2, 3);
    mem::BlockKey key2(2, 3);
    mem::BlockKey key3(2, 4);
    mem::BlockKey key4(3, 3);

    ASSERT_TRUE(key1 == key2);
    ASSERT_FALSE(key1 == key3);
    ASSERT_FALSE(key1 == key4);
}

TEST(BlockKey, InequalityOperator) {
    mem::BlockKey key1(2, 3);
    mem::BlockKey key2(2, 3);
    mem::BlockKey key3(2, 4);

    ASSERT_FALSE(key1 != key2);
    ASSERT_TRUE(key1 != key3);
}

TEST(BlockKey, LessThanOperator) {
    mem::BlockKey key1(1, 2);
    mem::BlockKey key2(1, 3);
    mem::BlockKey key3(2, 1);

    ASSERT_TRUE(key1 < key2);  // Same row, different col
    ASSERT_TRUE(key1 < key3);  // Different row
    ASSERT_FALSE(key2 < key1);
    ASSERT_FALSE(key3 < key1);
}

TEST(BlockKey, HashFunction) {
    std::unordered_map<mem::BlockKey, int> map;

    mem::BlockKey key1(1, 2);
    mem::BlockKey key2(2, 1);
    mem::BlockKey key3(1, 2);  // Same as key1

    map[key1] = 10;
    map[key2] = 20;
    map[key3] = 30;  // Should overwrite key1's value

    ASSERT_EQ(2u, map.size());  // Only 2 unique keys
    ASSERT_EQ(30, map[key1]);   // Value was overwritten
    ASSERT_EQ(20, map[key2]);
}

TEST(BlockKey, UnorderedSet) {
    std::unordered_set<mem::BlockKey> set;

    set.insert(mem::BlockKey(1, 2));
    set.insert(mem::BlockKey(2, 1));
    set.insert(mem::BlockKey(1, 2));  // Duplicate

    ASSERT_EQ(2u, set.size());
    ASSERT_TRUE(set.find(mem::BlockKey(1, 2)) != set.end());
    ASSERT_TRUE(set.find(mem::BlockKey(2, 1)) != set.end());
    ASSERT_FALSE(set.find(mem::BlockKey(3, 3)) != set.end());
}

TEST(BlockKey, StreamOutput) {
    mem::BlockKey key(3, 7);
    std::ostringstream oss;
    oss << key;

    ASSERT_EQ("(3,7)", oss.str());
}

TEST(BlockKey, LargeValues) {
    std::size_t large_val = 1000000;
    mem::BlockKey key(large_val, large_val + 1);

    ASSERT_EQ(large_val, key.block_row);
    ASSERT_EQ(large_val + 1, key.block_col);

    // Test hash doesn't overflow or cause issues
    std::hash<mem::BlockKey> hasher;
    std::size_t hash_val = hasher(key);
    ASSERT_NE(0u, hash_val);  // Should produce some hash value
}

}  // namespace tri::test