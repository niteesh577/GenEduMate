#include <iostream>
#include <vector>
#include <unordered_map>

std::vector<int> twoSum(std::vector<int>& nums, int target) {
    std::unordered_map<int, int> num_map; // Map to store the complement and index
    for (int i = 0; i < nums.size(); ++i) {
        int complement = target - nums[i];
        // Check if complement exists in map
        if (num_map.find(complement) != num_map.end()) {
            return {num_map[complement], i}; // Return the indices
        }
        num_map[nums[i]] = i; // Add the number and its index to the map
    }
    return {}; // Return an empty vector if no solution is found
}

int main() {
    std::vector<int> nums = {2, 7, 11, 15};
    int target = 9;
    std::vector<int> result = twoSum(nums, target);

    if (!result.empty()) {
        std::cout << "Indices: " << result[0] << ", " << result[1] << std::endl;
    } else {
        std::cout << "No solution found." << std::endl;
    }

    return 0;
}