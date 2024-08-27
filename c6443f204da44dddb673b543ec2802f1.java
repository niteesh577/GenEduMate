def twoSum(nums, target):
    num_map = {}  # Dictionary to store the number and its index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]  # Return the indices
        num_map[num] = i  # Add the number and its index to the dictionary
    return []  # Return an empty list if no solution is found

def main():
    nums = [2, 7, 11, 15]
    target = 9
    result = twoSum(nums, target)

    if result:
        print(f"Indices: {result[0]}, {result[1]}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()