def mergeOverlappingPairsAllTheWay(pairs):
    def mergeOverlappingPairs(pairs):
        """Merge pairs that share at least one common element"""
        if not pairs:
            return []
        
        result = [set(pairs[0])]
        
        for pair in pairs[1:]:
            pair_set = set(pair)
            merged = False
            
            for i, existing_group in enumerate(result):
                if pair_set & existing_group:  # If there's any overlap
                    result[i] = existing_group | pair_set
                    merged = True
                    break
            
            if not merged:
                result.append(pair_set)
        
        # Convert back to sorted lists
        return [sorted(list(group)) for group in result]

    data1 = mergeOverlappingPairs(pairs)
    data2 = []
    while data2 != data1:
        data2 = data1.copy()
        data1 = mergeOverlappingPairs(data1)
    
    return data1

# Test cases
def test_merge_function():
    print("Testing mergeOverlappingPairsAllTheWay function\n")
    
    # Test 1: Your original example
    test1 = [[1, 2], [3, 4], [2, 4], [5, 6], [2, 7]]
    result1 = mergeOverlappingPairsAllTheWay(test1)
    print(f"Test 1 - Input: {test1}")
    print(f"Output: {result1}")
    print(f"Expected: [[1, 2, 3, 4, 7], [5, 6]] (connected components)")
    print(f"Correct: {result1 == [[1, 2, 3, 4, 7], [5, 6]]}\n")
    
    # Test 2: No overlaps
    test2 = [[1, 2], [3, 4], [5, 6]]
    result2 = mergeOverlappingPairsAllTheWay(test2)
    print(f"Test 2 - Input: {test2}")
    print(f"Output: {result2}")
    print(f"Expected: [[1, 2], [3, 4], [5, 6]] (no merging)")
    print(f"Correct: {result2 == [[1, 2], [3, 4], [5, 6]]}\n")
    
    # Test 3: Chain connection
    test3 = [[1, 2], [2, 3], [3, 4], [5, 6]]
    result3 = mergeOverlappingPairsAllTheWay(test3)
    print(f"Test 3 - Input: {test3}")
    print(f"Output: {result3}")
    print(f"Expected: [[1, 2, 3, 4], [5, 6]] (chain merging)")
    print(f"Correct: {result3 == [[1, 2, 3, 4], [5, 6]]}\n")
    
    # Test 4: All connected
    test4 = [[1, 2], [2, 3], [3, 4], [4, 5]]
    result4 = mergeOverlappingPairsAllTheWay(test4)
    print(f"Test 4 - Input: {test4}")
    print(f"Output: {result4}")
    print(f"Expected: [[1, 2, 3, 4, 5]] (all in one group)")
    print(f"Correct: {result4 == [[1, 2, 3, 4, 5]]}\n")
    
    # Test 5: Empty input
    test5 = []
    result5 = mergeOverlappingPairsAllTheWay(test5)
    print(f"Test 5 - Input: {test5}")
    print(f"Output: {result5}")
    print(f"Expected: [] (empty input)")
    print(f"Correct: {result5 == []}\n")
    
    # Test 6: Single pair
    test6 = [[1, 2]]
    result6 = mergeOverlappingPairsAllTheWay(test6)
    print(f"Test 6 - Input: {test6}")
    print(f"Output: {result6}")
    print(f"Expected: [[1, 2]] (single pair)")
    print(f"Correct: {result6 == [[1, 2]]}\n")
    
    # Test 7: Complex branching
    test7 = [[1, 2], [1, 3], [2, 4], [5, 6], [6, 7], [8, 9]]
    result7 = mergeOverlappingPairsAllTheWay(test7)
    print(f"Test 7 - Input: {test7}")
    print(f"Output: {result7}")
    print(f"Expected: [[1, 2, 3, 4], [5, 6, 7], [8, 9]] (multiple groups)")
    expected7 = [[1, 2, 3, 4], [5, 6, 7], [8, 9]]
    print(f"Correct: {sorted(result7) == sorted(expected7)}\n")

if __name__ == "__main__":
    test_merge_function()