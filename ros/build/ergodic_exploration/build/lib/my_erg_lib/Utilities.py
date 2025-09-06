from scipy.linalg import logm, expm  # Add this import
import numpy as np
def logEuclideanMean(covariance_matrices):
    """Compute log-Euclidean mean of covariance matrices"""
    log_matrices = [logm(sigma) for sigma in covariance_matrices]
    mean_log = np.mean(log_matrices, axis=0)
    return expm(mean_log)


def mergeOverlappingPairsAllTheWay(pairs):
    """
    Merge all overlapping pairs into connected groups.

    Returns:
        List of merged groups where each group contains all connected elements
    Example:
        [[1, 2], [2, 3], [4, 5]] -> [[1, 2, 3], [4, 5]]
    """
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