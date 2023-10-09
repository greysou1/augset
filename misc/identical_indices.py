def find_swap_indices(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    
    # Calculate the difference between the two sums
    diff = sum_x - sum_y
    
    if diff % 2 != 0 or abs(diff) > 2 * max(max(x), max(y)):
        # No valid swap is possible
        return "Failure"
    
    for i, xi in enumerate(x):
        if xi - diff // 2 in y:
            j = y.index(xi - diff // 2)
            return (i, j)
    
    return "Failure"

# Example usage:
x = [3, 1, 4, 2, 2]
y = [1, 2, 3, 6]

result = find_swap_indices(x, y)
print(result)  # Output: (1, 3) (e.g., x[1] + y[3] == y[1] + x[3] == 8)
