# 1. Quick Sort (Divide and Conquer)
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print(sorted_arr)

# 2. Strassen’s Matrix Multiplication (Divide and Conquer)
import numpy as np

def strassen(A, B):
    n = len(A)

    if n == 1:
        return A * B

    mid = n // 2

    # Splitting matrices into 4 parts
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    # Calculating the 7 products
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # Computing the result submatrices
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combining submatrices
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

# Example (4x4 matrices only)
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 1, 2, 3],
              [4, 5, 6, 7]])

B = np.array([[7, 6, 5, 4],
              [3, 2, 1, 0],
              [9, 8, 7, 6],
              [5, 4, 3, 2]])

C = strassen(A, B)

print("Result of A x B using Strassen:")
print(C)

# 3. Dijkstra’s Algorithm (Greedy)

import heapq

def dijkstra(graph, start):
    pq = [(0, start)]  
    dist = {vertex: float('inf') for vertex in graph} 
    dist[start] = 0  
    
    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:  
            if current_dist + weight < dist[v]:
                dist[v] = current_dist + weight  
                heapq.heappush(pq, (dist[v], v))  

    return dist
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

start_node = 'A'
shortest_distances = dijkstra(graph, start_node)

print(f"Shortest distances from {start_node}:")
for node, distance in shortest_distances.items():
    print(f"{node}: {distance}")

# 4. Kruskal’s Algorithm (Greedy)
def kruskal(edges, n):
    parent = list(range(n))
    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        root1 = find(u)
        root2 = find(v)
        if root1 != root2:
            parent[root2] = root1
            return True
        return False
    mst = []
    edges.sort(key=lambda x: x[2])
    for u, v, weight in edges:
        if union(u, v):
            mst.append((u, v, weight))
    return mst

edges = [
    (0, 1, 10),
    (0, 2, 6),
    (0, 3, 5),
    (1, 3, 15),
    (2, 3, 4)
]
n = 4

mst = kruskal(edges, n)

print("Edges in MST:")
for u, v, w in mst:
    print(f"{u} - {v}: {w}")

# 5. Floyd-Warshall (Dynamic Programming)
def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

graph = [
    [0, 1, 4, float('inf')],
    [1, 0, 2, 5],
    [4, 2, 0, 1],
    [float('inf'), 5, 1, 0]
]
flloyd_result = floyd_warshall(graph)
print("Floyd-Warshall Result:")
for row in flloyd_result:
    print(row)

# 6. Longest Common Subsequence (Dynamic Programming)
def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1 
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) 

    return dp[m][n]

str1 = "AGGTAB"
str2 = "GXTXAYB"

result = longest_common_subsequence(str1, str2)
print(f"Length of Longest Common Subsequence: {result}")


# 7. 8-Queens Problem (Backtracking)
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or board[i] - i == col - row or board[i] + i == col + row:
                return False
        return True
    def solve(row):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(row + 1)
    result = []
    board = [-1]*n
    solve(0)
    return result

nqueenres= solve_n_queens(4)
print("N-Queen Solution")
print(nqueenres)

# 8. 15 Puzzle Problem (Branch and Bound with A*)
import copy
N = 4
row = [1, 0, -1, 0]
col = [0, -1, 0, 1]
GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

def manhattan(puzzle):
    dist = 0
    for i in range(N):
        for j in range(N):
            val = puzzle[i][j]
            if val == 0:
                continue
            goal_x = (val - 1) // N
            goal_y = (val - 1) % N
            dist += abs(i - goal_x) + abs(j - goal_y)
    return dist

def find_blank(board):
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                return i, j

class Node:
    def __init__(self, board, x, y, level, parent):
        self.board = board
        self.x = x
        self.y = y
        self.level = level
        self.cost = manhattan(board) + level
        self.parent = parent
    def __lt__(self, other):
        return self.cost < other.cost

def is_safe(x, y):
    return 0 <= x < N and 0 <= y < N

def print_board(board):
    for row in board:
        print(' '.join(str(cell).rjust(2) for cell in row))
    print()

def trace_path(node):
    path = []
    while node:
        path.append(node.board)
        node = node.parent
    path.reverse()
    for step in path:
        print_board(step)

def solve_puzzle(start_board):
    x, y = find_blank(start_board)
    root = Node(start_board, x, y, 0, None)
    pq = []
    heapq.heappush(pq, root)
    visited = set()
    while pq:
        current = heapq.heappop(pq)
        flat_board = sum(current.board, [])
        if flat_board == GOAL_STATE:
            print("Solution found in", current.level, "moves.")
            trace_path(current)
            return
        visited.add(tuple(flat_board))
        for i in range(4):
            new_x = current.x + row[i]
            new_y = current.y + col[i]
            if is_safe(new_x, new_y):
                new_board = copy.deepcopy(current.board)
                new_board[current.x][current.y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[current.x][current.y]
                flat_new = sum(new_board, [])
                if tuple(flat_new) not in visited:
                    heapq.heappush(pq, Node(new_board, new_x, new_y, current.level + 1, current))
start_board = [
    [1, 2, 3, 4],
    [5, 6, 0, 8],
    [9, 10, 7, 12],
    [13, 14, 11, 15]
]

solve_puzzle(start_board)

# 9. String Matching (Naive)
def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            return i
    return -1

text = "hello world"
pattern = "world"

index = naive_search(text, pattern)
print("Pattern found at index:", index)

# 10. Insertion Sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr


arr = [12, 11, 13, 5, 6]
sorted_arr = insertion_sort(arr)
print("Sorted array using Insertion Sort:", sorted_arr)