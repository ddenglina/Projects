def movingCount(threshold, rows, cols):

    def sums(i, j):
        s1 = s2 = 0
        while i != 0:
            s1 = s1 + i % 10
            i = i // 10
        while j != 0:
            s2 = s2 + j % 10
            j = j // 10
        return s1 + s2

    def dfs(i, j,count):
        if i < 0 or i >= cols or j < 0 or j >= rows or sums(i, j) > threshold or matrix[i][j]==True:
            return

        matrix[i][j] = True
        print(matrix)
        count+=1
        print(count)
        dfs(i + 1, j,count)
        dfs(i - 1, j,count)
        dfs(i, j + 1,count)
        dfs(i, j - 1,count)
        return count

    count = 0
    matrix = [[-1 for i in range(cols)] for j in range(rows)]
    tmp = dfs(0, 0,count)
    return tmp

res = movingCount(1,2,3)
print(res)