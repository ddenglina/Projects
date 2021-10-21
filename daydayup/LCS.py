def lcs(s, t):   # 最长公共子序列
    len1 = len(s)
    len2 = len(t)
    # 初始化一个二维数组，行数为t的大小，列数为s的大小
    res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if t[i - 1] == s[j - 1]:
                res[i][j] = 1 + res[i - 1][j - 1]
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    return res[-1][-1]

def lcs_string(s, t): # 最长公共子串
    len1 = len(s)
    len2 = len(t)
    # 初始化一个二维数组，行数为t的大小，列数为s的大小
    res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]
    # 声明一个变量，记录最大公共子串的值
    max_len = 0
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if t[i - 1] == s[j - 1]:
                res[i][j] = 1 + res[i - 1][j - 1]
            else:
                res[i][j] = 0 # 不等时，res=0
            max_len = max(max_len, res[i][j])
    return max_len

s = [3,5,7,4,8,6,7,8,2]
t = [1,3,4,5,6,7,7,8]

ret = lcs_string(s,t)
print(ret)