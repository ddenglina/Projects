# 最后一个剩下的数字，约瑟夫环问题
def LastRemaining_Solution(self, n: int, m: int) -> int:
    # 递归
    # if n == 0: return -1
    # if n == 1:
    #     return 0
    # else:
    #     return (self.LastRemaining_Solution(n - 1, m) + m) % n

    # 迭代
    if n == 0: return -1
    if n == 1: return 0
    res = 0
    for i in range(2, n + 1):
        res = (res + m) % i      # 这一步是精髓
    return res

# 剪绳子
# 数学
def cutRope(self, number):
    # write code here
    a = number // 3
    if number % 3 == 0:
        return pow(3, a)
    if number % 3 == 1:
        return pow(3, a - 1) * 4
    if number % 3 == 2:
        return pow(3, a) * 2

# 动态规划
def cutRope(self, number: int) -> int:
    # write code here
    # dp[i]标是长度为i的绳子构成乘积的最大值
    dp = [0] * (number + 1)
    dp[2] = 1
    for i in range(3, number + 1):
        for j in range(2, i):
            # 状态转移公式
            dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
    return dp[number]


# 从1到n中1出现的次数
 def NumberOf1Between1AndN_Solution(self, n):
        count=0
        bitNum=1
        high=n//10
        cur=n%10
        low=0
        while cur!=0 or high!=0:  # 以下是精髓
            if cur==0:
                count+=high*bitNum
            elif cur==1:
                count+=high*bitNum+low+1
            else:
                count+=(high+1)*bitNum

            low+=cur*bitNum
            bitNum*=10
            cur=high%10
            high=high//10
        return count