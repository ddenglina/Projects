# 股票买卖的最大利润
# 一次买入卖出：双指针，找股票最低点买入
prices = [1,2,3,0,2]

def oneMaxProfit(prices):
    res = 0
    minp = prices[0]
    for p in prices:
        if p<minp:
            minp = p
        elif p-minp>res:
            res = p-minp
    return res

# 多次买入卖出：赚差价：只要前一天价格比后一天低，就买入卖出
def moreMaxProfit(prices):
    res = 0
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            res += prices[i]-prices[i-1]
    return res

def moreMaxProfit1(prices): # 动态规划解法
    sell = 0
    hold = -prices[0]
    for i in range(1,len(prices)):
        sell = max(sell,hold+prices[i])
        hold = max(hold,sell-prices[i])
    return sell

# 多次买卖+手续费
def moreMaxProfit2(prices):
    sell = 0
    fee = 2
    hold = -prices[0]
    for i in range(1,len(prices)):
        sell = max(sell,hold+prices[i]-fee)
        hold = max(hold,sell-prices[i])
    return sell

# 多次购买+冷冻期
def moreMaxProfit3(prices):
    n = len(prices)
    dp = [[0 for i in range(3)] for j in range(n)]
    dp[0][0]=-prices[0]
    dp[0][1] = 0
    dp[0][2] = 0
    for i in range(1,n):
        # 持仓
        dp[i][0] = max(dp[i-1][0],dp[i-1][2]-prices[i])
        # 空仓冷静
        dp[i][1] = dp[i-1][0]+prices[i]
        # 空仓非冷静
        dp[i][2] = max(dp[i-1][1],dp[i-1][2])
    # 从空仓的两个里面选一个更大的
    res = max(dp[n-1][1],dp[n-1][2])
    return res
# res = oneMaxProfit(prices)
res = moreMaxProfit3(prices)
print(res)

# 可以两次买入卖出
# 可以k次买入卖出