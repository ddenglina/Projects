
# 你有三种硬币，面值分别为2元，5元，7元，每种硬币都足够多，买一本书需要27元。问：如何用最少的硬币组合正好付清，不需要对方找钱。
arr = [1,3,4]
tar = 9

dp= [float("inf") for i in range(tar+1)] #[0,9]
dp[0]=0

for i in range(1,tar+1):
    for j in range(len(arr)):
        if i>=arr[j]:
            dp[i]=min(dp[i],dp[i-arr[j]]+1)

if dp[tar]==float("inf"):
    print(-1)
else:
    print(dp[tar])