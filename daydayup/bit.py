def singleNumber(arr,k):
    res = 0
    bitsum = [0] * 32
    for i in range(32):
        for a in arr:
            bitsum[i] += (a >> i) & 1

    for i in range(32):
        if bitsum[i] % k == 1:
            res += (1 << i)
    return res

A=[5,2,5,2,3,9,9,20,15,8,8]
k = 2
num = singleNumber(A,k)
print(num)