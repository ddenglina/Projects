# 假设这里有一些修改
# 出现一次的数字
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

# 出现一次的两个数字
# 分治+异或

# 不用加减乘除做加法
def add(num1,num2):
    while num2 != 0:
        res = (num1 ^ num2) & 0xffffffff  #相加不进位
        carry = ((num1 & num2) << 1) & 0xffffffff # 进位 左移
        num1 = res
        num2 = carry
    if num1 <= 0x7fffffff:
        num1 = num1
    else:
        num1 = ~(num1 ^ 0xffffffff)
    return num1

# 1出现的个数
def NumberOf1( n):
    res = 0
    if n < 0:
        n = n & 0xffffffff #python中独有的，将负数转化为正数

    while n:
        res+=1
        n=n&(n-1)
    return res


# 数值的整数次方
# 快速幂递归Olog（n）
def Power(self, base, exponent):
    # write code here
    if exponent < 0:
        base = 1 / base
        exponent = -exponent
    return self.selfPower(base, exponent)


def selfPower(self, b, n):
    if n == 0: return 1.0
    res = self.selfPower(b, n // 2)
    if n & 1:
        return res * res * b
    else:
        return res * res

# 非递归Olog（n）
def Power(b, n):
    # write code here
    if n < 0:
        b = 1 / b
        n = -n
    res=1
    while n!=0:
        if n&1==1:
            res*=b
        b *=b
        n>>=1
    return res





# res = NumberOf1(10)
# print(res)

# A=[5,2,5,2,3,9,9,20,15,8,8]
# k = 2
# num = singleNumber(A,k)
# print(num)