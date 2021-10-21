s = 'bac'

# 字符串全排列无重复
def Permutation1(s,i):
    if i==len(s):
        res.append(''.join(s))
    for j in range(i,len(s)):
            s[i],s[j] = s[j],s[i]
            Permutation1(s,i+1)
            s[i], s[j] = s[j], s[i]

# 字符串全排列有重复
def Permutation2(s,i):
    if i==len(s):
        res.append(''.join(s))
    for j in range(i,len(s)):
        if s[j] not in s[i:j]: # 加一个条件，若是该字符在i到j之间有，就不进行交换
            s[i],s[j] = s[j],s[i]
            Permutation2(s,i+1)
            s[i], s[j] = s[j], s[i]

# 字符串全排列有重复、按字典序
# 需在全排列之后添加一个排序算法
def resort(res):
    for i in range(len(res)): # 简单选择
        for j in range(i+1,len(res)):
            if res[i]>res[j]:
                res[i],res[j]=res[j],res[i]

res = []
s = list(s) # 需要将字符串转为列表
Permutation2(s,0)
resort(res)
print(res)



