# strip(),删除空白字符
# strip(rm)，删除开头结尾的指定字符
def strtoint():# 将字符串表示为数值
    s = "-2a16"
    s = s.strip()
    if len(s)==0:print(0)
    res = 0
    s_list=["1","2","3","4","5","6","7","8","9","0"]
    flag = 1
    if s[0]=="-":
        flag = -1
        s = s[1:]
    elif s[0]=="+":
        s = s[1:]
    if len(s)==0:
        print(0)
    for i in s:
        if i in s_list:
            res = res*10+int(i)
        else:
            res = 0
            break
    print(flag*res)

def isNumeric(s): #判断字符串是否能转化为数值
    s = s.strip()
    n = False  # 观察数字，必须有数字
    e = False  # 观察字符e,前后必须有数字，只能出现0或1次
    d = False  # 观察字符小数点，只能出现0或1次，不能在e后面出现
    l = len(s)  # 字符长度
    for i in range(l):
        # 判断正负号，只能出现在首位和e后面一位
        if s[i] in ("+", "-"):
            if i != 0 and s[i - 1] not in ("e", "E"):
                return False
        # 判断小数点，只能出现0或1次，不能在e后面出现
        elif s[i] == ".":
            if e or d:
                return False
            d = True
        # 判断是否出现过数字
        elif "0" <= s[i] <= "9":
            n = True
        # 判断字符e,前后必须有数字，只能出现0或1次
        elif s[i] in ("e", "E"):
            if e or not n:
                return False
            e = True
            n = False
        else:
            return False
    return n