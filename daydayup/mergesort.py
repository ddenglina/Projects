
count=0

def merge_sort(data):
    global count
    if len(data) <= 1: return data
    n = len(data)
    mid = n // 2
    s1 = merge_sort(data[0:mid])
    s2 = merge_sort(data[mid:n])

    i, j = 0, 0
    res = []
    while i < len(s1) and j < len(s2):
        if s1[i] <= s2[j]:
            res.append(s1[i])
            i += 1
        else:
            res.append(s2[j])
            j += 1

            print("len(s1)",len(s1))
            count += len(s1)-i
            print("count:",count)

    res+=s1[i:]
    res+=s2[j:]
    return res

data = [4,1,2,3]
merge_sort(data)
print(count % 1000000007)

