
arr = [1,2,4,3,4,3,65,34,21,0]

# 冒泡排序
def bubbleSort(arr):
    N = len(arr)
    while N>1:
        for i,j in zip(range(0,N),range(1,N)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        N -= 1
    return arr

# 快速排序
def quickSort(arr,left,right):
    if left>=right: return arr
    key = arr[left]
    i,j = left,right
    while i<j:
        while i<j and arr[j]>=key:
            j-=1
        arr[i] = arr[j]
        while i<j and arr[i]<=key:
            i+=1
        arr[j] = arr[i]
    arr[i] = key
    quickSort(arr,left,i-1)
    quickSort(arr,i+1,right)
    return arr

# 归并排序
def merge(s1,s2):
    i,j=0,0
    res = []
    while i<len(s1) and j<len(s2):
        if s1[i]<=s2[j]:
            res.append(s1[i])
            i+=1
        else:
            res.append(s2[j])
            j+=1
    res+=s1[i:]
    res+=s2[j:]
    return res

def mergeSort(arr):
    if len(arr)<2: return arr
    n = len(arr)
    mid = n//2
    s1 = mergeSort(arr[0:mid])
    s2 = mergeSort(arr[mid:n])
    return merge(s1,s2)


# 堆排序


# arr = bubbleSort(arr)
# arr = quickSort(arr,0,len(arr)-1)
arr = mergeSort(arr)
print(arr)
