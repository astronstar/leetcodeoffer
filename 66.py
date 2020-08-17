# 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

#  

# 示例:

# 输入: [1,2,3,4,5]
# 输出: [120,60,40,30,24]
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        if not a:return []
        left=[1]
        right=[1]
        for i in range(1,len(a)):
            left.append(left[-1]*a[i-1])
            right.insert(0,right[0]*a[len(a)-i])
        #print(right)
        res=[]
        for i in range(len(left)):
            res.append(left[i]*right[i])
        return res