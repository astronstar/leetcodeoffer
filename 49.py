# 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
# 示例:

# 输入: n = 10
# 输出: 12
# 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap=[1]
        heapq.heapify(heap)
        i=0
        res=[]
        while i<n:
            cur=heapq.heappop(heap)
            if cur not in res:
                i+=1
                res.append(cur)
                for j in [2,3,5]:
                    
                    x=j*cur
                    heapq.heappush(heap,x)
        return res[-1]