# 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
# 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
# 示例 1：
# 输入：target = 9
# 输出：[[2,3,4],[4,5]]
# 示例 2：
# 输入：target = 15
# 输出：[[1,2,3,4,5],[4,5,6],[7,8]]

class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        nums=[i for i in range(1,target+1)]
        res=[]
        i=target//2
        start=2
        while i>=0 and start<=target//2+1:
            if sum(nums[i:i+start])==target:
                res.append(nums[i:i+start])
                i=i-1
                start+=1
            if sum(nums[i:i+start])>target:
                i=i-1
            if sum(nums[i:i+start])<target:
                start+=1
        return sorted(res)
