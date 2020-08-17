# 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

# 示例:

# 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
# 输出: [3,3,5,5,6,7] 
# 解释: 

#   滑动窗口的位置                最大值
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q=[]
        res=[]
        for i in range(len(nums)):
            if i-k>=0 and nums[i-k]==q[0]:
                q.pop(0)

            while q and nums[i]>q[-1]:
                q.pop()
            q.append(nums[i])    

            print(q)
            if i>=k-1:
                res.append(q[0])
        return res