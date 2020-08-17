# 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
# 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
# 示例 1：
# 输入：target = 9
# 输出：[[2,3,4],[4,5]]
# 示例 2：
# 输入：target = 15
# 输出：[[1,2,3,4,5],[4,5,6],[7,8]]
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i=0
        j=len(nums)-1
        while i<j:
            if nums[i]+nums[j]>target:
                j-=1
            elif nums[i]+nums[j]<target:
                i+=1
            else:
                return [nums[i],nums[j]]
