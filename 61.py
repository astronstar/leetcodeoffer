# 从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
# 示例 1:

# 输入: [1,2,3,4,5]
# 输出: True
#  

# 示例 2:

# 输入: [0,0,1,2,5]
# 输出: True
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        nums.sort()
        count=0
        # nums for i in nums
        for i in range(len(nums)-1):
            if nums[i]==0:
                count+=1
                continue
            elif nums[i]==nums[i+1]:
                return False
        if nums[-1]-nums[count]<5:
            return True
        else:    
            return False