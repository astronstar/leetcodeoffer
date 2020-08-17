#输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        j=len(nums)-1
        i=0
        while i<j:
            if nums[i]%2==0:
                if nums[j]%2==0:
                    j=j-1
                    continue
                else:
                    nums[i],nums[j]=nums[j],nums[i]
            i=i+1
        return nums
 