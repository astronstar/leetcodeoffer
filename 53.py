# 统计一个数字在排序数组中出现的次数。

#  

# 示例 1:

# 输入: nums = [5,7,7,8,8,10], target = 8
# 输出: 2
# 示例 2:

# 输入: nums = [5,7,7,8,8,10], target = 6
# 输出: 0
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # l,r=0,len(nums)-1
        # while l<r:
        #     mid=(l+r)//2
        #     if nums[mid]>target:
        #         r=mid
        #     elif nums[mid]<target:
        #         l=mid+1
        # res1=l

        # l,r=0,len(nums)-1
        # while l<r:
        #     mid=(l+r)//2
        #     if nums[mid]>target:
        #         r=mid
        #     elif nums[mid]<target:
        #         l=mid+1
        # res2=l
        # return res2-res1+1
        import bisect
        left=bisect.bisect_left(nums,target)
        right=bisect.bisect(nums,target)
        return right-left