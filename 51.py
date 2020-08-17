# 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
# 示例 1:

# 输入: [7,5,6,4]
# 输出: 5
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        self.cnt=0
        def merge(nums,start,mid,end):
            tmp=[]
            i=start
            j=mid+1
            while i<=mid and j<=end:
                if nums[i]<=nums[j]:
                    tmp.append(nums[i])
                    i+=1
                else:
                    self.cnt+=mid-i+1
                    tmp.append(nums[j])
                    j+=1
            
            while i<=mid:
                tmp.append(nums[i])
                i+=1
            while j<=end:
                tmp.append(nums[j])
                j+=1
            for i in range(len(tmp)):
                nums[start+i]=tmp[i]

        def mergeSort(nums,start,end):
            if start>=end:return
            mid=(start+end)//2
            mergeSort(nums,start,mid)
            mergeSort(nums,mid+1,end)
            merge(nums,start,mid,end)

        mergeSort(nums,0,len(nums)-1)
        return self.cnt