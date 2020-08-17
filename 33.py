# 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
# 参考以下这颗二叉搜索树：

#      5
#     / \
#    2   6
#   / \
#  1   3
# 示例 1：

# 输入: [1,6,3,2,5]
# 输出: false
# 示例 2：

# 输入: [1,3,2,6,5]
# 输出: true

class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        if not postorder or len(postorder) == 0:return True
        rootval=postorder[-1]
        n=len(postorder)
        mid=0
        for i in range(n):
            if postorder[i]>rootval:
                break
        mid=i
        for i in range(mid,n-1):
            if postorder[i]<rootval:
                return False
        left=True
        if mid>0:
            left=self.verifyPostorder(postorder[0:mid])
        right=True
        if i<n-1:
            right=self.verifyPostorder(postorder[mid:-1])
        return  left and right