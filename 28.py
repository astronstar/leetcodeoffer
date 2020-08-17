# 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

# 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
# 但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

#     1
#    / \
#   2   2
#    \   \
#    3    3
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def dfs(L,R):
            if not L and not R:return True
            if not L or not R or L.val!=R.val:return False
            return dfs(L.right,R.left) and dfs(L.left,R.right)

        if not root:return True
        else:
            return dfs(root.left,root.right)