# 请完成一个函数，输入一个二叉树，该函数输出它的镜像。

# 例如输入：

#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# 镜像输出：

#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        def dfs(root):
            if root is None:return []
            root.left,root.right=root.right,root.left
            if root.left:
                dfs(root.left)
            if root.right:
                dfs(root.right)
            return root
        return dfs(root)