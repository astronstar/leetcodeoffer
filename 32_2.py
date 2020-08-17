# 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
# 例如:
# 给定二叉树: [3,9,20,null,null,15,7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回其层次遍历结果：

# [
#   [3],
#   [9,20],
#   [15,7]
# ]

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        q=[]
        if root:
            q.append(root)
        res=[]
        while q:
            node=q.pop(0)
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

        return res 