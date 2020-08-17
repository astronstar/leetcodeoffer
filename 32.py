# 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
# 例如:
# 给定二叉树: [3,9,20,null,null,15,7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回：

# [3,9,20,15,7]
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
