# 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
# 示例:
# 给定如下二叉树，以及目标和 sum = 22，

#               5
#              / \
#             4   8
#            /   / \
#           11  13  4
#          /  \    / \
#         7    2  5   1
# 返回:

# [
#    [5,4,11,2],
#    [5,8,4,5]
# ]
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:return []
        res=[]
        def dfs(root,total,t):
            if not root.left and not root.right:
                print(t,sum,total)
                if total==sum:
                    res.append(t)
            #t=t+[root.val]
            if root.left:
                dfs(root.left,total+root.left.val,t+[root.left.val])   
            if root.right:
                dfs(root.right,total+root.right.val,t+[root.right.val]) 

        dfs(root,root.val,[root.val])
        return res