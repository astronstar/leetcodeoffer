# 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# 例如，给出

# 前序遍历 preorder = [3,9,20,15,7]
# 中序遍历 inorder = [9,3,15,20,7]
# 返回如下的二叉树：

#     3
#    / \
#   9  20
#     /  \
#    15   7

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder)==0:return None
        rootval=preorder[0]
        index=inorder.index(rootval)
        left=inorder[:index]
        right=inorder[index+1:]
        t=TreeNode(-1)
        t.val=rootval
        t.left=self.buildTree(preorder[1:index+1],left)
        t.right=self.buildTree(preorder[index+1:],right)
        return t
