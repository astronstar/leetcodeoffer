#给定一棵二叉搜索树，请找出其中第k大的节点。
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root:return 0
        res=[]
        def dfs(root):
                if root.left:
                    dfs(root.left)
                res.append(root.val)
                if root.right:
                    dfs(root.right)
        if root:        
            dfs(root)
        return res[len(res)-k]