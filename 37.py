#请实现两个函数，分别用来序列化和反序列化二叉树。
class Codec:

    def serialize(self, root):
        res=[]
        def dfs(root):
            if not root:
                res.append("#")
                return
            res.append(str(root.val))
            #if root.left:
            dfs(root.left)
            #if root.right:
            dfs(root.right)
        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        d=iter(data.split(","))
        def dfs():
            
            s=next(d)
            if s=="#":return
            node=TreeNode(int(s))
            node.left=dfs()
            node.right=dfs()
            return node
        return dfs()