# 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
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
#   [20,9],
#   [15,7]
# ]
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        q=[]
        if root:
            q.append([root])
        res=[]
        while q:
            nodelist=q.pop(0)
            
            level=[]
            t=[]
            for node in nodelist:
                t.append(node.val)
                if node.left:
                    level.append(node.left)
                if node.right:
                    level.append(node.right)
            if level:
                q.append(level)
            res.append(t)
        for i in range(len(res)):
            if i%2==1:
                res[i]=res[i][::-1]    
        return res 