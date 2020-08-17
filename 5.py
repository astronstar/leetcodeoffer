# 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
# 示例 1：

# 输入：s = "We are happy."
# 输出："We%20are%20happy."


class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix)==0:return False
        i=0
        j=len(matrix[0])-1
        while i<len(matrix) and j>=0:
            if matrix[i][j]<target:
                i+=1
            elif matrix[i][j]>target:
                j-=1
            else:
                return True
        return False