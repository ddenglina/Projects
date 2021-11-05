
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 二叉搜索树的第K个节点
    res = 0
    def KthNode(self, pRoot, k):
        # write code here
        if not pRoot or k == 0: return
        res = []

        def inorder(pRoot):
            if not pRoot or len(res) >= k:
                return
            inorder(pRoot.left)
            res.append(pRoot)
            inorder(pRoot.right)

        inorder(pRoot)
        if len(res) < k: return None
        return res[k - 1]


class Solution:
    # 树的子结构
    def HasSubtree(self, pRoot1: TreeNode, pRoot2: TreeNode) -> bool:
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        if self.judge(pRoot1, pRoot2):
            return True
        return self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)

    def judge(self, tree, subtree):
        if not subtree: return True
        if not tree: return False
        if tree.val != subtree.val:
            return False
        return self.judge(tree.left, subtree.left) and self.judge(tree.right, subtree.right)

class Solution:
    # 二叉树的镜像
    def Mirror(self , pRoot ):
        # write code here
        if not pRoot:return
        if not pRoot.left and not pRoot.right:return pRoot
        pRoot.left,pRoot.right=pRoot.right,pRoot.left
        self.Mirror(pRoot.left)
        self.Mirror(pRoot.right)
        return pRoot


class Solution:
    # 二叉搜索树的后续遍历序列
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence: return False
        root = sequence[-1]

        for i in range(len(sequence)):
            if sequence[i] >= root:
                break
        for j in range(i, len(sequence)):
            if sequence[j] < root:
                return False

        left = True
        if i > 0:
            left = self.VerifySquenceOfBST(sequence[0:i])
        right = True
        if i < len(sequence) - 1:
            right = self.VerifySquenceOfBST(sequence[i:len(sequence) - 1])
        return left and right

class Solution:
    # 二叉树中和为某一值的路径
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        pathList = []
        res = []
        def PathSum(root,tar):
            if not root:
                return None
            pathList.append(root.val)
            tar -= root.val
            if tar == 0 and not root.left and not root.right:
                res.append(list(pathList))
            PathSum(root.left,tar)
            PathSum(root.right,tar)
            pathList.pop()
            tar += root.val
        PathSum(root,expectNumber)
        return res


class Solution:
    # 二叉搜索树与双向链表
    def Convert(self, pRootOfTree):
        # write code here
        # 中序
        if not pRootOfTree:
            return
        self.head = self.pre = None

        def midTraversal(root):
            if not root:
                return
            midTraversal(root.left)
            if not self.head:
                self.head = self.pre = root
            else:
                self.pre.right = root
                root.left = self.pre
                self.pre = root
            midTraversal(root.right)


class Solution:
    # 二叉树的下一个节点
    def GetNext(self, pNode):
        # write code here
        res = []
        cur = pNode
        while cur.next:  # 找到根节点
            cur = cur.next

        self.inorder(cur, res)

        for i in range(len(res)):
            if pNode == res[i]:
                # 判断 i 是不是最后一个结点，是则返回none，否则返回下一个数组结点
                return None if i == len(res) - 1 else res[i + 1]
        return None

    # 递归遍历中序
    def inorder(self, cur, res):
        if cur:
            self.inorder(cur.left, res)
            res.append(cur)
            self.inorder(cur.right, res)


class Solution:
    # 1、有右子树，下一结点是右子树中的最左结点
    # 2、无右子树，且结点是该结点父结点的左子树，则下一结点是该结点的父结点
    # 3、无右子树，且结点是该结点父结点的右子树，则一直沿着父结点追朔，直到找到某个结点是其父结点的左子树，如果存在这样的结点，那么这个结点的父结点就是我们要找的下一结点，若并没有符合情况的结点，则没有下一结点
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return None
        # 有右子树,右子树的最左叶子节点
        if pNode.right:
            res = pNode.right
            while res.left:
                res = res.left
            return res
        # 没有右子树,当前节点是父节点的左子节点
        while pNode.next:
            tmp = pNode.next
            if tmp.left == pNode:  # 该节点是父节点的左子树
                return tmp
            # 该节点是父节点的右子树，沿着父节点追溯，直到找到某个结点父节点的左子树
            pNode = pNode.next
        return None  # 若没有，则不存在


class Solution:
    # 平衡二叉树
    def IsBalanced_Solution(self, pRoot):
        if not pRoot: return True
        return self.dfs(pRoot) != -1

    def dfs(self, node):
        if not node: return 0
        l = self.dfs(node.left)
        if l == -1: return -1  # 若是-1,提前返回,剪枝
        r = self.dfs(node.right)
        if r == -1: return -1
        if abs(l - r) > 1:
            return -1
        return max(l, r) + 1


class Solution:
    # 把二叉树打印成多行
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot: return []
        q = [pRoot]
        res = []
        while q:
            sz = len(q)
            tmp = []
            for i in range(sz):
                node = q.pop(0)
                tmp.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(tmp)

        return res